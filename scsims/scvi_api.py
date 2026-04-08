import pathlib
from typing import Union

import anndata as an
import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from sklearn.preprocessing import LabelEncoder

from scsims.lightning_train import DataModule
from scsims.model import SIMSClassifier
from scsims.pretraining import SIMSPretrainer, transfer_pretrained_weights

here = pathlib.Path(__file__).parent.absolute()


def _load_sims_checkpoint(weights_path: str, **kwargs):
    """Load a SIMS checkpoint, tolerating legacy artifacts.

    SIMS checkpoints serialize a ``LabelEncoder`` and numpy arrays inside
    Lightning hyperparameters. As of torch >= 2.6, ``torch.load`` defaults
    to ``weights_only=True`` and rejects those globals. SIMS checkpoints
    are trusted artifacts (produced by the user's own training run, or
    the curated bundle shipped with sims_app), so it's appropriate to
    opt out of weights-only mode here. We do this in exactly one place
    so the rest of the application keeps the safer torch.load default.
    """
    # Caller may already have set weights_only/strict explicitly; respect it.
    kwargs.setdefault("weights_only", False)
    kwargs.setdefault("strict", False)
    return SIMSClassifier.load_from_checkpoint(weights_path, **kwargs)

class UnconfiguredModelError(Exception):
    pass

class SIMS:
    def __init__(
        self,
        *,
        weights_path: str = None,
        data: an.AnnData = None,
        class_label: str = None,
        **kwargs,
    ) -> None:
        if weights_path is not None:
            self.model = _load_sims_checkpoint(weights_path, **kwargs)

        self.datamodule = DataModule(data=data, class_label=class_label, **kwargs)

        for att, value in self.datamodule.__dict__.items():
            setattr(self, att, value)

    # Map of model_size string -> (n_d, n_a) for the underlying TabNet.
    _MODEL_SIZE_PRESETS = {
        "tall": (8, 8),
        "grande": (32, 32),
        "venti": (64, 64),
        "trenta": (128, 128),
    }

    def setup_model(self, *args, **kwargs):
        print("Setting up model ...")
        model_size = kwargs.pop("model_size", None)
        if model_size is not None:
            if model_size not in self._MODEL_SIZE_PRESETS:
                raise ValueError(
                    f"model_size must be one of {sorted(self._MODEL_SIZE_PRESETS)}, "
                    f"got {model_size!r}"
                )
            n_d, n_a = self._MODEL_SIZE_PRESETS[model_size]
            kwargs["n_d"] = n_d
            kwargs["n_a"] = n_a

        self.model = SIMSClassifier(
            input_dim=self.datamodule.input_dim, 
            output_dim=self.datamodule.output_dim, 
            genes=self.datamodule.genes,
            cells=self.datamodule.cells,
            label_encoder=self.datamodule.label_encoder,
            *args, 
            **kwargs
        )

    def setup_trainer(
        self,
        early_stopping_patience: int = None,
        checkpoint_dir: str = "./sims_checkpoints",
        monitor: str = "val_loss",
        *args,
        **kwargs,
    ):
        """Build a Lightning ``Trainer`` with sensible default callbacks.

        ``ModelCheckpoint`` and ``Timer`` are added automatically if the
        caller didn't provide them; ``EarlyStopping`` is added when
        ``early_stopping_patience`` is set.
        """
        print("Setting up trainer ...")
        callbacks = list(kwargs.pop("callbacks", []))

        if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
            callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir))
        if not any(isinstance(cb, Timer) for cb in callbacks):
            callbacks.append(Timer())
        if early_stopping_patience is not None and not any(
            isinstance(cb, EarlyStopping) for cb in callbacks
        ):
            callbacks.append(EarlyStopping(monitor=monitor, patience=early_stopping_patience))

        kwargs["callbacks"] = callbacks
        self._trainer = pl.Trainer(*args, **kwargs)

    # ------------------------------------------------------------------
    # Self-supervised pretraining
    # ------------------------------------------------------------------

    def pretrain(
        self,
        pretraining_ratio: float = 0.2,
        checkpoint_dir: str = "./sims_pretrain_checkpoints",
        *args,
        **kwargs,
    ):
        """Run TabNet self-supervised pretraining on the SIMS DataModule.

        Parameters
        ----------
        pretraining_ratio:
            Fraction of features to obfuscate per training example. The
            model is asked to reconstruct these obfuscated features from
            the rest of the input. ``0.2`` is the default from the TabNet
            paper.
        checkpoint_dir:
            Where to save pretrainer ``.ckpt`` files.
        *args, **kwargs:
            Passed to :class:`lightning.pytorch.Trainer` (``max_epochs``,
            ``accelerator``, ``devices``, ``callbacks``, etc.). Note that
            this is *separate* from the supervised ``setup_trainer`` /
            ``train`` lifecycle and uses its own checkpoint directory.

        After this returns, the pretrained encoder lives on
        ``self.pretrainer`` and will be used to warm-start the next
        :meth:`train` call.
        """
        print("Beginning self-supervised pretraining")

        # Pull out arguments meant for the pretrainer model itself, leaving
        # the rest as Trainer kwargs.
        pretrainer_kwargs = {}
        for k in (
            "n_d", "n_a", "n_steps", "gamma", "cat_idxs", "cat_dims",
            "cat_emb_dim", "n_independent", "n_shared", "epsilon",
            "virtual_batch_size", "momentum", "mask_type",
            "n_shared_decoder", "n_indep_decoder", "grouped_features",
            "optim_params", "scheduler_params",
        ):
            if k in kwargs:
                pretrainer_kwargs[k] = kwargs.pop(k)

        self.pretrainer = SIMSPretrainer(
            input_dim=self.datamodule.input_dim,
            pretraining_ratio=pretraining_ratio,
            genes=self.datamodule.genes,
            **pretrainer_kwargs,
        )

        # Build a Trainer specifically for pretraining. Don't reuse
        # self._trainer because we want a separate checkpoint dir and
        # the user typically pretrains for far more epochs than fine-tunes.
        callbacks = list(kwargs.pop("callbacks", []))
        if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
            callbacks.append(
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    monitor="val_pretrain_loss",
                    mode="min",
                    save_top_k=1,
                )
            )
        if not any(isinstance(cb, Timer) for cb in callbacks):
            callbacks.append(Timer())
        kwargs["callbacks"] = callbacks

        pretrain_trainer = pl.Trainer(*args, **kwargs)
        pretrain_trainer.fit(self.pretrainer, datamodule=self.datamodule)

        print("Finished self-supervised pretraining")
        return self.pretrainer

    def load_pretrainer(self, weights_path: str, **kwargs) -> SIMSPretrainer:
        """Load a previously-saved :class:`SIMSPretrainer` checkpoint.

        Useful for two-stage workflows where pretraining and fine-tuning
        run in separate Python processes.
        """
        kwargs.setdefault("weights_only", False)
        kwargs.setdefault("strict", False)
        self.pretrainer = SIMSPretrainer.load_from_checkpoint(weights_path, **kwargs)
        return self.pretrainer

    # ------------------------------------------------------------------
    # Supervised training
    # ------------------------------------------------------------------

    def train(self, *args, **kwargs):
        if self.datamodule.class_label is None:
            raise ValueError(
                "SIMS.train() is supervised and requires a class_label. Build "
                "the SIMS instance with `SIMS(data=adata, class_label=...)` "
                "before calling .train(). For unsupervised pretraining on an "
                "unlabeled dataset, use SIMS(data=adata).pretrain(...) instead."
            )
        print("Beginning training")
        if not hasattr(self, "_trainer"):
            self.setup_trainer(*args, **kwargs)
        if not hasattr(self, "model"):
            self.setup_model(*args, **kwargs)

        # If we already ran pretraining (or loaded a pretrainer checkpoint),
        # warm-start the supervised classifier from those encoder weights.
        if hasattr(self, "pretrainer") and self.pretrainer is not None:
            n_transferred = transfer_pretrained_weights(self.model, self.pretrainer)
            print(
                f"Warm-started supervised classifier from pretrained encoder "
                f"({n_transferred} tensors transferred)"
            )

        self._trainer.fit(self.model, datamodule=self.datamodule)
        print("Finished training")

    def predict(self, inference_data: an.AnnData, *args, **kwargs):
        if not hasattr(self, 'model'):
            raise UnconfiguredModelError(
                """The model attribute is not configured. This is likely 
                because you are running the predict method after re-initializing the 
                SIMS class. Reinitialize the SIMS class with the weights_path
                pointing to the .ckpt file to continue."""
            )
        self.results = self.model.predict(inference_data, *args, **kwargs)
        return self.results

    def explain(self, datafiles: an.AnnData, labelfile=None, class_label=None, *args, **kwargs):
        print('Computing explainability matrix ...')
        results = self.model.explain(datafiles, *args, **kwargs)

        return results

    def decode_predictions(self, labelfile: str, class_label: str):
        labels = pd.read_csv(labelfile)
        label_encoder = LabelEncoder()
        label_encoder.fit(labels[class_label])
        results = self.results
        try:
            pred_columns = [col for col in results.columns if 'pred' in col]
            results[pred_columns] = results[pred_columns].apply(lambda x: label_encoder.inverse_transform(x))
            #results = self.results.apply(lambda x: label_encoder.inverse_transform(x))
        except AttributeError:
            raise AttributeError(
                """The results attribute is not configured. This is likely 
                because you are running the decode_predictions method before running the predict method.
                Run the predict method first, then run the decode_predictions method."""
            )
        return results

    def set_temperature(self, dataloader):
        self.model = self.model.set_temperature(dataloader)