import pathlib
from typing import Union

import anndata as an
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Timer, EarlyStopping

from scsims.lightning_train import DataModule
from scsims.model import SIMSClassifier

here = pathlib.Path(__file__).parent.absolute()

class UnconfiguredModelError(Exception):
    pass

class SIMS:
    def __init__(
        self,
        datafiles: Union[list[str], list[an.AnnData]] = None,
        weights_path: str = None,
        *args,
        **kwargs,
    ) -> None:
        print('Setting up data module ...')
        if weights_path is not None:
            self._model = SIMSClassifier.load_from_checkpoint(weights_path, *args, **kwargs)

        self.datamodule = DataModule(
            datafiles=[datafiles] if isinstance(datafiles, an.AnnData) else datafiles, 
            *args,
            **kwargs,
        )

        for att, value in self.datamodule.__dict__.items():
            setattr(self, att, value)

    def setup_model(self, *args, **kwargs):
        print('Setting up model ...')
        if 'model_size' in kwargs:
            assert kwargs['model_size'] in ["tall", "grande", "venti", "trenta"], "if specified, model_size must be one of 'big', 'medium', or 'small'"
            if kwargs['model_size'] == "trenta":
                kwargs['n_a'] = 128
                kwargs['n_d'] = 128
            if kwargs['model_size'] == "venti":
                kwargs['n_a'] = 64
                kwargs['n_d'] = 64
            if kwargs['model_size'] == "grande":
                kwargs['n_a'] = 32
                kwargs['n_d'] = 32
            if kwargs['model_size'] == "tall":
                kwargs['n_a'] = 8
                kwargs['n_d'] = 8
        self._model = SIMSClassifier(self.datamodule.input_dim, self.datamodule.output_dim, *args, **kwargs)

    def setup_trainer(self, early_stopping_patience: int = 20, *args, **kwargs):
        print('Setting up trainer ...')
        if 'callbacks' in kwargs:
            # check if any of the list of callbacks is a modelcheckpoint or timer
            callbacks = kwargs['callbacks']
            if not any([isinstance(callback, ModelCheckpoint) for callback in callbacks]):
                callbacks.append(ModelCheckpoint())
            if not any([isinstance(callback, Timer) for callback in callbacks]):
                callbacks.append(Timer())
            if not any([isinstance(callback, EarlyStopping) for callback in callbacks]):
                callbacks.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience))
        else:
            kwargs['callbacks'] = [ModelCheckpoint(), Timer(), EarlyStopping(monitor='val_loss', patience=early_stopping_patience)]
        self._trainer = pl.Trainer(*args, **kwargs)

    def train(self, *args, **kwargs):
        print('Beginning training')
        if not hasattr(self, "_trainer"):
            self.setup_trainer(*args, **kwargs)
        if not hasattr(self, "_model"):
            self.setup_model(*args, **kwargs)

        self._trainer.fit(self._model, datamodule=self.datamodule)

        print('Finished training')

    def predict(self, inference_data: an.AnnData, *args, **kwargs):
        if not hasattr(self, '_model'):
            raise UnconfiguredModelError(
                """The model attribute is not configured. This is likely 
                because you are running the predict method after re-initializing the 
                SIMS class. Reinitialize the SIMS class with the weights_path
                pointing to the .ckpt file to continue."""
            )
        results = self._model.predict(inference_data, *args, **kwargs)
        try:
            results = results.apply(lambda x: self.label_encoder.inverse_transform(x))
        except Exception as e:
            if "has no attribute" in str(e):
                print("""
                    Unable to encoder numeric predictions back to class labels, since the original
                    labelfile and class_label column were not passed upon initialization. Alternatively, use 
                    SIMS.decode_predictions([labelfiles]) to convert the numeric labels to string names.
                """)
        self.results = results 

        print('Finished prediction, returning results and storing in results attribute ...')
        return results

    def explain(self, datafiles: an.AnnData, *args, **kwargs):
        print('Computing explainability matrix ...')
        results = self._model.explain(datafiles, *args, **kwargs)

        return results

    def decode_predictions(predictions, labelfiles, class_labels, datafiles, sep=None):
        labelencoder, _ = DataModule.get_unique_targets(labelfiles, sep, class_labels, datafiles)

        return labelencoder.inverse_transform(predictions)
