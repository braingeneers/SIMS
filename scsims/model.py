import os
from typing import Any, Callable, Dict, Union

import anndata as an
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import create_explain_matrix, create_group_matrix
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.preprocessing import LabelEncoder
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall, Specificity
from torchmetrics.functional import stat_scores
from tqdm import tqdm

from scsims.data import CollateLoader
from scsims.inference import DatasetForInference
from scsims.temperature_scaling import _ECELoss

class SIMSClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        lambda_sparse=1e-3,
        optim_params: Dict[str, float] = None,
        scheduler_params: Dict[str, float] = None,
        weights: torch.Tensor = None,
        loss: Callable = None,  # will default to cross_entropy
        pretrained: bool = None,
        no_explain: bool = False,
        genes: list[str] = None,
        cells: list[str] = None,
        label_encoder: LabelEncoder = None,
        grouped_features: list[list[int]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.genes = genes
        self.label_encoder = label_encoder

        # Stuff needed for training
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_sparse = lambda_sparse
        self.optim_params = optim_params
        self.weights = weights
        self.loss = loss

        if self.loss is None:
            self.loss = F.cross_entropy

        if pretrained is not None:
            self._from_pretrained(**pretrained.get_params())

        # MetricCollections are registered as submodules so Lightning moves them
        # to the right device automatically. No more manual .to(device).
        base_metrics = aggregate_metrics(num_classes=self.output_dim)
        self.train_metrics = MetricCollection(base_metrics, prefix="train_")
        self.val_metrics = MetricCollection(
            {k: v.clone() for k, v in base_metrics.items()},
            prefix="val_",
        )

        self.optim_params = (
            optim_params
            if optim_params is not None
            else {
                "optimizer": torch.optim.Adam,
                "lr": 0.01,
                "weight_decay": 0.01,
            }
        )

        self.scheduler_params = (
            scheduler_params
            if scheduler_params is not None
            else {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "factor": 0.75,  # Reduce LR by 25% on plateau
            }
        )

        # pytorch-tabnet >= 4.0 added a `group_attention_matrix` parameter and
        # the default `[]` triggers a crash inside EmbeddingGenerator. Build
        # the identity-style matrix here so each gene is its own group when
        # the user doesn't supply explicit feature groupings.
        group_attention_matrix = create_group_matrix(
            grouped_features if grouped_features is not None else [],
            input_dim,
        )

        print("Initializing network")
        self.network = TabNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=group_attention_matrix,
        )

        print(f"Initializing explain matrix")
        if not no_explain:
            self.reducing_matrix = create_explain_matrix(
                self.network.input_dim,
                self.network.cat_emb_dim,
                self.network.cat_idxs,
                self.network.post_embed_dim,
            )

        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits, M_loss = self.network(x)
        # temp scaling will be 1 so logits wont change until model is calibrated
        return self.temperature_scale(logits), M_loss

    def _shared_step(self, batch):
        x, y = batch
        logits, M_loss = self.network(x)

        loss = self.loss(logits, y, weight=self.weights)
        loss = loss - self.lambda_sparse * M_loss

        # Take softmax for metrics. For binary tasks, torchmetrics expects the
        # positive-class probability vector.
        probs = logits.softmax(dim=-1)
        if probs.shape[-1] == 2:
            probs = probs[:, 1]

        return loss, probs, y

    def training_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        # MetricCollection handles per-step accumulation; .compute() at epoch end.
        self.train_metrics.update(probs, y)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.val_metrics.update(probs, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def configure_optimizers(self):
        # Copy so configure_optimizers() can be called more than once safely
        # (e.g. when re-fitting a model in the same process). Previously the
        # `pop()` calls mutated `self.optim_params` and broke the second call.
        optim_params = dict(self.optim_params)
        optimizer_cls = optim_params.pop("optimizer", torch.optim.Adam)
        optimizer = optimizer_cls(self.parameters(), **optim_params)
        print(f"Initializing with {optimizer = }")

        if self.scheduler_params is None:
            return optimizer

        scheduler_params = dict(self.scheduler_params)
        scheduler_cls = scheduler_params.pop("scheduler")
        scheduler = scheduler_cls(optimizer, **scheduler_params)
        print(f"Initializing with {scheduler = }")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def _parse_data(
            self,
            inference_data,
            batch_size=64,
            num_workers=os.cpu_count(),
            rows=None,
            **kwargs
        ) -> torch.utils.data.DataLoader:

        if isinstance(inference_data, str):
            inference_data = an.read_h5ad(inference_data)
        
        # handle zero inflation or deletion
        #TODO: Increase speed and memory consumption of zero inflation
        inference_genes = list(inference_data.var_names)
        training_genes = list(self.genes)

        # more inference genes than training genes
        assert len(set(inference_genes).intersection(set(training_genes))) > 0, "inference data shares zero genes with training data, double check the string formats and gene names"

        left_genes = list(set(inference_genes) - set(training_genes))  # genes in inference that aren't in training
        right_genes = list(set(training_genes) - set(inference_genes))  # genes in training that aren't in inference 
        intersection_genes = list(set(inference_genes).intersection(set(training_genes))) # genes in both

        if len(left_genes) > 0:
            inference_data = inference_data[:, intersection_genes].copy()
        if len(right_genes) > 0:
            print(f"Inference data has {len(right_genes)} less genes than training; performing zero inflation.")

            #zero_inflation = an.AnnData(X=np.zeros((inference_data.shape[0], len(right_genes))), obs=inference_data.obs)
            zero_inflation = an.AnnData(X= csr_matrix((inference_data.shape[0], len(right_genes))),obs=inference_data.obs)
            zero_inflation.var_names = right_genes
            inference_data = an.concat([zero_inflation, inference_data], axis=1)

        # now make sure the columns are the correct order
        inference_data = inference_data[:, training_genes].copy()

        if isinstance(inference_data, an.AnnData):
            inference_data = DatasetForInference(inference_data.X[rows, :] if rows is not None else inference_data.X)

        if not isinstance(inference_data, torch.utils.data.DataLoader):
            inference_data = CollateLoader(
                dataset=inference_data,
                batch_size=batch_size,
                num_workers=num_workers,
                **kwargs,
            )

        return inference_data

    def explain(
        self,
        anndata,
        rows=None,
        batch_size=64,
        num_workers=os.cpu_count(),
        currgenes=None,
        refgenes=None,
        normalize=False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        loader = self._parse_data(anndata, batch_size=batch_size, num_workers=num_workers, rows=rows, currgenes=currgenes, refgenes=refgenes, **kwargs)

        self.network.eval()
        res_explain = []

        all_labels = np.empty(len(loader.dataset))
        all_labels[:] = np.nan

        for batch_nb, data in enumerate(tqdm(loader)):
            # if we are running this on already labeled pairs and not just for inference
            if isinstance(data, tuple):
                X, label = data
                all_labels[batch_nb * batch_size : (batch_nb + 1) * batch_size] = label
            else:
                X = data.float()

            M_explain, masks = self.network.forward_masks(X)
            for key, value in masks.items():
                masks[key] = csc_matrix.dot(
                    value.cpu().detach().numpy(), self.reducing_matrix
                )

            original_feat_explain = csc_matrix.dot(M_explain.cpu().detach().numpy(),
                                                   self.reducing_matrix)
            res_explain.append(original_feat_explain)

            if batch_nb == 0:
                res_masks = masks
            else:
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])

        res_explain = np.vstack(res_explain)

        if normalize:
            res_explain /= np.sum(res_explain, axis=1)[:, None]

        return res_explain, all_labels

    def _compute_feature_importances(self, dataloader):
        M_explain, _ = self.explain(dataloader, normalize=False)
        sum_explain = M_explain.sum(axis=0)
        feature_importances_ = sum_explain / np.sum(sum_explain)

        return feature_importances_

    def feature_importances(self, dataloader, cache=False):
        if cache and self._feature_importances is not None:
            return self._feature_importances
        else:
            f = self._compute_feature_importances(dataloader)
            if cache:
                self._feature_importances = f
            return f

    def predict(
        self,
        inference_data: Union[str, an.AnnData, np.ndarray],
        batch_size: int = 32,
        num_workers: int = 4,
        top_k: int = 3,
        rows=None,
        currgenes=None,
        refgenes=None,
        **kwargs,
    ) -> pd.DataFrame:
        """Run inference and return the top-``top_k`` predicted labels per cell.

        Parameters
        ----------
        inference_data:
            An :class:`anndata.AnnData`, the path to an ``.h5ad`` file, or a
            raw numpy array of shape ``(n_cells, n_genes)``.
        batch_size:
            Inference batch size.
        num_workers:
            DataLoader worker count.
        top_k:
            Number of ranked predictions to return per cell. Capped at the
            number of training classes. Defaults to ``3``.
        rows, currgenes, refgenes:
            Forwarded to :meth:`_parse_data` for partial-row and gene-alignment
            workflows.

        Returns
        -------
        pandas.DataFrame
            One row per cell with columns ``pred_0`` … ``pred_{top_k-1}`` (the
            decoded label strings) and ``prob_0`` … ``prob_{top_k-1}`` (the
            corresponding softmax probabilities).
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        effective_top_k = min(top_k, len(self.label_encoder.classes_))

        print("Parsing inference data...")
        loader = self._parse_data(
            inference_data,
            batch_size=batch_size,
            num_workers=num_workers,
            rows=rows,
            currgenes=currgenes,
            refgenes=refgenes,
            **kwargs,
        )

        n_rows = len(loader.dataset)
        preds = np.full((n_rows, effective_top_k), np.nan)
        probs = np.full((n_rows, effective_top_k), np.nan)
        all_labels = np.full(n_rows, np.nan)

        prev_network_state = self.network.training
        self.network.eval()

        # batch size might differ if user passes in a dataloader
        loader_batch_size = loader.batch_size
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(loader)):
                top_probs, top_preds, label = self._inference_batch(
                    batch, top_k=effective_top_k
                )
                start = idx * loader_batch_size
                stop = start + top_preds.shape[0]
                preds[start:stop] = top_preds
                probs[start:stop] = top_probs
                if label is not None:
                    all_labels[start:stop] = label

        # restore previous training mode
        if prev_network_state:
            self.network.train()

        preds_df = pd.DataFrame(
            preds.astype(int),
            columns=[f"pred_{i}" for i in range(effective_top_k)],
        )
        preds_df = preds_df.apply(lambda c: self.label_encoder.inverse_transform(c))

        probs_df = pd.DataFrame(
            probs,
            columns=[f"prob_{i}" for i in range(effective_top_k)],
        )

        final = pd.concat([preds_df, probs_df], axis=1)
        if not np.all(np.isnan(all_labels)):
            final["label"] = all_labels
        return final

    def _inference_batch(self, batch, top_k: int):
        """Run a single inference batch and return ``(probs, preds, labels)``.

        Internal helper used by :meth:`predict`. Kept separate from
        :meth:`predict_step` (which is the Lightning hook) so that ``top_k``
        can be passed in without conflicting with Lightning's fixed signature.
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, label = batch
            label = label.detach().cpu().numpy() if torch.is_tensor(label) else label
        else:
            data, label = batch, None
        data = data.float()
        logits = self(data)[0]
        probs, preds = logits.topk(top_k, dim=1)
        probs = probs.softmax(dim=-1)
        return (
            probs.detach().cpu().numpy(),
            preds.detach().cpu().numpy(),
            label,
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Lightning ``predict_step`` hook. Returns top-3 by default."""
        top_k = min(3, len(self.label_encoder.classes_))
        return self._inference_batch(batch, top_k=top_k)
 
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))  # (Batch, Classes)
        return logits / temperature  # (Batch, Classes)

    def set_temperature(self, dataloader, max_iter=50, lr=0.01):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL.
        dataloader (DataLoader): validation set loader
        """
        nll_criterion = torch.nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        print("Setting temperature ...")
        with torch.no_grad():
            for data, label in tqdm(dataloader):
                logits = self(data)[0]
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list) # (num_samples*batch_size, num_classes)
            labels = torch.cat(labels_list)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

def confusion_matrix(model, dataloader, num_classes):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(tqdm(dataloader)):
            outputs, _ = model(inputs)

            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix


def median_f1(tps, fps, fns):
    precisions = tps / (tps + fps)
    recalls = tps / (tps + fns)

    f1s = 2 * (np.dot(precisions, recalls)) / (precisions + recalls)

    return np.nanmedian(f1s)


def aggregate_metrics(num_classes: int) -> Dict[str, Callable]:
    """Build the standard scsims metric dict for either binary or multiclass tasks.

    torchmetrics >= 1.0 rejects ``average=`` for binary tasks, so we branch on
    the task type and only pass the kwargs that are valid for each.
    """
    if num_classes == 2:
        return {
            "accuracy": Accuracy(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
            "f1": F1Score(task="binary"),
            "specificity": Specificity(task="binary"),
        }

    common = {"task": "multiclass", "num_classes": num_classes}
    return {
        "micro_accuracy": Accuracy(average="micro", **common),
        "macro_accuracy": Accuracy(average="macro", **common),
        "weighted_accuracy": Accuracy(average="weighted", **common),
        "precision": Precision(average="macro", **common),
        "recall": Recall(average="macro", **common),
        "f1": F1Score(average="macro", **common),
        "specificity": Specificity(average="macro", **common),
    }



