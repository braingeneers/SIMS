from functools import partial
from typing import Callable, Dict, Union

import anndata as an
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import create_explain_matrix
from scipy.sparse import csc_matrix
from torchmetrics.functional import (accuracy, auroc, f1_score, precision,
                                     recall, specificity)
from torchmetrics.functional.classification.stat_scores import _stat_scores_update
from tqdm import tqdm

from scsims.data import CollateLoader
from scsims.inference import MatrixDatasetWithoutLabels


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
        metrics: Dict[str, Callable] = None,
        scheduler_params: Dict[str, float] = None,
        weights: torch.Tensor = None,
        loss: Callable = None,  # will default to cross_entropy
        pretrained: bool = None,
        no_explain: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Stuff needed for training
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_sparse = lambda_sparse

        self.optim_params = optim_params

        self.weights = weights
        self.loss = loss

        if pretrained is not None:
            self._from_pretrained(**pretrained.get_params())

        if metrics is None:
            self.metrics = aggregate_metrics(num_classes=self.output_dim)
        else:
            self.metrics = metrics

        self.optim_params = (
            optim_params
            if optim_params is not None
            else {
                "optimizer": torch.optim.Adam,
                "lr": 0.001,
                "weight_decay": 0.001,
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

        print(f"Initializing network")
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
        )

        print(f"Initializing explain matrix")
        if not no_explain:
            self.reducing_matrix = create_explain_matrix(
                self.network.input_dim,
                self.network.cat_emb_dim,
                self.network.cat_idxs,
                self.network.post_embed_dim,
            )

    def forward(self, x):
        return self.network(x)

    def _compute_loss(self, y, y_hat):
        # If user doesn't specify, just set to cross_entropy
        if self.loss is None:
            self.loss = F.cross_entropy

        return self.loss(y, y_hat, weight=self.weights)

    def _compute_metrics(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        tag: str,
        on_epoch=True,
        on_step=True,
    ):
        for name, metric in self.metrics.items():
            val = metric(y_hat, y)
            self.log(
                f"{tag}_{name}",
                val,
                on_epoch=on_epoch,
                on_step=on_step,
                logger=True,
            )

    def _step(self, batch, tag):
        x, y = batch
        y_hat, M_loss = self.network(x)

        loss = self._compute_loss(y_hat, y)
        # Add the overall sparsity loss
        loss = loss - self.lambda_sparse * M_loss

        self.log(f"{tag}_loss", loss, logger=True, on_epoch=True, on_step=True)
        self._compute_metrics(y_hat, y, tag)

        tp, fp, _, fn = _stat_scores_update(
            preds=y_hat,
            target=y,
            num_classes=self.output_dim,
            reduce="macro",
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Calculations on step
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _epoch_end(self, step_outputs, tag):
        tps, fps, fns = [], [], []

        for i in range(len(step_outputs)):
            res = step_outputs[i]
            tp, fp, fn = res["tp"], res["fp"], res["fn"]

            tps.append(tp.cpu().numpy())
            fps.append(fp.cpu().numpy())
            fns.append(fn.cpu().numpy())

        tp = np.sum(np.array(tps), axis=0)
        fp = np.sum(np.array(fps), axis=0)
        fn = np.sum(np.array(fns), axis=0)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s = 2 * (precision * recall) / (precision + recall)
        f1s = np.nan_to_num(f1s)

        self.log(
            f"{tag}_median_f1",
            np.nanmedian(f1s),
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        return f1s

    # Calculation on epoch end, for "median F1 score"
    def training_epoch_end(self, step_outputs):
        self._epoch_end(step_outputs, "train")

    def validation_epoch_end(self, step_outputs):
        self._epoch_end(step_outputs, "val")

    def test_epoch_end(self, step_outputs):
        self._epoch_end(step_outputs, "test")

    def configure_optimizers(self):
        if "optimizer" in self.optim_params:
            optimizer = self.optim_params.pop("optimizer")
            optimizer = optimizer(self.parameters(), **self.optim_params)
        else:
            optimizer = torch.optim.Adam(self.parameters(), **self.optim_params)
        print(f"Initializing with {optimizer = }")

        if self.scheduler_params is not None:
            scheduler = self.scheduler_params.pop("scheduler")
            scheduler = scheduler(optimizer, **self.scheduler_params)
            print(f"Initializating with {scheduler = }")

        if self.scheduler_params is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def __parse_data(self, inference_data, batch_size=32, num_workers=4, rows=None, currgenes=None, refgenes=None, **kwargs):
        if isinstance(inference_data, str):
            inference_data = an.read_h5ad(inference_data)

        if isinstance(inference_data, an.AnnData):
            inference_data = MatrixDatasetWithoutLabels(inference_data.X[rows, :] if rows is not None else inference_data.X)

        if not isinstance(inference_data, torch.utils.data.DataLoader):
            inference_data = CollateLoader(
                dataset=inference_data,
                batch_size=batch_size,
                num_workers=num_workers,
                currgenes=currgenes,
                refgenes=refgenes,
                **kwargs,
            )
        
        return inference_data

    def explain(
        self,
        anndata,
        rows=None,
        batch_size=4,
        num_workers=0,
        currgenes=None,
        refgenes=None,
        cache=False,
        normalize=False,
        **kwargs,
    ):
        loader = self.__parse_data(anndata, batch_size=batch_size, num_workers=num_workers, rows=rows, currgenes=currgenes, refgenes=refgenes, **kwargs)

        if cache and self._explain_matrix is not None:
            return self._explain_matrix

        self.network.eval()
        res_explain = []
        labels = []

        for batch_nb, data in enumerate(tqdm(loader)):
            # if we are running this on already labeled pairs and not just for inference
            if isinstance(data, tuple):
                X, label = data
                labels.extend(label.numpy())
            else:
                X = data

            M_explain, masks = self.network.forward_masks(X)
            for key, value in masks.items():
                masks[key] = csc_matrix.dot(value.cpu().detach().numpy(), self.reducing_matrix)

            original_feat_explain = csc_matrix.dot(
                M_explain.cpu().detach().numpy(),
                self.reducing_matrix,
            )

            res_explain.append(original_feat_explain)

            if batch_nb == 0:
                res_masks = masks
            else:
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])

        res_explain = np.vstack(res_explain)

        if normalize:
            res_explain /= np.sum(res_explain, axis=1)[:, None]

        if cache:
            self._explain_matrix = res_explain

        return res_explain, labels

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

    def predict(self, inference_data: Union[str, an.AnnData, np.array], batch_size=32, num_workers=4, rows=None, currgenes=None, refgenes=None, **kwargs):
        """Does inference on data

        :param inference_data: Anndata, torch Dataset, or torch DataLoader object to do inference on
        """
        loader = self.__parse_data(inference_data, batch_size=batch_size, num_workers=num_workers, rows=rows, currgenes=currgenes, refgenes=refgenes, **kwargs)

        preds = []
        labels = []
        prev_network_state = self.network.training
        self.network.eval()
        with torch.no_grad():
            for X in tqdm(loader):
                # Some dataloaders will have labels, handle this case
                if len(X) == 2:
                    data, label = X
                    labels.extend(label.numpy())
                else:
                    data = X

                res, _ = self(data)
                _, top_preds = res.topk(3, axis=1)  # to get indices
                preds.extend(top_preds.numpy())

        final = pd.DataFrame(preds)
        final = final.rename(
            {
                0: "first_prob",
                1: "second_prob",
                2: "third_prob",
            },
            axis=1,
        )

        if hasattr(self, "datamodule") and hasattr(self.datamodule, "label_encoder"):
            encoder = self.datamodule.label_encoder
            final = final.apply(lambda x: encoder.inverse_transform(x))

        if labels != []:
            final["actual_label"] = labels

        # if network was in training mode before inference, set it back to that
        if prev_network_state:
            self.network.train()

        return final


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


def aggregate_metrics(num_classes) -> Dict[str, Callable]:
    metrics = {
        # Accuracies
        "micro_accuracy": accuracy,
        "macro_accuracy": partial(accuracy, num_classes=num_classes, average="macro"),
        "weighted_accuracy": partial(accuracy, num_classes=num_classes, average="weighted"),
        # Precision, recall and f1s, all macro weighted
        "precision": partial(precision, num_classes=num_classes, average="macro"),
        "recall": partial(recall, num_classes=num_classes, average="macro"),
        "f1": partial(f1_score, num_classes=num_classes, average="macro"),
        # Random stuff I might want
        "specificity": partial(specificity, num_classes=num_classes, average="macro"),
        # 'confusion_matrix': partial(confusion_matrix, num_classes=num_classes),
        "auroc": partial(auroc, num_classes=num_classes, average="macro"),
    }

    return metrics
