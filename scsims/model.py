from functools import partial
from typing import Any, Callable, Dict, Union

import os
import anndata as an
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import create_explain_matrix
from scipy.sparse import csc_matrix
from torchmetrics.functional.classification.stat_scores import _stat_scores_update
from tqdm import tqdm
import torch.utils.data

from scsims.data import CollateLoader
from scsims.inference import DatasetForInference
from scsims.temperature_scaling import _ECELoss
from torchmetrics import Accuracy, F1Score, Precision, Recall, Specificity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        label_encoder: Callable = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.genes = genes
        self.cells = cells
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

        self.metrics = {
            "train": {x: y.to(device) for x, y in aggregate_metrics(num_classes=self.output_dim).items()},
            "val": {x: y.to(device) for x, y in aggregate_metrics(num_classes=self.output_dim).items()},
        }

        self.optim_params = (
            optim_params
            if optim_params is not None
            else {
                "optimizer": torch.optim.Adam,
                "lr": 3e-4,
                "weight_decay": 1e-8,
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

        self._inference_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits, M_loss = self.network(x)
        # temp scaling will be 1 so logits wont change until model is calibrated
        return self.temperature_scale(logits), M_loss

    def _step(self, batch, tag):
        x, y = batch
        logits, M_loss = self.network(x)

        loss = self.loss(logits, y, weight=self.weights)
        loss = loss - self.lambda_sparse * M_loss

        # take softmax for metrics
        probs = logits.softmax(dim=-1)

        # if binary, probs will be (batch, 2), so take second column
        if probs.shape[-1] == 2:
            probs = probs[:, 1]

        tp, fp, _, fn = _stat_scores_update(
            preds=logits,
            target=y,
            num_classes=self.output_dim,
            reduce="macro",
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "probs": probs,
        }

    # Calculations on step
    def training_step(self, batch, batch_idx):
        results = self._step(batch, "train")
        self.log(f"train_loss", results["loss"], on_epoch=True, on_step=True)
        for name, metric in self.metrics["train"].items():
            value = metric(results["probs"], batch[1])
            self.log(f"train_{name}", value=value)

        return results["loss"]

    def on_train_epoch_end(self) -> None:
        for name, metric in self.metrics["train"].items():
            value = metric.compute()
            self.log(f"train_{name}", value=value)
            metric.reset() # inplace

    def validation_step(self, batch, batch_idx):
        results = self._step(batch, "val")
        self.log(f"val_loss", results["loss"], on_epoch=True, on_step=True)
        for name, metric in self.metrics["val"].items():
            value = metric(results["probs"], batch[1])
            self.log(f"val_{name}", value=value)

        return results["loss"]

    def on_validation_epoch_end(self) -> None:
        for name, metric in self.metrics["val"].items():
            value = metric.compute()
            self.log(f"val_{name}", value=value)
            metric.reset()
    
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
        inference_genes = list(inference_data.var_names)
        training_genes = list(self.genes)

        # more inference genes than training genes
        assert len(set(inference_genes).intersection(set(training_genes))) > 0, "inference data shares zero genes with training data, double check the string formats and gene names"
        if len(inference_genes) - len(training_genes) > 0:
            inference_data = inference_data[:, training_genes].copy()
        else:
            diff = list(set(training_genes) - set(inference_genes))
            print(f"Inference data has {len(diff)} less genes than training; performing zero inflation.")

            zero_inflation = an.AnnData(X=np.zeros((inference_data.shape[0], len(diff))), obs=inference_data.obs)
            zero_inflation.var_names = diff
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
                X = data

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

    def predict(self, inference_data: Union[str, an.AnnData, np.array], batch_size=32, num_workers=4, rows=None, currgenes=None, refgenes=None, **kwargs):
        print("parsing data...")
        loader = self._parse_data(
            inference_data,
            batch_size=batch_size,
            num_workers=num_workers,
            rows=rows,
            currgenes=currgenes,
            refgenes=refgenes,
            **kwargs
        )

        # initialize arrays in memory and fill with nans to start
        # this makes it easier to see bugs/wrong predictions than filling zeros
        preds = np.empty((len(loader.dataset), 3))
        preds[:] = np.nan

        all_labels = np.empty(len(loader.dataset))
        all_labels[:] = np.nan

        # save probs 
        probs = np.empty((len(loader.dataset), 3))
        probs[:] = np.nan

        prev_network_state = self.network.training

        # batch size might differ if user passes in a dataloader
        batch_size = loader.batch_size
        for idx, X in enumerate(tqdm(loader)):
            # Some dataloaders will have all_labels, handle this case
            top_probs, top_preds, label = self.predict_step(batch=X, batch_idx=idx)
            all_labels[idx * batch_size : (idx + 1) * batch_size] = label
            preds[idx * batch_size : (idx + 1) * batch_size] = top_preds
            probs[idx * batch_size : (idx + 1) * batch_size] = top_probs

        preds = pd.DataFrame(preds).astype(int)
        preds = preds.rename(
            {
                0: "first_pred",
                1: "second_pred",
                2: "third_pred",
            },
            axis=1,
        )

        preds = preds.apply(lambda x: self.label_encoder.inverse_transform(x))

        probs = pd.DataFrame(probs)
        probs = probs.rename(
            {
                0: "first_prob",
                1: "second_prob",
                2: "third_prob",
            },
            axis=1,
        )

        final = pd.concat([preds, probs], axis=1)

        if not np.all(np.isnan(all_labels)):
            final["label"] = all_labels


        # if network was in training mode before inference, set it back to that
        if prev_network_state:
            self.network.train()

        return final

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if len(batch) == 2:
            data, label = batch
        else:
            data, label = batch, None
        data = data.float()
        res = self(data)[0]
        probs, top_preds = res.topk(3, axis=1)  # to get indices
        probs = probs.softmax(dim=-1)

        return probs.detach().cpu().numpy(), top_preds.detach().cpu().numpy(), label
 
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


def aggregate_metrics(num_classes) -> Dict[str, Callable]:
    task = "binary" if num_classes == 2 else "multiclass"
    num_classes = None if num_classes == 2 else num_classes

    metrics = {
        "micro_accuracy": Accuracy(task=task, num_classes=num_classes, average="micro"),
        "macro_accuracy": Accuracy(task=task, num_classes=num_classes, average="macro"),
        "weighted_accuracy": Accuracy(task=task, num_classes=num_classes, average="weighted"),
        "precision": Precision(task=task, num_classes=num_classes, average="macro"),
        "recall": Recall(task=task, num_classes=num_classes, average="macro"),
        "f1": F1Score(task=task, num_classes=num_classes, average="macro"),
        "specificity": Specificity(task=task, num_classes=num_classes, average="macro"),
    }

    return metrics



