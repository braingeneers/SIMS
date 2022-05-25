
import shutil 
import json 
import zipfile 
import io 
import copy
import warnings
from pathlib import Path 
from typing import (
    Dict, 
    Callable,
)
from functools import partial

import torch 
import numpy as np 
import torchmetrics 

import pytorch_lightning as pl 
from scipy.sparse import csc_matrix 
from pytorch_tabnet.utils import (
    create_explain_matrix,
    ComplexEncoder,
)
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNet
from torchmetrics.functional.classification.stat_scores import _stat_scores_update
from tqdm import tqdm 

from .metrics import aggregate_metrics

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
        lambda_sparse = 1e-3,
        optim_params: Dict[str, float]=None,
        metrics: Dict[str, Callable]=None,
        scheduler_params: Dict[str, float]=None,
        weights: torch.Tensor=None,
        loss: Callable=None, # will default to cross_entropy
        pretrained: bool=None,
        no_explain: bool=False,
    ) -> None:
        super().__init__()

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

        if optim_params is None:
            self.optim_params = {
                'optimizer': torch.optim.Adam,
                'lr': 0.001,
                'weight_decay': 0.001,
            }
        else:
            self.optim_params = optim_params

        if scheduler_params is None:
            self.scheduler_params={
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
                'factor': 0.75, # Reduce LR by 25% on plateau
            }
        else:
            self.scheduler_params = scheduler_params

        print(f'Initializing network')
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

        print(f'Initializing explain matrix')
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

    def _compute_metrics(self, 
        y_hat: torch.Tensor, 
        y: torch.Tensor, 
        tag: str,
        on_epoch=True, 
        on_step=False,
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

    def _step(self, batch):
        x, y = batch
        y_hat, M_loss = self.network(x)

        loss = self._compute_loss(y_hat, y)

        # Add the overall sparsity loss
        loss = loss - self.lambda_sparse * M_loss
        return y, y_hat, loss


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
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')
    
    def _epoch_end(self, step_outputs, tag):
        tps, fps, fns = [], [], []
        
        for i in range(len(step_outputs)):
            res = step_outputs[i]
            tp, fp, fn = res['tp'], res['fp'], res['fn']
                
            tps.append(tp.cpu().numpy())
            fps.append(fp.cpu().numpy())
            fns.append(fn.cpu().numpy())
            
        tp = np.sum(np.array(tps), axis=0)
        fp = np.sum(np.array(fps), axis=0)
        fn = np.sum(np.array(fns), axis=0)
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s = 2*(precision * recall) / (precision + recall)
        f1s = np.nan_to_num(f1s)

        self.log(
            f"{tag}_median_f1", 
            np.nanmedian(f1s), 
            logger=True, 
            on_step=False, 
            on_epoch=True
        )

        return f1s 

    # Calculation on epoch end, for "median F1 score"
    def training_epoch_end(self, step_outputs):
        self._epoch_end(step_outputs,'train')
        
    def validation_epoch_end(self, step_outputs):
        self._epoch_end(step_outputs, 'val') 
    
    def test_epoch_end(self, step_outputs):
        self._epoch_end(step_outputs, 'test') 

    def configure_optimizers(self):
        if 'optimizer' in self.optim_params:
            optimizer = self.optim_params.pop('optimizer')
            optimizer = optimizer(self.parameters(), **self.optim_params)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.2, weight_decay=1e-5)

        if self.scheduler_params is not None:
            scheduler = self.scheduler_params.pop('scheduler')
            scheduler = scheduler(optimizer, **self.scheduler_params)

        if self.scheduler_params is None:
            return optimizer
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss',
        }

    def explain(self, loader, cache=False, normalize=False):
        if cache and self._explain_matrix is not None:
            return self._explain_matrix 
            
        self.network.eval()
        res_explain = []

        for batch_nb, data in enumerate(tqdm(loader)):
            if isinstance(data, tuple): # if we are running this on already labeled pairs and not just for inference
                data, _ = data 
                
            M_explain, masks = self.network.forward_masks(data)
            for key, value in masks.items():
                masks[key] = csc_matrix.dot(
                    value.cpu().detach().numpy(), self.reducing_matrix
                )

            original_feat_explain = csc_matrix.dot(
                M_explain.cpu().detach().numpy(),
                self.reducing_matrix
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

        return res_explain, res_masks

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
    
    def save_model(self, path):
        saved_params = {}
        init_params = {}
        for key, val in self.get_params().items():
            if isinstance(val, type):
                # Don't save torch specific params
                continue
            else:
                init_params[key] = val
        saved_params["init_params"] = init_params

        class_attrs = {
            "preds_mapper": self.preds_mapper
        }
        saved_params["class_attrs"] = class_attrs

        # Create folder
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save models params
        with open(Path(path).joinpath("model_params.json"), "w", encoding="utf8") as f:
            json.dump(saved_params, f, cls=ComplexEncoder)

        # Save state_dict
        torch.save(self.network.state_dict(), Path(path).joinpath("network.pt"))
        shutil.make_archive(path, "zip", path)
        shutil.rmtree(path)
        print(f"Successfully saved model at {path}.zip")
        return f"{path}.zip"

    def load_model(self, filepath):
        try:
            with zipfile.ZipFile(filepath) as z:
                with z.open("model_params.json") as f:
                    loaded_params = json.load(f)
                    loaded_params["init_params"]["device_name"] = self.device_name
                with z.open("network.pt") as f:
                    try:
                        saved_state_dict = torch.load(f, map_location=self.device)
                    except io.UnsupportedOperation:
                        # In Python <3.7, the returned file object is not seekable (which at least
                        # some versions of PyTorch require) - so we'll try buffering it in to a
                        # BytesIO instead:
                        saved_state_dict = torch.load(
                            io.BytesIO(f.read()),
                            map_location=self.device,
                        )
        except KeyError:
            raise KeyError("Your zip file is missing at least one component")

        self.__init__(**loaded_params["init_params"])

        self._set_network()
        self.network.load_state_dict(saved_state_dict)
        self.network.eval()
        self.load_class_attrs(loaded_params["class_attrs"])

    def load_weights_from_unsupervised(self, unsupervised_model):
        update_state_dict = copy.deepcopy(self.network.state_dict())
        for param, weights in unsupervised_model.network.state_dict().items():
            if param.startswith("encoder"):
                # Convert encoder's layers name to match
                new_param = "tabnet." + param
            else:
                new_param = param
            if self.network.state_dict().get(new_param) is not None:
                # update only common layers
                update_state_dict[new_param] = weights

    def _from_pretrained(self, **kwargs):
        update_list = [
            "cat_dims",
            "cat_emb_dim",
            "cat_idxs",
            "input_dim",
            "mask_type",
            "n_a",
            "n_d",
            "n_independent",
            "n_shared",
            "n_steps",
        ]
        for var_name, value in kwargs.items():
            if var_name in update_list:
                try:
                    exec(f"global previous_val; previous_val = self.{var_name}")
                    if previous_val != value:  # noqa
                        wrn_msg = f"Pretraining: {var_name} changed from {previous_val} to {value}"  # noqa
                        warnings.warn(wrn_msg)
                        exec(f"self.{var_name} = value")
                except AttributeError:
                    exec(f"self.{var_name} = value")

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
    precisions = tps / (tps+fps)
    recalls = tps / (tps+fns)
    
    f1s = 2*(np.dot(precisions, recalls)) / (precisions + recalls)
    
    return np.nanmedian(f1s)

def per_class_f1(*args, **kwargs):
    res = torchmetrics.functional.f1_score(*args, **kwargs, average='none')
    return res

def per_class_precision(*args, **kwargs):
    res = torchmetrics.functional.precision(*args, **kwargs, average='none')
    
    return res

def per_class_recall(*args, **kwargs):
    res = torchmetrics.functional.precision(*args, **kwargs, average='none')
    
    return res 

def weighted_accuracy(*args, **kwargs):
    res = torchmetrics.functional.accuracy(*args, **kwargs, average='weighted')
    
    return res 

def balanced_accuracy(*args, **kwargs):
    res = torchmetrics.functional.accuracy(*args, **kwargs, average='macro')
    
    return res 

def aggregate_metrics(num_classes) -> Dict[str, Callable]:
    metrics = {
        # Accuracies
        'total_accuracy': torchmetrics.functional.accuracy,
        'balanced_accuracy': partial(balanced_accuracy, num_classes=num_classes),
        'weighted_accuracy': partial(weighted_accuracy, num_classes=num_classes),
        
        # Precision, recall and f1s
        'precision': torchmetrics.functional.precision,
        'recall': torchmetrics.functional.recall,
        'f1': torchmetrics.functional.f1_score,
        
        # Per class 
        'per_class_f1': partial(per_class_f1, num_classes=num_classes),
        'per_class_precision': partial(per_class_precision, num_classes=num_classes),
        'per_class_recall': partial(per_class_recall, num_classes=num_classes),
        
        # Random stuff I might want
        'specificity': partial(torchmetrics.functional.specificity, num_classes=num_classes),
        'confusion_matrix': partial(torchmetrics.functional.confusion_matrix, num_classes=num_classes),
        'auroc': partial(torchmetrics.functional.auroc, num_classes=num_classes)
    }
    
    return metrics 