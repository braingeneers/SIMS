from multiprocessing.sharedctypes import Value
from typing import *

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, precision, recall 
from pytorch_tabnet.tab_network import TabNet

# Set all seeds for reproducibility
class GeneClassifier(pl.LightningModule):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        base_model=None,
        optim_params: Dict[str, float]={
            'optimizer': torch.optim.Adam,
            'lr': 0.001,
            'weight_decay': 0.01,
        },
        metrics: Dict[str, Callable]={
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        },
        weighted_metrics=False,
        *args,
        **kwargs,
    ):
        """
        Initialize the gene classifier neural network

        :param input_dim: Number of features in the inpute matrix 
        :type input_dim: Number of classes
        :param output_dim: Number of classes
        :type output_dim: int
        :param base_model: Model to use in training, otherwise defaults to TabNet, defaults to None
        :type base_model: _type_, optional
        :param optim_params: Dictionary containing information to instantiate optimizer, defaults to { 'optimizer': torch.optim.Adam, 'lr': 0.001, 'weight_decay': 0.01, }
        :type optim_params: _type_, optional
        :param metrics: List of pl_lightning.functional to compute, defaults to { 'accuracy': accuracy, 'precision': precision, 'recall': recall, }
        :type metrics: _type_, optional
        :param weighted_metrics: To use class-weighted metric calcuation, defaults to False
        :type weighted_metrics: bool, optional
        """    
        super().__init__()
        print(f'Model initialized. {input_dim = }, {output_dim = }. Metrics are {metrics.keys()} and {weighted_metrics = }')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optim_params = optim_params
        self.metrics = metrics
        self.weighted_metrics = weighted_metrics

        if base_model is None:
            self.base_model = TabNetGeneClassifier(
                input_dim=input_dim,
                output_dim=output_dim,
                *args,
                **kwargs
            )
        else:
            self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)

    def _step(self,
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Calculate the model output and loss for a given batch, to be used in training and validation steps

        :param batch: Standard DataLoader batch
        :type batch: Tuple[torch.Tensor, torch.Tensor]
        :param batch_idx: index of batch, irrelevant
        :type batch_idx: int
        :return: label tensor, logits tensor, loss 
        :rtype: Tuple[torch.Tensor, torch.Tensor, float]
        """        
        if isinstance(self.base_model, TabNetGeneClassifier):
            # Hacky and annoying, but the extra M_loss from TabNet means we need to handle this specific case 
            y_hat, y, loss = self.base_model._step(batch, batch_idx)
        else:
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)

        return y, y_hat, loss 

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        y, y_hat, loss = self._step(batch, batch_idx)        

        self.log("train_loss", loss, logger=True, on_epoch=True, on_step=True)
        self._compute_metrics(y_hat, y, 'train')

        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        y, y_hat, loss = self._step(batch, batch_idx)    

        self.log("val_loss", loss, logger=True, on_epoch=True, on_step=True)
        self._compute_metrics(y_hat, y, 'val')
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        y, y_hat, loss = self._step(batch, batch_idx)    

        self.log("test_loss", loss, logger=True, on_epoch=True, on_step=True)
        self._compute_metrics(y_hat, y, 'test')
        
        return loss

    def _compute_metrics(self, 
        y_hat: torch.Tensor, 
        y: torch.Tensor, 
        tag: str, 
        on_epoch=True, 
        on_step=False,
    ):
        """
        Compute metrics for the given batch

        :param y_hat: logits of model
        :type y_hat: torch.Tensor
        :param y: tensor of labels
        :type y: torch.Tensor
        :param tag: log name, to specify train/val/test batch calculation
        :type tag: str
        :param on_epoch: log on epoch, defaults to True
        :type on_epoch: bool, optional
        :param on_step: log on step, defaults to True
        :type on_step: bool, optional
        """
        for name, metric in self.metrics.items():
            if self.weighted_metrics: # We dont consider class support in calculation
                val = metric(y_hat, y, average='weighted', num_classes=self.output_dim)
                self.log(
                    f"weighted_{tag}_{name}", 
                    val, 
                    on_epoch=on_epoch, 
                    on_step=on_step,
                    logger=True,
                )
            else:
                val = metric(y_hat, y)
                self.log(
                    f"{tag}_{name}", 
                    val, 
                    on_epoch=on_epoch, 
                    on_step=on_step,
                    logger=True,
                )

    def configure_optimizers(self):
        optimizer = self.optim_params.pop('optimizer')
        optimizer = optimizer(self.parameters(), **self.optim_params)

        return optimizer

class TabNetGeneClassifier(TabNet):
    """
    Just a simple wrapper to only return the regular output instead the output and M_loss as defined in the tabnet paper.
    This allows me to use a single train/val/test loop for both models. 
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        out, M_loss = super().forward(x)
        return out, M_loss

    # leaving this in case we ever want to add lambda_sparse parameter, should be easy 
    def _step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.forward(x)
        
        # Add extra sparsity as required by TabNet 
        loss = F.cross_entropy(y_hat, y)

        return y_hat, y, loss