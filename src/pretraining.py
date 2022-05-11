from typing import *

import pandas as pd 
import torch
import numpy as np

import pytorch_lightning as pl 
from pytorch_tabnet.tab_network import (
    EmbeddingGenerator, 
    RandomObfuscator, 
    TabNetEncoder, 
    TabNetDecoder,
)

import sys, os 
# ALL THIS IS FLATTENING THE API FROM https://github.com/dreamquark-ai/tabnet
# GIVE MASSIVE CREDIT 

class NoiseObfuscator(torch.nn.Module):
    def __init__(
        self,
        variance: float=1,
    ) -> None:
        """
        Custom dataset interiting from parent Dataset that adds Gaussian noise with a given variance to each sample, to be used for pretraining

        :param variance: Variance of Gaussian noise, defaults to 1
        :type variance: float, optional
        """

        self.variance = variance

    def forward(self, x):
        return x + self.variance*torch.randn_like(x)

def UnsupervisedLoss(y_pred, embedded_x, obf_vars, eps=1e-9):
    """
    Implements unsupervised loss function.
    This differs from orginal paper as it's scaled to be batch size independent
    and number of features reconstructed independent (by taking the mean)
    Parameters
    ----------
    y_pred : torch.Tensor or np.array
        Reconstructed prediction (with embeddings)
    embedded_x : torch.Tensor
        Original input embedded by network
    obf_vars : torch.Tensor
        Binary mask for obfuscated variables.
        1 means the variable was obfuscated so reconstruction is based on this.
    eps : float
        A small floating point to avoid ZeroDivisionError
        This can happen in degenerated case when a feature has only one value
    Returns
    -------
    loss : torch float
        Unsupervised loss, average value over batch samples.
    """
    errors = y_pred - embedded_x
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    batch_means = torch.mean(embedded_x, dim=0)
    batch_means[batch_means == 0] = 1

    batch_stds = torch.std(embedded_x, dim=0) ** 2
    batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
    features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    # take the mean of the reconstructed variable errors
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    # here we take the mean per batch, contrary to the paper
    loss = torch.mean(features_loss)
    return loss

def UnsupervisedLossNumpy(y_pred, embedded_x, obf_vars, eps=1e-9):
    """Compute Euclidean distance between reconstructed and original vector 

    :param y_pred: _description_
    :type y_pred: _type_
    :param embedded_x: _description_
    :type embedded_x: _type_
    :param obf_vars: _description_
    :type obf_vars: _type_
    :param eps: _description_, defaults to 1e-9
    :type eps: _type_, optional
    :return: _description_
    :rtype: _type_
    """    
    errors = y_pred - embedded_x
    reconstruction_errors = np.multiply(errors, obf_vars) ** 2
    batch_means = np.mean(embedded_x, axis=0)
    batch_means = np.where(batch_means == 0, 1, batch_means)

    batch_stds = np.std(embedded_x, axis=0, ddof=1) ** 2
    batch_stds = np.where(batch_stds == 0, batch_means, batch_stds)
    features_loss = np.matmul(reconstruction_errors, 1 / batch_stds)
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables = np.sum(obf_vars, axis=1)
    # take the mean of the reconstructed variable errors
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    # here we take the mean per batch, contrary to the paper
    loss = np.mean(features_loss)
    return loss

class TabNetPretraining(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        pretraining_ratio=0.2,
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
        n_shared_decoder=1,
        n_indep_decoder=1,
    ):
        super(TabNetPretraining, self).__init__()

        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.pretraining_ratio = pretraining_ratio
        self.n_shared_decoder = n_shared_decoder
        self.n_indep_decoder = n_indep_decoder

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.masker = RandomObfuscator(self.pretraining_ratio)
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )

        self.decoder = TabNetDecoder(
            self.post_embed_dim,
            n_d=n_d,
            n_steps=n_steps,
            n_independent=self.n_indep_decoder,
            n_shared=self.n_shared_decoder,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

    def forward(self, x):
        embedded_x = self.embedder(x)
        if self.training:
            masked_x, obf_vars = self.masker(embedded_x)
            # set prior of encoder with obf_mask
            prior = 1 - obf_vars
            steps_out, _ = self.encoder(masked_x, prior=prior)
            res = self.decoder(steps_out)
            return res, embedded_x, obf_vars
        else:
            steps_out, _ = self.encoder(embedded_x)
            res = self.decoder(steps_out)
            return res, embedded_x, torch.ones(embedded_x.shape).to(x.device)

    def forward_masks(self, x):
        embedded_x = self.embedder(x)
        return self.encoder.forward_masks(embedded_x)

    def training_step(self, batch, batch_idx):
        y, y_hat, loss = self._step(batch)

        self.log("train_loss", loss, logger=True, on_epoch=True, on_step=True)
        self._compute_metrics(y_hat, y, 'train')

        return loss

    def _compute_metrics(self, 
        y_hat: torch.Tensor, 
        y: torch.Tensor, 
        tag: str,
        on_epoch=True, 
        on_step=False,
    ):
        val = metric(y_hat, y, average='weighted', num_classes=self.output_dim)

        self.log(
            f"Reconstuction loss", 
            val, 
            on_epoch=on_epoch, 
            on_step=on_step,
            logger=True,
        )

def pretrain_model(
    datamodule,
    model,
    trainer,
):
    """
    Pretrain model via random masking or denoising

    :param datamodule: Datamodule to train model on 
    :type datamodule: pl.DataModule
    :param model: Model to train on 
    :type model: Model
    :param trainer: PyTorch Lightning trainer used to train model
    :type trainer: pl.LightningTrainer
    """

    pass 
