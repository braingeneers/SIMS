"""TabNet self-supervised pretraining for SIMS.

This module provides :class:`SIMSPretrainer`, a Lightning module that runs
TabNet self-supervised pretraining (random feature obfuscation +
reconstruction loss, as described in section 3.4 of the original TabNet
paper). The encoder learned during pretraining can be transferred into a
fresh :class:`scsims.SIMSClassifier` to warm-start supervised fine-tuning.

Typical usage:

.. code-block:: python

    from scsims import SIMS

    sims = SIMS(data=adata, class_label="cell_type")

    # Stage 1: unsupervised pretraining on the same dataset (or a larger
    # unlabeled one).
    sims.pretrain(max_epochs=50, accelerator="gpu", devices=1)

    # Stage 2: supervised fine-tuning with the pretrained encoder
    # warm-started automatically.
    sims.train(max_epochs=20, accelerator="gpu", devices=1)

Notes
-----
- Pretraining is unsupervised: cell type labels in the AnnData object are
  ignored entirely. The DataModule still yields ``(features, labels)``
  tuples; we discard the labels.
- The pretrainer is an ordinary ``pl.LightningModule``, so any Lightning
  ``Trainer`` knob (DDP, mixed precision, callbacks, loggers) works as
  you would expect.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from pytorch_tabnet.tab_network import TabNetPretraining
from pytorch_tabnet.utils import create_group_matrix


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def unsupervised_reconstruction_loss(
    y_pred: torch.Tensor,
    embedded_x: torch.Tensor,
    obf_vars: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """TabNet self-supervised reconstruction loss.

    Computes the squared reconstruction error of the obfuscated features,
    normalized by per-feature variance so that high-variance features don't
    dominate the loss. Mirrors the formulation from the TabNet paper.

    Parameters
    ----------
    y_pred:
        Reconstructed embeddings from the decoder, shape ``(batch, post_embed_dim)``.
    embedded_x:
        The original (un-obfuscated) embeddings produced by the embedder.
    obf_vars:
        Binary mask, ``1`` where the feature was obfuscated and the model
        is being asked to reconstruct it.
    eps:
        Numerical safety floor for the per-row normalisation.
    """
    errors = y_pred - embedded_x
    reconstruction_errors = (errors * obf_vars) ** 2

    # Normalise by per-feature variance to weight all features comparably.
    batch_means = embedded_x.mean(dim=0)
    batch_means = torch.where(
        batch_means == 0,
        torch.ones_like(batch_means),
        batch_means,
    )
    batch_vars = embedded_x.var(dim=0, unbiased=False)
    batch_vars = torch.where(batch_vars == 0, batch_means, batch_vars)

    features_loss = torch.matmul(reconstruction_errors, 1.0 / batch_vars)

    # Average over the number of obfuscated variables per row.
    nb_obfuscated = obf_vars.sum(dim=1)
    features_loss = features_loss / (nb_obfuscated + eps)

    return features_loss.mean()


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------


class SIMSPretrainer(pl.LightningModule):
    """Lightning wrapper around :class:`pytorch_tabnet.tab_network.TabNetPretraining`.

    Parameters mirror :class:`scsims.SIMSClassifier` so that pretrainer and
    classifier hyperparameters stay aligned. Genes are stored on the
    pretrainer so a downstream :class:`scsims.SIMSClassifier` can be
    sanity-checked against the same input ordering.
    """

    def __init__(
        self,
        input_dim: int,
        pretraining_ratio: float = 0.2,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        cat_idxs: Optional[list] = None,
        cat_dims: Optional[list] = None,
        cat_emb_dim: int = 1,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        mask_type: str = "sparsemax",
        n_shared_decoder: int = 1,
        n_indep_decoder: int = 1,
        grouped_features: Optional[list[list[int]]] = None,
        optim_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        genes: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["genes"])

        self.input_dim = input_dim
        self.genes = genes

        cat_idxs = cat_idxs or []
        cat_dims = cat_dims or []

        # NOTE: SIMSClassifier (the supervised path) builds the
        # group_attention_matrix as a sparse identity to avoid the
        # 4.7 GB allocation that pytorch_tabnet's create_group_matrix
        # incurs for 34k-input single-cell models. We can't do the same
        # here because TabNetPretraining performs an `> 0` comparison
        # on group_attention_matrix internally, which is not implemented
        # for any sparse layout (COO/CSR/CSC) as of pytorch-tabnet 4.1
        # and torch 2.11. Pretraining is a long-running training job
        # typically run on workstations with plenty of RAM, so the dense
        # cost is acceptable here.
        group_attention_matrix = create_group_matrix(
            grouped_features if grouped_features is not None else [],
            input_dim,
        )

        self.network = TabNetPretraining(
            input_dim=input_dim,
            pretraining_ratio=pretraining_ratio,
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
            n_shared_decoder=n_shared_decoder,
            n_indep_decoder=n_indep_decoder,
            group_attention_matrix=group_attention_matrix,
        )

        self.optim_params = optim_params or {
            "optimizer": torch.optim.Adam,
            "lr": 2e-2,
            "weight_decay": 1e-5,
        }
        self.scheduler_params = scheduler_params

    # ------------------------------------------------------------------
    # Standard Lightning hooks
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def _step(self, batch) -> torch.Tensor:
        # The supervised DataModule yields (features, labels). Pretraining
        # is unsupervised so we drop the label component.
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.float()
        y_pred, embedded_x, obf_vars = self.network(x)
        return unsupervised_reconstruction_loss(y_pred, embedded_x, obf_vars)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("pretrain_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_pretrain_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Same defensive-copy pattern as SIMSClassifier so a second
        # configure_optimizers call (e.g. resuming a run) doesn't crash.
        optim_params = dict(self.optim_params)
        optimizer_cls = optim_params.pop("optimizer", torch.optim.Adam)
        optimizer = optimizer_cls(self.parameters(), **optim_params)

        if self.scheduler_params is None:
            return optimizer

        scheduler_params = dict(self.scheduler_params)
        scheduler_cls = scheduler_params.pop("scheduler")
        scheduler = scheduler_cls(optimizer, **scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "pretrain_loss",
        }


# ---------------------------------------------------------------------------
# Weight transfer: pretrainer -> SIMSClassifier
# ---------------------------------------------------------------------------


def transfer_pretrained_weights(classifier, pretrainer: SIMSPretrainer) -> int:
    """Copy encoder + embedder weights from a pretrainer into a classifier.

    Mirrors the algorithm used by :func:`pytorch_tabnet.abstract_model.
    TabModel.load_weights_from_unsupervised`. Encoder/embedder layers from
    the pretrainer's ``TabNetPretraining`` network are mapped onto the
    corresponding layers inside the classifier's ``TabNet`` network. Layers
    that exist only in the classifier (e.g. the supervised classification
    head) are left at their random initialization.

    Returns the number of tensors that were transferred, for logging.
    """
    target_state = deepcopy(classifier.network.state_dict())
    source_state = pretrainer.network.state_dict()
    transferred = 0

    for source_key, source_value in source_state.items():
        # The supervised TabNet wraps its encoder under `tabnet.`, so encoder.*
        # in the pretrainer maps to tabnet.encoder.* in the classifier.
        # Embedder layers keep the same path.
        if source_key.startswith("encoder"):
            target_key = f"tabnet.{source_key}"
        elif source_key.startswith("embedder"):
            target_key = source_key
        else:
            # decoder, masker, etc. — only meaningful at pretraining time.
            continue

        target_tensor = target_state.get(target_key)
        if target_tensor is None:
            continue
        if target_tensor.shape != source_value.shape:
            # Architectural mismatch (different n_d, n_a, n_steps...).
            # Skip rather than crash so partial transfer still works.
            continue
        target_state[target_key] = source_value.clone()
        transferred += 1

    classifier.network.load_state_dict(target_state, strict=False)
    return transferred
