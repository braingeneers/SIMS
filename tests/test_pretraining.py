"""Tests for the v4 self-supervised pretraining feature.

The pretrainer was non-functional dead code in scsims 3.x; v4 rewrites
it as a real LightningModule wrapping pytorch_tabnet's TabNetPretraining
plus a `transfer_pretrained_weights` helper. These tests pin the
behavior we promised in the v4 release notes.
"""

from __future__ import annotations

import os

import lightning.pytorch as pl
import torch

from scsims import SIMS, SIMSPretrainer, transfer_pretrained_weights


def test_pretrainer_constructs_with_default_kwargs():
    """Smoke test: SIMSPretrainer should build with just an input_dim."""
    pretrainer = SIMSPretrainer(input_dim=20)
    assert pretrainer.input_dim == 20
    assert isinstance(pretrainer, pl.LightningModule)


def test_pretrainer_forward_pass_shapes():
    """The pretrainer should return (reconstruction, embedded_x, obf_mask),
    all the same shape as the post-embedded input."""
    pretrainer = SIMSPretrainer(input_dim=20)
    pretrainer.eval()  # disable masking randomness for a deterministic shape check
    x = torch.randn(8, 20)
    out = pretrainer(x)
    assert isinstance(out, tuple) and len(out) == 3
    y_pred, embedded, obf_vars = out
    assert y_pred.shape == embedded.shape == obf_vars.shape


def test_unsupervised_loss_is_positive_and_finite():
    """The reconstruction loss should be finite and non-negative on noise."""
    from scsims.pretraining import unsupervised_reconstruction_loss

    torch.manual_seed(0)
    embedded = torch.randn(16, 20)
    y_pred = embedded + 0.1 * torch.randn_like(embedded)
    obf_vars = (torch.rand_like(embedded) > 0.8).float()
    loss = unsupervised_reconstruction_loss(y_pred, embedded, obf_vars)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_pretrain_warm_start_transfers_encoder_weights(synthetic_anndata, tmp_path):
    """Stage 1 (pretrain) -> Stage 2 (fine-tune with warm start) should
    actually transfer encoder weights between the two networks."""
    sims = SIMS(data=synthetic_anndata, class_label="blobs")

    # Stage 1: pretrain
    sims.pretrain(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        enable_progress_bar=False,
        logger=False,
        checkpoint_dir=str(tmp_path / "pretrain"),
    )
    assert sims.pretrainer is not None
    assert isinstance(sims.pretrainer, SIMSPretrainer)

    # Stage 2: build the classifier and run the transfer manually so we can
    # assert on the tensor count + actually inspect a transferred weight.
    sims.setup_trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        enable_progress_bar=False,
        logger=False,
        checkpoint_dir=str(tmp_path / "finetune"),
    )
    sims.setup_model()
    n_transferred = transfer_pretrained_weights(sims.model, sims.pretrainer)
    assert n_transferred > 0, "transfer_pretrained_weights should report at least one tensor"

    # Spot-check one transferred encoder tensor: it should now match the
    # pretrainer's value bit-for-bit.
    src_state = sims.pretrainer.network.state_dict()
    dst_state = sims.model.network.state_dict()
    encoder_keys = [k for k in src_state if k.startswith("encoder")]
    assert encoder_keys, "pretrainer must have at least one encoder.* key"
    for src_key in encoder_keys:
        dst_key = f"tabnet.{src_key}"
        if dst_key in dst_state and dst_state[dst_key].shape == src_state[src_key].shape:
            assert torch.equal(dst_state[dst_key], src_state[src_key]), (
                f"transferred tensor mismatch at {src_key}"
            )
            break
    else:
        raise AssertionError("no shape-matching encoder keys to spot-check")


def test_pretrain_on_truly_unlabeled_anndata(synthetic_anndata, tmp_path):
    """`SIMS(data=adata)` (no class_label) should support pretraining on
    a literally unlabeled AnnData. The 3.x API forced users to pass a
    class_label even when pretraining ignored it; v4 cleaned that up."""
    # Drop the labels column entirely so any code path that tries to
    # read it would crash.
    del synthetic_anndata.obs["blobs"]
    assert "blobs" not in synthetic_anndata.obs.columns

    sims = SIMS(data=synthetic_anndata)  # no class_label kwarg!
    assert sims.datamodule.class_label is None
    assert sims.datamodule.label_encoder is None
    assert sims.datamodule.weights is None

    sims.pretrain(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        enable_progress_bar=False,
        logger=False,
        checkpoint_dir=str(tmp_path / "pretrain"),
    )
    assert isinstance(sims.pretrainer, SIMSPretrainer)


def test_train_without_class_label_raises_clear_error(synthetic_anndata):
    """Calling .train() on a SIMS built without a class_label should raise
    a ValueError that points users at the right API."""
    import pytest

    del synthetic_anndata.obs["blobs"]
    sims = SIMS(data=synthetic_anndata)
    with pytest.raises(ValueError, match="class_label"):
        sims.train()


def test_load_pretrainer_round_trip(synthetic_anndata, tmp_path):
    """Two-process workflow: pretrain in 'process A', save .ckpt, fresh
    SIMS in 'process B' loads it via load_pretrainer() and warm-starts
    a fine-tune. Mimics the realistic case where pretrain and fine-tune
    happen in separate Python sessions."""
    pretrain_dir = tmp_path / "pretrain"
    finetune_dir = tmp_path / "finetune"

    sims_a = SIMS(data=synthetic_anndata, class_label="blobs")
    sims_a.pretrain(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        enable_progress_bar=False,
        logger=False,
        checkpoint_dir=str(pretrain_dir),
    )
    pretrain_ckpt = os.path.join(pretrain_dir, sorted(os.listdir(pretrain_dir))[0])
    assert pretrain_ckpt.endswith(".ckpt")

    # Fresh SIMS, no pretrainer attribute yet
    sims_b = SIMS(data=synthetic_anndata, class_label="blobs")
    assert not hasattr(sims_b, "pretrainer")

    sims_b.load_pretrainer(pretrain_ckpt)
    assert isinstance(sims_b.pretrainer, SIMSPretrainer)

    sims_b.setup_trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        enable_progress_bar=False,
        logger=False,
        checkpoint_dir=str(finetune_dir),
    )
    # If train() finishes without raising, the warm-start path worked.
    sims_b.train()
