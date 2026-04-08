"""End-to-end smoke tests for the supervised SIMS workflow.

These are direct ports of the three tests that used to live in
``scsims/tests.py`` (which itself never ran via pytest because the
package shipped no test runner config). The originals had two latent
bugs:

1. They referenced ``predictions["first_pred"]`` which has never been a
   column scsims actually produces. The new tests use the canonical
   ``pred_0`` / ``prob_0`` names.
2. They wrote checkpoints to literal directories in the cwd
   (``./gene_removal_test_checkpoint``, etc.), which made parallel
   pytest runs racy. The new tests use the ``tmp_path`` fixture.
"""

from __future__ import annotations

import os

import lightning.pytorch as pl
import torch

from scsims import SIMS


def _train_tiny_sims(adata, ckpt_dir, max_epochs: int = 2):
    """Helper: train a SIMS model on the supplied AnnData and return the
    path to the resulting checkpoint."""
    sims = SIMS(data=adata, class_label="blobs")
    sims.setup_trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        enable_progress_bar=False,
        logger=False,
        checkpoint_dir=str(ckpt_dir),
    )
    sims.train()
    ckpt_files = sorted(os.listdir(ckpt_dir))
    assert ckpt_files, f"no checkpoints written to {ckpt_dir}"
    return os.path.join(ckpt_dir, ckpt_files[0])


def test_train_predict_round_trip(synthetic_anndata, tmp_path):
    """The bread-and-butter SIMS lifecycle: train -> save -> reload -> predict."""
    ckpt_path = _train_tiny_sims(synthetic_anndata, tmp_path)

    loaded = SIMS(weights_path=ckpt_path)
    predictions = loaded.predict(synthetic_anndata, num_workers=0, batch_size=16)

    assert len(predictions) == len(synthetic_anndata)
    # Default top_k=3 so we expect pred_0..pred_2 + prob_0..prob_2.
    assert {"pred_0", "pred_1", "pred_2", "prob_0", "prob_1", "prob_2"}.issubset(
        predictions.columns
    )


def test_inference_more_genes_than_training(
    synthetic_anndata, synthetic_anndata_more_genes, tmp_path
):
    """Inference-time gene-alignment: when the inference data has more genes
    than the training set, the extra genes should be silently dropped."""
    ckpt_path = _train_tiny_sims(synthetic_anndata, tmp_path)
    loaded = SIMS(weights_path=ckpt_path)

    predictions = loaded.predict(
        synthetic_anndata_more_genes, num_workers=0, batch_size=16
    )
    assert len(predictions) == len(synthetic_anndata_more_genes)


def test_inference_fewer_genes_than_training(
    synthetic_anndata_more_genes, synthetic_anndata_fewer_genes, tmp_path
):
    """Inference-time zero-inflation: when the inference data has *fewer*
    genes than training, the missing genes should be zero-padded."""
    # Force a non-numeric label to also exercise the LabelEncoder round-trip.
    synthetic_anndata_more_genes.obs["blobs"] = synthetic_anndata_more_genes.obs[
        "blobs"
    ].apply(lambda x: f"label_{x}")

    ckpt_path = _train_tiny_sims(synthetic_anndata_more_genes, tmp_path)
    loaded = SIMS(weights_path=ckpt_path)

    predictions = loaded.predict(
        synthetic_anndata_fewer_genes, num_workers=0, batch_size=16
    )
    assert len(predictions) == len(synthetic_anndata_fewer_genes)
    assert all(label.startswith("label_") for label in predictions["pred_0"].unique())


def test_predict_top_k_parameter(synthetic_anndata, tmp_path):
    """The new top_k parameter should produce the right number of columns
    and cap at the number of training classes."""
    ckpt_path = _train_tiny_sims(synthetic_anndata, tmp_path)
    loaded = SIMS(weights_path=ckpt_path)
    n_classes = len(loaded.model.label_encoder.classes_)

    # top_k=1
    p1 = loaded.predict(synthetic_anndata, num_workers=0, batch_size=16, top_k=1)
    assert list(p1.columns) == ["pred_0", "prob_0"]

    # top_k=2
    p2 = loaded.predict(synthetic_anndata, num_workers=0, batch_size=16, top_k=2)
    assert list(p2.columns) == ["pred_0", "pred_1", "prob_0", "prob_1"]

    # top_k > n_classes should silently cap.
    p_huge = loaded.predict(
        synthetic_anndata, num_workers=0, batch_size=16, top_k=999
    )
    assert sum(c.startswith("pred_") for c in p_huge.columns) == n_classes
    assert sum(c.startswith("prob_") for c in p_huge.columns) == n_classes


def test_explain_returns_matrix_with_gene_columns(synthetic_anndata, tmp_path):
    """The explainability path should return a (n_cells, n_genes) matrix."""
    import pandas as pd

    ckpt_path = _train_tiny_sims(synthetic_anndata, tmp_path)
    loaded = SIMS(weights_path=ckpt_path)

    explain_matrix, _labels = loaded.model.explain(
        synthetic_anndata, num_workers=0, batch_size=16
    )
    df = pd.DataFrame(explain_matrix, columns=loaded.model.genes)
    assert df.shape == (len(synthetic_anndata), len(loaded.model.genes))
