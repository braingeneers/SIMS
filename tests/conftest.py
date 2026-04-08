"""Shared pytest fixtures for the scsims test suite."""

from __future__ import annotations

import warnings

import numpy as np
import pytest


# Suppress the firehose of FutureWarning / DeprecationWarning that scanpy,
# anndata, lightning and friends emit on every import. We're not testing
# upstream library hygiene here.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture
def synthetic_anndata():
    """Build a small synthetic AnnData using scanpy's `blobs` dataset.

    Returns a 100-cell, 10-gene AnnData with a 5-class `blobs` label
    column. Defaults are deliberately tiny so the full test suite runs
    in seconds on CPU.
    """
    from scanpy.datasets import blobs

    adata = blobs(n_variables=10, n_observations=100)
    adata.X = adata.X.astype(np.float32)
    return adata


@pytest.fixture
def synthetic_anndata_more_genes():
    """A second synthetic AnnData with *more* genes than `synthetic_anndata`.

    Used to exercise the inference-time gene-alignment / zero-inflation
    code path in :meth:`SIMSClassifier._parse_data`.
    """
    from scanpy.datasets import blobs

    adata = blobs(n_variables=20, n_observations=100)
    adata.X = adata.X.astype(np.float32)
    return adata


@pytest.fixture
def synthetic_anndata_fewer_genes():
    """An AnnData with *fewer* genes than `synthetic_anndata_more_genes`,
    used to exercise the symmetric zero-inflation case (more training
    genes than inference genes).
    """
    from scanpy.datasets import blobs

    adata = blobs(n_variables=10, n_observations=100)
    adata.X = adata.X.astype(np.float32)
    return adata
