"""Regression test for the legacy SIMS checkpoint format.

The shipped sims_app/checkpoint/*.ckpt files were trained against
scsims 3.x. v4 must be able to load them unchanged. This test is
*skipped* in CI (the .ckpt files are huge and live in a separate
repo) but can be run locally to verify legacy compatibility.

Run with:

    SIMS_LEGACY_CKPT=/path/to/MGE_cortex.ckpt pytest tests/test_legacy_checkpoint.py
"""

from __future__ import annotations

import os

import pytest
import torch

from scsims import SIMS

LEGACY_CKPT = os.environ.get("SIMS_LEGACY_CKPT")


@pytest.mark.skipif(
    LEGACY_CKPT is None,
    reason="set SIMS_LEGACY_CKPT to a v3-era .ckpt path to enable",
)
def test_legacy_checkpoint_loads_and_forwards():
    """A v3-era checkpoint should:

    1. Deserialize cleanly into the v4 SIMSClassifier (no UnpicklingError
       from torch>=2.6's weights_only=True default).
    2. Round-trip a forward pass to produce logits of the expected shape.
    """
    sims = SIMS(weights_path=LEGACY_CKPT, map_location=torch.device("cpu"))
    assert sims.model.input_dim > 0
    assert sims.model.output_dim > 0
    assert sims.model.label_encoder is not None
    assert len(sims.model.label_encoder.classes_) == sims.model.output_dim

    sims.model.eval()
    batch = torch.randn(2, sims.model.input_dim)
    with torch.no_grad():
        logits, _M_loss = sims.model(batch)
    assert logits.shape == (2, sims.model.output_dim)
