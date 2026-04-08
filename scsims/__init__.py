from .data import *
from .inference import DatasetForInference
from .lightning_train import *
from .model import *
from .pretraining import SIMSPretrainer, transfer_pretrained_weights
from .scvi_api import SIMS
from .temperature_scaling import *

# UploadCallback now lives in scsims.contrib (depends on the optional
# `boto3`, install via `pip install scsims[s3]`). The legacy import path
# `scsims.networking.UploadCallback` continues to work via a deprecation
# shim and will be removed in scsims 5.0. We deliberately do NOT re-export
# UploadCallback from the top-level scsims namespace anymore: callers
# should import it explicitly from scsims.contrib so it's clear that the
# `s3` extra is required.

__version__ = "4.0.0"

__all__ = [
    "SIMS",
    "SIMSClassifier",
    "SIMSPretrainer",
    "DataModule",
    "DatasetForInference",
    "transfer_pretrained_weights",
]
