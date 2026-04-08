from .data import *
from .inference import DatasetForInference
from .lightning_train import *
from .model import *
from .pretraining import SIMSPretrainer, transfer_pretrained_weights
from .scvi_api import SIMS
from .temperature_scaling import *

# UploadCallback depends on boto3, which is an optional `s3` extra.
# Will be relocated to scsims.contrib in v4.x; for now keep it importable
# from the top level when boto3 is available, but don't crash without it.
try:
    from .networking import UploadCallback  # noqa: F401
except ModuleNotFoundError:
    UploadCallback = None  # type: ignore[assignment]

__version__ = "4.0.0"

__all__ = [
    "SIMS",
    "SIMSClassifier",
    "SIMSPretrainer",
    "DataModule",
    "UploadCallback",
    "DatasetForInference",
    "transfer_pretrained_weights",
]
