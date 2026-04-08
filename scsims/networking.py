"""Deprecated module: use ``scsims.contrib.upload`` instead.

This shim exists so existing code that does
``from scsims.networking import UploadCallback`` still works for one
release. It will be removed in scsims 5.0.
"""

import warnings

from scsims.contrib.upload import UploadCallback as _RealUploadCallback
from scsims.contrib.upload import _legacy_braingeneers_default

warnings.warn(
    "scsims.networking is deprecated and will be removed in scsims 5.0. "
    "Use `from scsims.contrib import UploadCallback` instead. The contrib "
    "version takes `bucket` and `endpoint_url` as explicit arguments rather "
    "than hard-coding the braingeneersdev S3 bucket.",
    DeprecationWarning,
    stacklevel=2,
)


class UploadCallback(_RealUploadCallback):
    """Backwards-compatible wrapper that hard-codes the braingeneersdev bucket.

    Equivalent to ``scsims.contrib.UploadCallback`` but with
    ``bucket="braingeneersdev"`` and the Nautilus endpoint pre-filled.
    Existing scripts that constructed the old ``UploadCallback`` without
    a bucket argument will keep working.
    """

    def __init__(self, desc, *, path="model_checkpoints",
                 upload_prefix="model_checkpoints",
                 metric="val_micro_accuracy", mode="max"):
        super().__init__(
            desc=desc,
            bucket="braingeneersdev",
            endpoint_url="https://s3-west.nrp-nautilus.io",
            path=path,
            upload_prefix=upload_prefix,
            metric=metric,
            mode=mode,
        )
