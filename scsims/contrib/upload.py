"""S3 model checkpoint upload callback (optional, requires ``boto3``).

Install with::

    pip install scsims[s3]

Usage::

    from scsims.contrib import UploadCallback

    cb = UploadCallback(
        desc="my_run",
        bucket="my-bucket",
        endpoint_url="https://s3.us-west-2.amazonaws.com",
        upload_prefix="model_checkpoints",
        metric="val_macro_accuracy",
        mode="max",
    )

    sims.setup_trainer(callbacks=[cb], ...)
    sims.train()

This used to live at ``scsims.networking.UploadCallback`` and was
hard-coded to the braingeneersdev S3 bucket. The old import path still
works for one release but emits a ``DeprecationWarning``.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import boto3
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


class UploadCallback(pl.callbacks.Callback):
    """Lightning callback that uploads the best checkpoint to S3 when training ends.

    Wraps a ``ModelCheckpoint`` (so the user gets the standard "best so
    far" semantics) and uploads the resulting ``.ckpt`` file to S3 in the
    ``on_train_end`` hook.

    Parameters
    ----------
    desc:
        Free-form description appended to the checkpoint filename. Useful
        when many runs share the same metric/mode.
    bucket:
        S3 bucket name.
    endpoint_url:
        S3 endpoint. Defaults to AWS US-West-2; pass a custom URL to
        target a non-AWS S3-compatible store (e.g. Nautilus, MinIO).
    path:
        Local directory to write checkpoints to.
    upload_prefix:
        Key prefix inside the bucket. Final S3 key is
        ``{upload_prefix}/{checkpoint_filename}``.
    metric:
        Metric name to monitor for "best so far" selection.
    mode:
        ``"max"`` (higher is better) or ``"min"`` (lower is better).
    aws_access_key_id, aws_secret_access_key:
        Credentials. If not provided, falls back to the
        ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` environment
        variables; if those are also missing, ``boto3`` will use any
        configured credential chain (instance profile, ``~/.aws``, etc.).
    """

    def __init__(
        self,
        desc: str,
        bucket: str,
        endpoint_url: Optional[str] = None,
        path: str = "model_checkpoints",
        upload_prefix: str = "model_checkpoints",
        metric: str = "val_micro_accuracy",
        mode: str = "max",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.desc = desc
        self.bucket = bucket
        self.upload_prefix = upload_prefix
        self.metric = metric
        self.mode = mode

        # Build a boto3 resource. boto3 will follow the standard credential
        # chain if explicit keys are not supplied; we only override when the
        # caller actually wants to.
        s3_kwargs = {}
        if endpoint_url is not None:
            s3_kwargs["endpoint_url"] = endpoint_url
        access_key = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        if access_key and secret_key:
            s3_kwargs["aws_access_key_id"] = access_key
            s3_kwargs["aws_secret_access_key"] = secret_key

        self.s3 = boto3.resource("s3", **s3_kwargs)

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.path,
            filename=f"best_{self.metric}_{self.desc}",
            monitor=self.metric,
            mode=self.mode,
            save_top_k=1,
            verbose=False,
        )

    def on_validation_end(self, trainer, pl_module):
        # Delegate to the wrapped ModelCheckpoint so it tracks "best so far".
        self.checkpoint_callback.on_validation_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        best = self.checkpoint_callback.best_model_path
        if not best:
            print(f"[UploadCallback] No best checkpoint to upload for {self.desc!r}")
            return
        # `best_model_path` is already absolute under `self.path`, so don't
        # double-join it (the previous version of this code did, producing
        # paths like `model_checkpoints/model_checkpoints/...` that didn't
        # exist on disk).
        try:
            self.s3.Bucket(self.bucket).upload_file(
                Filename=best,
                Key=os.path.join(self.upload_prefix, os.path.basename(best)),
            )
            print(f"[UploadCallback] Uploaded best checkpoint to s3://{self.bucket}/{self.upload_prefix}/{os.path.basename(best)}")
        except Exception as e:
            print(f"[UploadCallback] Error uploading checkpoint: {e}")


def _legacy_braingeneers_default(**overrides) -> UploadCallback:
    """Construct an UploadCallback with the legacy braingeneersdev defaults.

    Used by the deprecation shim at ``scsims.networking.UploadCallback``.
    """
    defaults = dict(
        bucket="braingeneersdev",
        endpoint_url="https://s3-west.nrp-nautilus.io",
    )
    defaults.update(overrides)
    return UploadCallback(**defaults)
