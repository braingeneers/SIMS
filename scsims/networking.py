import os
from typing import *

import boto3
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class UploadCallback(pl.callbacks.Callback):
    """Custom PyTorch callback for uploading the best model checkpoint to a S3 bucket.

    Parameters:
    path: Local path to folder where model checkpoints are saved
    desc: Description of checkpoint that is appended to checkpoint file name on save
    s3: boto3.resource instance
    bucket: Name of the S3 bucket
    upload_prefix: Path in bucket to upload model checkpoints to, defaults to model_checkpoints
    metric: Metric to monitor for checkpointing
    mode: 'max' or 'min' depending on whether higher values of the metric are better or worse
    """

    def __init__(
        self,
        desc: str,
        path: str = "model_checkpoints",
        upload_prefix = "model_checkpoints",
        metric: str = 'val_micro_accuracy',
        mode: str = 'max',
    ) -> None:
        super().__init__()
        self.path = path
        self.desc = desc

        self.s3 = boto3.resource(
            "s3",
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            endpoint_url="https://s3-west.nrp-nautilus.io",
        )
        self.bucket = "braingeneersdev"
        self.upload_prefix = upload_prefix
        self.metric = metric
        self.mode = mode

        # Initialize ModelCheckpoint callback
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.path,
            filename=f"best_{self.metric}_{self.desc}",
            monitor=self.metric,
            mode=self.mode,
            save_top_k=1,  # Save only the best checkpoint
            verbose=False,
        )

    def on_validation_end(self, trainer, pl_module):
        # Call the ModelCheckpoint callback's on_validation_end method
        self.checkpoint_callback.on_validation_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        # Upload the best checkpoint to S3
        best_checkpoint_path = os.path.join(self.path, self.checkpoint_callback.best_model_path)
        try:
            self.s3.Bucket(self.bucket).upload_file(
                Filename=best_checkpoint_path,
                Key=os.path.join(self.upload_prefix, os.path.basename(best_checkpoint_path)),
            )
            print(f"Uploaded best checkpoint to S3: {best_checkpoint_path}")
        except Exception as e:
            print(f"Error when uploading the best checkpoint to S3: {e}")