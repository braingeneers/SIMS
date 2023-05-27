import os
from typing import *

import boto3
import pytorch_lightning as pl


class UploadCallback(pl.callbacks.Callback):
    """Custom PyTorch callback for uploading model checkpoints to a S3 bucket.

    Parameters:
    path: Local path to folder where model checkpoints are saved
    desc: Description of checkpoint that is appended to checkpoint file name on save
    upload_prefix: Path in bucket/ to upload model checkpoints to, defaults to model_checkpoints
    """

    def __init__(
        self,
        path: str,
        desc: str,
        s3: boto3.resource,
        bucket: str,
        upload_prefix="model_checkpoints",
        n_epochs: int = 10,
        quiet: bool = False,
    ) -> None:
        super().__init__()
        self.path = path
        self.desc = desc

        self.s3 = s3
        self.bucket = bucket
        self.upload_prefix = upload_prefix
        self.epochs = n_epochs
        self.quiet = quiet

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch % self.epochs == 0:  # Save every ten epochs
            checkpoint = f"checkpoint-{epoch}-desc-{self.desc}.ckpt"
            checkpoint_path = os.path.normpath(os.path.join(self.path, checkpoint))

            if not self.quiet:
                print(f"Saving checkpoint on epoch {epoch} to {checkpoint_path}")

            trainer.save_checkpoint(checkpoint_path)

            if not self.quiet:
                print(f"Uploading checkpoint at epoch {epoch}")

            try:
                self.s3.Bucket(self.bucket).upload_file(
                    Filename=checkpoint_path,
                    Key=os.path.join(self.upload_prefix, checkpoint_path),
                )

            except Exception as e:
                print(f"Error when uploading on epoch {epoch}")
                print(e)
