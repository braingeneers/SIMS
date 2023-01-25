import os
import pathlib
from typing import *

import boto3
import pytorch_lightning as pl

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, "data")

with open(os.path.join(here, "credentials")) as f:
    key, access = [line.rstrip() for line in f.readlines()]

s3 = boto3.resource(
    "s3",
    endpoint_url="https://s3-west.nrp-nautilus.io/",
    aws_access_key_id=key,
    aws_secret_access_key=access,
)


def upload(file_name, remote_name=None) -> None:
    """
    Uploads a file to the braingeneersdev S3 bucket

    Parameters:
    file_name: Local file to upload
    remote_name: Key for S3 bucket. Default is file_name
    """
    if remote_name == None:
        remote_name = file_name

    s3.Bucket("braingeneersdev").upload_file(
        Filename=file_name,
        Key=remote_name,
    )


class UploadCallback(pl.callbacks.Callback):
    """Custom PyTorch callback for uploading model checkpoints to the braingeneers S3 bucket.

    Parameters:
    path: Local path to folder where model checkpoints are saved
    desc: Description of checkpoint that is appended to checkpoint file name on save
    upload_path: Subpath in braingeneersdev/jlehrer/ to upload model checkpoints to
    """

    def __init__(
        self,
        path: str,
        desc: str,
        upload_path="model_checkpoints",
        epochs: int = 10,
    ) -> None:
        super().__init__()
        self.path = path
        self.desc = desc
        self.upload_path = upload_path
        self.epochs = epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch % self.epochs == 0 and epoch > 0:  # Save every ten epochs
            checkpoint = f"checkpoint-{epoch}-desc-{self.desc}.ckpt"
            trainer.save_checkpoint(os.path.join(self.path, checkpoint))
            print(f"Uploading checkpoint at epoch {epoch}")
            try:
                upload(
                    os.path.join(self.path, checkpoint),
                    os.path.join("jlehrer", self.upload_path, checkpoint),
                )
            except Exception as e:
                print(f"Error when uploading on epoch {epoch}")
                print(e)
