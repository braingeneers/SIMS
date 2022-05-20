import os
import pathlib 
import sys
import anndata as an
import torch 
import argparse 

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

import pandas as pd 
import numpy as np 
import anndata as an 
import sys, os 
sys.path.append('../src')

import sys
import os
import pathlib 
from typing import *

import torch
import numpy as np 
import pandas as pd 
import anndata as an

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data import *
from lightning_train import *
from model import *
from torchmetrics.functional import *
from networking import download 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        '--test',
        action='store_true',
        required=False,
    )

    device = ('cuda:0' if torch.cuda.is_available() else None)

    args = parser.parse_args()
    name, test = args.name, args.test 

    here = pathlib.Path(__file__).parent.resolve()
    data_path = join(here, '..', 'data', 'dental')

    print('Making data folder')
    os.makedirs(data_path, exist_ok=True)

    for file in ['human_dental_T.h5ad', 'labels_human_dental.tsv']:
        print(f'Downloading {file}')

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join('jlehrer', 'dental', file),
                file_name=join(data_path, file),
            )

    module = DataModule(
        datafiles=[join(data_path, 'human_dental_T.h5ad')],
        labelfiles=[join(data_path, 'labels_human_dental.tsv')],
        class_label='cell_type',
        sep='\t',
        batch_size=256,
        num_workers=32,
        deterministic=True,
        normalize=True,
    )

    wandb_logger = WandbLogger(
        project=f"Dental Model",
        name=name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    upload_callback = UploadCallback(
        path='checkpoints',
        desc='dental'
    )
    
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
    )

    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        auto_lr_find=False,
        logger=wandb_logger,
        max_epochs=500,
        gradient_clip_val=0.5,
        callbacks=[
            lr_callback, 
            upload_callback,
            early_stopping_callback,
        ]
    )

    if not test:
        model = SIMSClassifier(
            input_dim=module.num_features,
            output_dim=module.num_labels,
            weights=compute_class_weights([join(data_path, 'labels_human_dental.tsv')], class_label, sep='\t', device=device),
            scheduler_params={
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
                'factor': 0.75,
            },
        )

        trainer.fit(model, datamodule=module)
        trainer.test(model, datamodule=module)
    else:
        checkpoint_path = join(here, '..', 'checkpoints/checkpoint-80-desc-dental.ckpt')
        if not os.path.isfile(checkpoint_path):
            os.makedirs(join(here, '..', 'checkpoints'), exist_ok=True)
            download(
                remote_name='jlehrer/model_checkpoints/checkpoint-80-desc-dental.ckpt',
                file_name=checkpoint_path
            )

        model = SIMSClassifier.load_from_checkpoint(
            join(here, '..', 'checkpoints/checkpoint-80-desc-dental.ckpt'),
            input_dim=module.input_dim,
            output_dim=module.output_dim
        )

        trainer.test(model, datamodule=module)