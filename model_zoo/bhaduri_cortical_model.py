import os
import pathlib 
import sys
import anndata as an
import torch 
import argparse 

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

import anndata as an 
import sys, os 
sys.path.append('../src')

import pathlib 
from typing import *

import torch
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
        '--lr',
        type=float,
        default=0.02,
        required=False,
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=3e-4,
        required=False,
    )

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
    lr, weight_decay, name, test = args.lr, args.weight_decay, args.name, args.test 

    here = pathlib.Path(__file__).parent.resolve()
    data_path = join(here, '..', 'data', 'bhaduri')

    print('Making data folder')
    os.makedirs(data_path, exist_ok=True)

    for file in [
        'primary_T.h5ad', 
        'organoid_T.h5ad', 
        'primary_labels_clean.csv', 
        'organoid_labels_clean.csv'
    ]:
        print(f'Downloading {file}')

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join('jlehrer', 'bhaduri_data', file),
                file_name=join(data_path, file),
            )

    primary = an.read_h5ad(join(data_path, 'primary_T.h5ad'), backed='r+')
    organoid = an.read_h5ad(join(data_path, 'organoid_T.h5ad'), backed='r+')

    prim_genes = [x.upper() for x in primary.var['index'].values]
    org_genes = [x.upper() for x in organoid.var['index'].apply(lambda x: x.split('|')[0]).values]
    refgenes = list(set(prim_genes).intersection(org_genes))

    module = DataModule(
        datafiles=[join(data_path, 'primary_T.h5ad')],
        labelfiles=[join(data_path, 'primary_labels_clean.csv')],
        class_label='Subtype',
        sep=',',
        batch_size=256,
        index_col='cell',
        num_workers=32,
        deterministic=True,
        normalize=True,
        assume_numeric_label=False,
        currgenes=prim_genes,
        refgenes=refgenes,
        preprocess=True,
    )

    wandb_logger = WandbLogger(
        project=f"Bhaduri Primary Cortical Model",
        name=name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    upload_callback = UploadCallback(
        path='checkpoints',
        desc='bhaduri_primary_cortical_weighted'
    )
    
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
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
        module.prepare_data()
        module.setup()

        model = SIMSClassifier(
            input_dim=module.num_features,
            output_dim=module.num_labels,
            weights=module.weights,
            n_d=32,
            n_a=32,
            n_steps=10,
        )

        trainer.fit(model, datamodule=module)
        trainer.test(model, datamodule=module)
    else:
        raise NotImplementedError("No checkpoints downloaded yet")