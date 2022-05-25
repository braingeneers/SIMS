import os
import pathlib 
import sys
import anndata as an
import torch 
import argparse 

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from data import *
from model import *
from lightning_train import *
from networking import download 

import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import * 

from functools import partial 

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
        default=False,
        action='store_true',
        required=False,
    )

    args = parser.parse_args()
    lr, weight_decay, name, test = args.lr, args.weight_decay, args.name, args.test 

    here = pathlib.Path(__file__).parent.resolve()
    data_path = join(here, '..', 'data', 'mouse')

    print('Making data folder')
    os.makedirs(data_path, exist_ok=True)

    for file in ['MouseAdultInhibitoryNeurons.h5ad', 'Mo_PV_paper_TDTomato_mouseonly.h5ad', 'MouseAdultInhibitoryNeurons_labels.csv']:
        print(f'Downloading {file}')

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join('jlehrer', 'mouse_data', file),
                file_name=join(data_path, file),
            )

    # Calculate gene intersection
    mouse_atlas = an.read_h5ad(join(data_path, 'MouseAdultInhibitoryNeurons.h5ad'))
    mo_data = an.read_h5ad(join(data_path, 'Mo_PV_paper_TDTomato_mouseonly.h5ad'))

    mo_genes = mo_data.var.index.values
    atlas_genes = mouse_atlas.var.index.values

    mo_genes = [x.strip().upper() for x in mo_genes]
    atlas_genes = [x.strip().upper() for x in atlas_genes]

    refgenes = sorted(list(set(mo_genes).intersection(atlas_genes)))

    # # Define labelfiles and trainer 
    datafiles=[join(data_path, 'MouseAdultInhibitoryNeurons.h5ad')]
    labelfiles=[join(data_path, 'MouseAdultInhibitoryNeurons_labels.csv')]

    device = ('cuda:0' if torch.cuda.is_available() else None)

    module = DataModule(
        datafiles=datafiles,
        labelfiles=labelfiles,
        class_label='numeric_class',
        batch_size=8,
        num_workers=0,
        shuffle=True,
        drop_last=True,
        normalize=True,
        refgenes=refgenes,
        currgenes=atlas_genes,
        deterministic=True,
    )

    print(f'Number of classes in {__file__} is {module.output_dim}')

    wandb_logger = WandbLogger(
        project=f"Mostajo Mouse Model",
        name=name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    upload_callback = UploadCallback(
        path='checkpoints',
        desc='mostajo-mouse-5-24-22-more-more-layers-weighted'
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
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
            n_d=64,
            n_a=64,
            n_steps=10,
            weights=module.weights,
        )

        # train model
        trainer.fit(model, datamodule=module)
        trainer.test(model, datamodule=module)
    else:
        checkpoint_path = join(here, '..', 'checkpoints', 'checkpoint-280-desc-mouse.ckpt')

        if not os.path.isfile(checkpoint_path):
            os.makedirs(join(here, '../checkpoints'), exist_ok=True)
            print('Downloading file')
            download(
                'jlehrer/model_checkpoints/checkpoint-280-desc-mouse.ckpt',
                checkpoint_path,
            )

        model = SIMSClassifier.load_from_checkpoint(
            checkpoint_path,
            input_dim=module.input_dim,
            output_dim=module.output_dim,
        )

        trainer.test(model, datamodule=module)
