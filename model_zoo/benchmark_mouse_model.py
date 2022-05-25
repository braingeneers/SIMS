import os
import pathlib 
import sys
import anndata as an
import torch 
import argparse 

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from typing import *
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data import *
from lightning_train import *
from model import *
from torchmetrics.functional import *
from networking import download, list_objects

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
    data_path = join(here, '..', 'data', 'benchmark')

    print('Making data folder')
    os.makedirs(data_path, exist_ok=True)

    labels = list_objects('jlehrer/benchmark/mouse_labels')
    # Download training labels set 
    for file in labels:
        print(f'Downloading {file}')

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join(file),
                file_name=join(data_path, file.split('/')[-1]),
            )

    # Download training data
    for file in ['human_labels_clean.csv', 'human.h5ad']:
        print(f'Downloading {file}')

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join('jlehrer', 'human_benchmark', file),
                file_name=join(data_path, file),
            )

    # Download mouse data so we can get column intersection
    if not os.path.isfile(join(data_path, 'mouse_clipped.h5ad')):
        download(
            remote_name=join('jlehrer', 'mouse_benchmark', 'mouse_clipped.h5ad'),
            file_name=join(data_path, 'mouse_clipped.h5ad')
        )

    # human = an.read_h5ad(join(data_path, 'human.h5ad'), backed='r+')
    # mouse = an.read_h5ad(join(data_path, 'mouse_clipped.h5ad'), backed='r+')

    # human_cols = [x.strip().upper() for x in human.var.index.values]
    # mouse_cols = [x.strip().upper() for x in mouse.var.index.values]

    # refgenes = list(set(human_cols).intersection(mouse_cols))

    module = DataModule(
        datafiles=[join(data_path, 'mouse_clipped.h5ad')],
        labelfiles=[join(data_path, 'mouse_VISp_labels_clean.csv')],
        class_label='subclass_label',
        sep=',',
        batch_size=16,
        index_col='cell',
        num_workers=0,
        deterministic=True,
        normalize=True,
        assume_numeric_label=False,
        stratify=False,
        # currgenes=human_cols,
        # refgenes=refgenes,
        # preprocess=False,
    )

    wandb_logger = WandbLogger(
        project=f"Mouse Benchmark",
        name=f"Mouse Benchmark",
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    upload_callback = UploadCallback(
        path='checkpoints',
        desc='mouse_benchmark_visp'
    )
    
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
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
        )

        trainer.fit(model, datamodule=module)
        trainer.test(model, datamodule=module)
    else:
        raise NotImplementedError("No checkpoints downloaded yet")