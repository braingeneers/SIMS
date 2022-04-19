import pathlib 
import os 
from typing import *

import torch
import pandas as pd 
import anndata as an
from functools import cached_property

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .model import GeneClassifier
from .data import generate_dataloaders

class DataModule(pl.LightningDataModule):
    """
    Creates the DataModule for PyTorch-Lightning training.
    """
    def __init__(
        self, 
        datafiles,
        labelfiles,
        class_label,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.datafiles = datafiles 
        self.labelfiles = labelfiles 
        self.class_label = class_label

        self.args = args 
        self.kwargs = kwargs
        
    def setup(self, stage: Optional[str] = None):
        print('Creating train/val/test DataLoaders...')
        trainloader, valloader, testloader = generate_dataloaders(
            datafiles=self.datafiles,
            labelfiles=self.labelfiles,
            class_label=self.class_label,
            *self.args,
            **self.kwargs,
            pin_memory=True, # For gpu training
        )

        print('Done, continuing to training.')
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        
    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader

    def test_dataloader(self):
        return self.testloader

    @cached_property
    def num_labels(self):
        val = []
        sep = self.kwargs['sep'] if 'sep' in self.kwargs else ','
        for file in self.labelfiles:
            val.append(pd.read_csv(file, sep=sep).loc[:, self.class_label].values.max())

        return max(val) + 1

    @cached_property
    def num_features(self):
        if 'refgenes' in self.kwargs:
            return len(self.kwargs['refgenes'])
        elif hasattr(self, 'trainloader'):
            return next(iter(self.trainloader))[0].shape[1]
        elif pathlib.Path(self.datafiles[0]).suffix == '.h5ad':
            return an.read_h5ad(self.datafiles[0]).X.shape[1]
        else:
            return pd.read_csv(self.datafiles[0], nrows=1).shape[1]
    
# This has to be outside of the datamodule 
# Since we have to download the files to calculate the gene intersection 
def prepare_data(
    data_path: str, 
    datafiles: List[str], 
    labelfiles: List[str],
) -> None:
    """
    Prepare data for model training, by downloading the transposed and clean labels from the S3 bucket

    :param data_path: Path to the top-level folder containing the data subfolders
    :type data_path: str
    :param datafiles: List of absolute paths to datafiles 
    :type datafiles: List[str]
    :param labelfiles: List of absolute paths to labelfiles
    :type labelfiles: List[str]
    """    
    os.makedirs(os.path.join(data_path, 'interim'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'processed', 'labels'), exist_ok=True)

    for datafile, labelfile in zip(datafiles, labelfiles):
        if not os.path.isfile(datafile):
            print(f'Downloading {datafile}')
            download(
                remote_name=os.path.join('jlehrer/expression_data/interim/', datafile.split('/')[-1]),
                file_name=datafile,
            )
        else:
            print(f'{datafile} exists, continuing...')

        if not os.path.isfile(labelfile):
            print(f'Downloading {labelfile}')
            download(
                remote_name=os.path.join('jlehrer/expression_data/labels/', labelfile.split('/')[-1]),
                file_name=labelfile,
            )
        else:
            print(f'{labelfile} exists, continuing...\n')    

def generate_trainer(
    datafiles: List[str],
    labelfiles: List[str],
    class_label: str,
    weighted_metrics: bool,
    batch_size: int,
    num_workers: int,
    optim_params: Dict[str, Any],
    wandb_name='',
    *args,
    **kwargs,
):
    """
    Generates PyTorch Lightning trainer and datasets for model training.

    :param datafiles: List of absolute paths to datafiles
    :type datafiles: List[str]
    :param labelfiles: List of absolute paths to labelfiles
    :type labelfiles: List[str]
    :param class_label: Class label to train on 
    :type class_label: str
    :param weighted_metrics: To use weighted metrics in model training 
    :type weighted_metrics: bool
    :param batch_size: Batch size in dataloader
    :type batch_size: int
    :param num_workers: Number of workers in dataloader
    :type num_workers: int
    :param optim_params: Dictionary defining optimizer and any needed/optional arguments for optimizer initializatiom
    :type optim_params: Dict[str, Any]
    :param wandb_name: Name of run in Wandb.ai, defaults to ''
    :type wandb_name: str, optional
    :return: Trainer, model, datamodule 
    :rtype: Trainer, model, datamodule 
    """

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device is {device}')
    
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', '..', 'data')

    wandb_logger = WandbLogger(
        project=f"tabnet-classifer-sweep",
    )

    uploadcallback = UploadCallback(
        path=os.path.join(here, 'checkpoints'),
        desc=f'TabNet Gene Classifier'
    )

    earlystoppingcallback = EarlyStopping(
        monitor="val_loss",
        patience=50,
        verbose=True
    )

    prepare_data(
        data_path=data_path,
        datafiles=datafiles,
        labelfiles=labelfiles,
    )

    refgenes = gene_intersection()
    module = DataModule(
        datafiles=datafiles, 
        labelfiles=labelfiles, 
        class_label=class_label, 
        refgenes=refgenes,
        batch_size=batch_size,
        num_workers=num_workers,
        *args,
        **kwargs,
    )

    model = GeneClassifier(
        input_dim=len(refgenes),
        output_dim=19,
        weighted_metrics=weighted_metrics,
        optim_params=optim_params
    )
    
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        auto_lr_find=False,
        gradient_clip_val=0.5,
        logger=wandb_logger,
        callbacks=[
            uploadcallback, 
            earlystoppingcallback,
        ],
        max_epochs=kwargs['max_epochs'],
        val_check_interval=0.25, # Calculate validation every quarter epoch instead of full since dataset is large, and would like to test this 
        profiler="advanced",
    )

    return trainer, model, module

