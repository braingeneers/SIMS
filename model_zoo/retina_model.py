import os
import pathlib 
import sys
import anndata as an
import torch 
import argparse 
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from helper import download
from models.lib.data import *
from models.lib.neural import *
from models.lib.lightning_train import *

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

args = parser.parse_args()
lr, weight_decay, name = args.lr, args.weight_decay, args.name

here = pathlib.Path(__file__).parent.resolve()
data_path = join(here, '..', 'data', 'retina')

print('Making data folder')
os.makedirs(data_path, exist_ok=True)

for file in ['retina_T.h5ad', 'retina_labels_numeric.csv']:
    print(f'Downloading {file}')

    if not os.path.isfile(join(data_path, file)):
        download(
            remote_name=join('jlehrer', 'retina', file),
            file_name=join(data_path, file),
        )

# Define labelfiles and trainer
datafiles=[join(data_path, 'retina_T.h5ad')]
labelfiles=[join(data_path, 'retina_labels_numeric.csv')]

device = ('cuda:0' if torch.cuda.is_available() else None)

module = DataModule(
    datafiles=datafiles,
    labelfiles=labelfiles,
    class_label='class_label',
    index_col='cell',
    batch_size=16,
    num_workers=0,
    shuffle=True,
    drop_last=True,
    normalize=True,
    deterministic=True,
)

model = TabNetLightning(
    input_dim=module.num_features,
    output_dim=module.num_labels,
    optim_params={
        'optimizer': torch.optim.Adam,
        'lr': lr,
        'weight_decay': weight_decay,
    },
    scheduler_params={
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'factor': 0.75, # lr := 0.75 * lr
    },
    # weights=compute_class_weights(
    #     labelfiles=labelfiles,
    #     class_label='class_label',
    #     device=device
    # ),
    n_d=32,
    n_a=32,
    n_steps=10,
    weighted_metrics=True,
)

print(f'{module.num_features =}, {module.num_labels =}')

wandb_logger = WandbLogger(
    project=f"Retina Model",
    name=name,
)

lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=join(here, 'checkpoints'), 
    filename='{epoch}-{weighted_val_accuracy}'
)

upload_callback = UploadCallback(
    path='checkpoints',
    desc='retina'
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
        checkpoint_callback, 
        upload_callback,
        early_stopping_callback,
    ]
)

# train model
trainer.fit(model, datamodule=module)
trainer.test(model, datamodule=module)