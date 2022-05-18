import os
import pathlib 
import sys
import torch 
import argparse 
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from data import *
from model import *
from lightning_train import *
from networking import download 

here = pathlib.Path(__file__).parent.resolve()
data_path = join(here, '..', 'data', 'retina')

def explain_retina():
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
        num_workers=32,
        shuffle=True,
        drop_last=True,
        normalize=True,
        deterministic=True,
    )

    module.setup()

    download(
        remote_name=join('jlehrer', 'model_checkpoints', 'checkpoint-80-desc-retina.ckpt'),
        file_name=join(data_path, 'checkpoint-80-desc-retina.ckpt')
    )
    model = SIMSClassifier.load_from_checkpoint(
        join(data_path, 'checkpoint-80-desc-retina.ckpt'),
        input_dim=module.input_dim,
        output_dim=module.output_dim,
        n_d=32,
        n_a=32,
        n_steps=10,
    )

    loader = module.trainloader
    mask = model.explain(loader)
    mask.tofile(
        join(data_path, 'retina_mask.npy')
    )

    upload(
        file_name=join(data_path, 'retina_mask.npy'),
        remote_name=join(data_path, 'retina_mask.npy')
    )

def explain_dental():
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

    class_label = 'cell_type'
    module = DataModule(
        datafiles=[join(data_path, 'human_dental_T.h5ad')],
        labelfiles=[join(data_path, 'labels_human_dental.tsv')],
        class_label=class_label,
        sep='\t',
        batch_size=16,
        num_workers=32,
        deterministic=True,
        normalize=True,
    )

    download(
        remote_name=join('jlehrer', 'model_checkpoints', 'checkpoint-80-desc-dental.ckpt'),
        file_name=join(data_path, 'checkpoint-80-desc-dental.ckpt')
    )

    model = SIMSClassifier.load_from_checkpoint(
        join(data_path, 'checkpoint-80-desc-dental.ckpt'),
        input_dim=module.input_dim,
        output_dim=module.output_dim,
    )

    loader = module.trainloader
    mask = model.explain(loader)
    mask.tofile(
        join(data_path, 'dental_mask.npy')
    )

    upload(
        file_name=join(data_path, 'dental_mask.npy'),
        remote_name=join(data_path, 'dental_mask.npy')
    )

def explain_mouse():
    here = pathlib.Path(__file__).parent.resolve()
    data_path = join(here, '..', 'data', 'benchmark')

    print('Making data folder')
    os.makedirs(data_path, exist_ok=True)

    for file in ['mouse_labels_clean.csv', 'mouse_clipped.h5ad']:
        print(f'Downloading {file}')

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join('jlehrer', 'mouse_benchmark', file),
                file_name=join(data_path, file),
            )

    class_label = 'subclass_label'
    module = DataModule(
        datafiles=[join(data_path, 'mouse_clipped.h5ad')],
        labelfiles=[join(data_path, 'mouse_labels_clean.csv')],
        class_label=class_label,
        sep=',',
        batch_size=32,
        index_col='cell',
        num_workers=32,
        deterministic=True,
        normalize=True,
        assume_numeric_label=False,
    )
    download(
        remote_name=join('jlehrer', 'model_checkpoints', 'checkpoint-80-desc-dental.ckpt'),
        file_name=join(data_path, 'checkpoint-80-desc-dental.ckpt')
    )

    model = SIMSClassifier.load_from_checkpoint(
        join(data_path, 'checkpoint-80-desc-dental.ckpt'),
        input_dim=module.input_dim,
        output_dim=module.output_dim,
    )

    loader = module.trainloader
    mask = model.explain(loader)
    mask.tofile(
        join(data_path, 'mouse_cortical_mask.npy')
    )

    upload(
        file_name=join(data_path, 'mouse_cortical_mask.npy'),
        remote_name=join(data_path, 'mouse_cortical_mask.npy')
    )

def explain_human():
    here = pathlib.Path(__file__).parent.resolve()
    data_path = join(here, '..', 'data', 'benchmark')

    print('Making data folder')
    os.makedirs(data_path, exist_ok=True)

    for file in ['human_labels_clean.csv', 'human.h5ad']:
        print(f'Downloading {file}')

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join('jlehrer', 'human_benchmark', file),
                file_name=join(data_path, file),
            )

    class_label = 'subclass_label'
    module = DataModule(
        datafiles=[join(data_path, 'human.h5ad')],
        labelfiles=[join(data_path, 'human_labels_clean.csv')],
        class_label=class_label,
        sep=',',
        batch_size=32,
        index_col='cell',
        num_workers=0,
        deterministic=True,
        normalize=True,
        assume_numeric_label=False,
    )

    download(
        remote_name=join('jlehrer', 'model_checkpoints', 'checkpoint-20-desc-human_cortical.ckpt'),
        file_name=join(data_path, 'checkpoint-20-desc-human_cortical.ckpt')
    )

    model = SIMSClassifier.load_from_checkpoint(
        join(data_path, 'checkpoint-20-desc-human_cortical.ckpt'),
        input_dim=module.input_dim,
        output_dim=module.output_dim,
    )

    loader = module.trainloader
    mask = model.explain(loader)
    mask.tofile(
        join(data_path, 'human_cortical_mask.npy')
    )

    upload(
        file_name=join(data_path, 'human_cortical_mask.npy'),
        remote_name=join(data_path, 'human_cortical_mask.npy')
    )

if __name__ == "__main__":
    # print('Working on retina')
    # explain_retina()

    # print('Working on dental')
    # explain_dental()

    # print('Working on human cortical')
    # explain_human()

    print('Working on mouse cortical')
    explain_mouse()

