import sys
import argparse
import pathlib
import os
import ast
from typing import *
from lib.lightning_train import generate_trainer
import torch 

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.lib.data import GeneExpressionData
from models.lib.neural import GeneClassifier
import helper

from os.path import join, dirname, abspath

from helper import download, list_objects
from models.lib.neural import GeneClassifier
from models.lib.lightning_train import DataModule, generate_trainer

if __name__ == "__main__":
    data_path = join(pathlib.Path(__file__).parent.resolve(), '..', 'data', 'interim')

    for file in ['retina_T.csv', 'retina_labels_numeric.csv']:
        print(f'Downloading {file}')

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join('jlehrer', 'retina_data', file),
                file_name=join(data_path, file),
            )