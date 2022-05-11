import unittest 
import sys
import os 
import random 
import pandas as pd 
import anndata as an 
import pandas as pd 
import numpy as np
import torch
import pathlib 
import sklearn as sk 

import pytorch_lightning as pl 
from torch.utils.data import *

random.seed(42)
from os.path import join, dirname, abspath 
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from models.lib.data import *
from models.lib.lightning_train import *
from models.lib.neural import *
from helper import *

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.here = pathlib.Path(__file__).parent.resolve()

        cls.datapath = os.path.join(cls.here, 'datapath')
        if not os.path.isdir(cls.datapath):
            print(f'Making test directories')
            os.makedirs(cls.datapath, exist_ok=True)

        # Define test file locations 
        cls.datafile = os.path.join(cls.datapath, 'test_data_pipeline.csv')
        cls.labelfile = os.path.join(cls.datapath, 'test_data_pipeline.csv')

    def train_pipeline(self):
        # Generate synthetic data 
        data = sk.datasets.make_classification()

        df, labels = data
        df = pd.DataFrame(df)
        labels = pd.DataFrame(df)
        labels.index.name = 'class_label'

        df.to_csv(self.datafile, index=False)
        labels.to_csv(self.labelfile, index=False)

        module = DataModule(
            datafiles=self.datafile,
            labelfiles=self.labelfile,
            class_label='class_label',
            batch_size=4,
            num_workers=0,
        )

        model = TabNetLightning(
            input_dim=module.num_features,
            output_dim=module.num_labels,
        )

        trainer = pl.Trainer(
            gpus=(1 if torch.cuda.is_available() else 0),
            auto_lr_find=False,
            max_epochs=1,
        )

        trainer.fit(model, datamodule=module)

if __name__ == "__main__":
    unittest.main()
