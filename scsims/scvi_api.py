import pandas as pd
import numpy as np 
import anndata as an 
import pytorch_lightning as pl 
import pathlib 

from tqdm import tqdm 
from os.path import join 
from .model import SIMSClassifier
from .lightning_train import DataModule

here = pathlib.Path(__file__).parent.absolute()

class SIMS:
    def __init__(
        self,
        adata,
        labels_key,
        verbose=True,
        *args,
        **kwargs,
    ) -> None:
        self.adata = adata 
        self.labels_key = labels_key

        if verbose: print('Setting up label file for training')
        self.labels = pd.DataFrame(an.read_h5ad(adata, backed='r+').obs[labels_key])
        self.labels.to_csv(join(here, '_temp_labels.csv'), index=True)

        if verbose: print('Setting up DataModule')
        self.datamodule = DataModule(
            datafiles=[adata],
            labelfiles=[join(here, '_temp_labels.csv')],
            class_label=labels_key,
            *args,
            **kwargs,
        )

    def setup_data(self):
        self.datamodule.prepare_data()
        self.datamodule.setup()

    def setup_model(self, *args, **kwargs):
        self.model = SIMSClassifier(
            self.datamodule.input_dim,
            self.datamodule.output_dim,
            *args, 
            **kwargs
        )

    def train(self, *args, **kwargs):
        if not hasattr(self, 'model'):
            raise ValueError("Cannot train without initialized model. Run setup_model() before training")

        trainer = pl.Trainer(*args, **kwargs)
        trainer.fit(self.model, datamodule=self.datamodule)

    def predict(self, loader):
        return self.datamodule.label_encoder(
            [self.model(X) for X in tqdm(loader)]
        )






