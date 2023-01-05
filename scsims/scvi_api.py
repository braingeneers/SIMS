import pathlib
from os.path import join

import anndata as an
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from scsims.lightning_train import DataModule
from scsims.model import SIMSClassifier

here = pathlib.Path(__file__).parent.absolute()


class SIMS:
    def __init__(
        self,
        adata,
        labels_key,
        verbose=True,
    ) -> None:
        self.adata = adata
        self.labels_key = labels_key
        self.verbose = verbose

    def setup_data(self, *args, **kwargs):
        if self.verbose:
            print("Setting up label file for training")
        self.labels = pd.DataFrame(an.read_h5ad(self.adata, backed="r+").obs[self.labels_key])
        self.labels.to_csv(join(here, "_temp_labels.csv"), index=True)

        if self.verbose:
            print("Setting up DataModule")
        self.datamodule = DataModule(
            datafiles=[self.adata],
            labelfiles=[join(here, "_temp_labels.csv")],
            class_label=self.labels_key,
            *args,
            **kwargs,
        )

        self.datamodule.prepare_data()
        self.datamodule.setup()
        self.label_encoder = self.datamodule.label_encoder

    def setup_model(self, *args, **kwargs):
        self.model = SIMSClassifier(self.datamodule.input_dim, self.datamodule.output_dim, *args, **kwargs)

    def setup_trainer(self, *args, **kwargs):
        self.trainer = pl.Trainer(
            *args,
            **kwargs,
            max_epochs=1000,
        )

    def setup(self):
        self.setup_data()
        self.setup_model()
        self.setup_trainer()

    def train(self, *args, **kwargs):
        if not hasattr(self, "datamodule"):
            self.setup_data()
        if not hasattr(self, "trainer"):
            self.setup_trainer()
        if not hasattr(self, "model"):
            self.setup_model()

        self.trainer.fit(self.model, datamodule=self.datamodule)

    def predict(self, loader):
        results = self.trainer.predict(self.model, loader)
        results = [torch.argmax(output[0], dim=1) for output in results]

        return self.label_encoder.inverse_transform(results)
