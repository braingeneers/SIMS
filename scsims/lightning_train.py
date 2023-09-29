import pathlib
from functools import cached_property
from typing import Optional

import anndata as an
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder

from .data import generate_dataloaders
from sklearn.utils.class_weight import compute_class_weight

here = pathlib.Path(__file__).parent.absolute()

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: an.AnnData = None,
        class_label: str = None,
        test_size: float = 0.2,
        batch_size = 32,
        num_workers = 0,
        device=("cuda:0" if torch.cuda.is_available() else None),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.class_label = class_label
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = data
        self.test_size = test_size

        self.args = args
        self.kwargs = kwargs

        self.prepared = False
        self.setuped = False

        if data is not None:
            self.prepare_data()
            self.setup()

    def prepare_data(self):
        unique_targets = self.data.obs.loc[:, self.class_label].unique()
        label_encoder = LabelEncoder().fit(unique_targets)

        self.label_encoder = label_encoder

        if not pd.api.types.is_numeric_dtype(self.data.obs.loc[:, self.class_label]):
            print("Numerically encoding class labels")
            self.data.obs.loc[:, f"numeric_{self.class_label}"] = self.label_encoder.transform(
                self.data.obs.loc[:, self.class_label]
            ).astype(int)

            self.class_label = f"numeric_{self.class_label}"

    def setup(self, stage: Optional[str] = None):
        loaders = generate_dataloaders(
            data=self.data,
            class_label=self.class_label,
            test_size=self.test_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            split=self.test_size is not None,
            *self.args,
            **self.kwargs,
        )

        if len(loaders) == 1:
            self.trainloader = loaders[0]
            self.valloader = None
            self.testloader = None
        else: 
            self.trainloader, self.valloader, self.testloader = loaders

        print("Calculating weights")
        labels = self.data.obs.loc[:, self.class_label].values
        self.weights = torch.from_numpy(
            compute_class_weight(
                y=labels,
                classes=np.unique(labels),
                class_weight="balanced",
            )
        ).float().to(self.device)


        self.setuped = True

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader

    def test_dataloader(self):
        return self.testloader

    @cached_property
    def num_labels(self):
        return self.data.obs.loc[:, self.class_label].nunique()

    @cached_property
    def num_features(self):
        return self.data.shape[1]

    @cached_property
    def input_dim(self):
        return self.num_features

    @cached_property
    def output_dim(self):
        return self.num_labels

    @cached_property
    def genes(self):
        return self.data.var.index.tolist()
    
    @cached_property
    def cells(self):
        return self.data.obs.index.tolist()