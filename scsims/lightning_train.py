import os
import pathlib
import urllib
import warnings
from functools import cached_property
from os.path import join
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata as an
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import LabelEncoder

from .data import compute_class_weights, generate_dataloaders

here = pathlib.Path(__file__).parent.absolute()


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        class_label: str = None,
        datafiles: Union[list[str], list[an.AnnData]] = None,
        labelfiles: list[str] = None,
        urls: Dict[str, list[str]] = None,
        sep: str = None,
        unzip: bool = True,
        datapath: str = "",
        batch_size = 32,
        num_workers = 0,
        device=("cuda:0" if torch.cuda.is_available() else None),
        split=True,
        *args,
        **kwargs,
    ):
        """
        Creates the DataModule for PyTorch-Lightning training.

        :param class_label: Class label to train on. Must be in all label files
        :type class_label: str
        :param datafiles: List of absolute paths to datafiles, if not using URLS. defaults to None
        :type datafiles: List[str], optional
        :param labelfiles: List of absolute paths to labelfiles, if not using URLS. defaults to None
        :type labelfiles: List[str], optional
        :param label_key: if no label file is provided, this is the key in the anndata.obs that corresponds to the class
        label to train on
        :param urls: Dictionary of URLS to download, as specified in the above docstring, defaults to None
        :type urls: Dict[str, List[str, str]], optional
        :param unzip: Boolean, whether to unzip the datafiles in the url, defaults to False
        :type unzip: bool, optional
        :param sep: Separator to use in reading in both datafiles and labelfiles. WARNING: Must be homogeneous between all datafile and labelfiles, defaults to '\t'
        :type sep: str, optional
        :param datapath: Path to local directory to download datafiles and labelfiles to, if using URL. defaults to None
        :type datapath: str, optional
        :param assume_numeric_label: If the class_label column in all labelfiles is numeric. Otherwise, we automatically apply sklearn.preprocessing.LabelEncoder to the intersection of all possible labels, defaults to True
        :type assume_numeric_label: bool, optional
        :raises ValueError: If both a dictionary of URL's is passed and labelfiles/datafiles are passed. We can only handle one, not a mix of both, since there isn't a way to determine easily if a string is an external url or not.
        """
        super().__init__()
        # Make sure we don't have datafiles/labelfiles AND urls at start
        if urls is not None and datafiles is not None or urls is not None and labelfiles is not None:
            raise ValueError(
                "Either a dictionary of data to download, or paths to datafiles and labelfiles are supported, but not both."
            )

        if labelfiles is not None or datafiles is not None:
            assert class_label is not None, "If labelfiles/datafiles are passed, we need a class label for training."

        self.device = device
        self.class_label = class_label
        self.urls = urls
        self.unzip = unzip
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.sep = None

        # If we have a list of urls, we can generate the list of paths of datafiles/labelfiles that will be downloaded after self.prepare_data()
        if self.urls is not None:
            self.datafiles = [join(self.datapath, f) for f in self.urls.keys()]
            self.labelfiles = [join(self.datapath, f"labels_{f}") for f in self.urls.keys()]
        else:
            self.datafiles = datafiles
            self.labelfiles = labelfiles

        if self.labelfiles is not None:
            # Warn user in case tsv/csv ,/\t don't match, this can be annoying to diagnose
            suffix = pathlib.Path(self.labelfiles[0]).suffix
            if (sep == "\t" and suffix == "csv") or (sep == "," and suffix == ".tsv"):
                warnings.warn(f"Passed delimiter sep={sep} doesn't match file extension, continuing...")

            # Infer sep based on .csv/.tsv of labelfile (assumed to be homogeneous in case of delimited datafiles) if sep is not passed
            if sep is None:
                if suffix == ".tsv":
                    self.sep = "\t"
                elif suffix == ".csv":
                    self.sep = ","
                else:
                    warnings.warn(f'Separator not passed and not able to be inferred from suffix={suffix}. Falling back to ","')
                    self.sep = ","
            else:
                self.sep = sep

        self.args = args
        self.kwargs = kwargs

        self.prepared = False
        self.setuped = False

        self.prepare_data()
        self.setup()

    @staticmethod
    def get_unique_targets(labelfiles, sep, class_label, datafiles):
        if labelfiles is not None:
            unique_targets = np.array(
                list(
                    set(
                        np.concatenate(
                            [pd.read_csv(df, sep=sep).loc[:, class_label].unique() for df in labelfiles]
                        )
                    )
                )
            )
        elif isinstance(datafiles[0], an.AnnData):
            unique_targets = np.array(
                list(set(np.concatenate([datafile.obs.loc[:, class_label].unique() for datafile in datafiles])))
            )
        else:
            raise NotImplementedError(
                "Need to implement support for handling csv/tsv files in the generate_dataloaders class first."
            )

        print("Building LabelEncoder.")
        label_encoder = LabelEncoder().fit(unique_targets)

        return label_encoder, unique_targets

    def prepare_data(self):
        labelencoder, unique_targets = DataModule.get_unique_targets(
            self.labelfiles,
            self.sep,
            self.class_label,
            self.datafiles,
        )

        self.label_encoder = labelencoder
        if not np.issubdtype(unique_targets.dtype, np.number):
            print(f"Encoding labels, training on new encoded column: numeric_{self.class_label}")
            if self.labelfiles is not None:
                for idx, file in enumerate(self.labelfiles):
                    print(f"Transforming labelfile {idx + 1}/{len(self.labelfiles)}")

                    labels = pd.read_csv(file, sep=self.sep)

                    labels.loc[:, f"numeric_{self.class_label}"] = self.label_encoder.transform(
                        labels.loc[:, self.class_label]
                    ).astype(int)

                    # Don't need to re-index here
                    labels.to_csv(file, index=False, sep=self.sep)
            else:
                for idx, data in enumerate(self.datafiles):
                    print(f"Transforming datafile {idx + 1}/{len(self.datafiles)}")
                    data.obs.loc[:, f"numeric_{self.class_label}"] = self.label_encoder.transform(
                        data.obs.loc[:, self.class_label]
                    ).astype(int)

            self.class_label = f"numeric_{self.class_label}"

    def setup(self, stage: Optional[str] = None):
        if not self.setuped and self.datafiles is not None:
            if isinstance(self.datafiles[0], str):
                suffix = pathlib.Path(self.datafiles[0]).suffix
                if suffix == ".tsv" or suffix == ".csv":
                    raise NotImplementedError(
                        "Need to implement support for handling csv/tsv files in the generate_dataloaders class."
                    )

                self.datafiles = [an.read_h5ad(file, backed="r+") for file in self.datafiles]

            if self.split:
                print("Creating train/val/test DataLoaders...")
            else:
                print("Creating single dataloader (under trainloader attribute)")

            loaders = generate_dataloaders(
                datafiles=self.datafiles,
                labelfiles=self.labelfiles,
                class_label=self.class_label,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,  # For gpu training
                split=self.split,
                *self.args,
                **self.kwargs,
            )

            print("Done, continuing to training.")
            if len(loaders) > 1:
                trainloader, valloader, testloader = loaders
                self.trainloader = trainloader
                self.valloader = valloader
                self.testloader = testloader
            else:
                self.trainloader = loaders[0]
                self.valloader = None
                self.testloader = None

            print("Calculating weights")
            self.weights = compute_class_weights(
                labelfiles=self.labelfiles,
                class_label=self.class_label,
                datafiles=self.datafiles,
                sep=self.sep,
                device=self.device,
            )

            self.setuped = True

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader

    def test_dataloader(self):
        return self.testloader

    @cached_property
    def num_labels(self):
        val = []
        if self.labelfiles is not None:
            for file in self.labelfiles:
                val.extend(pd.read_csv(file, sep=self.sep).loc[:, self.class_label].unique())
        else:
            for file in self.datafiles:
                val.extend(file.obs.loc[:, self.class_label].unique())

        return len(set(val))

    @cached_property
    def num_features(self):
        if self.urls is not None and not os.path.isfile(self.datafiles[0]):
            print("Trying to calcuate num_features before data has been downloaded. Downloading and continuing...")
            self.prepare_data()

        if "refgenes" in self.kwargs:
            return len(self.kwargs["refgenes"])
        elif hasattr(self, "trainloader"):
            return next(iter(self.trainloader))[0].shape[1]
        elif pathlib.Path(self.datafiles[0]).suffix == ".h5ad":
            return an.read_h5ad(self.datafiles[0]).X.shape[1]
        else:
            return pd.read_csv(self.datafiles[0], nrows=1, sep=self.sep).shape[1]

    @cached_property
    def input_dim(self):
        return self.num_features

    @cached_property
    def output_dim(self):
        return self.num_labels
