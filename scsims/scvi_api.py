import pathlib
from typing import Union

import anndata as an
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Timer, EarlyStopping

from scsims.lightning_train import DataModule
from scsims.model import SIMSClassifier
import pandas as pd 
from sklearn.preprocessing import LabelEncoder


here = pathlib.Path(__file__).parent.absolute()

class UnconfiguredModelError(Exception):
    pass

class SIMS:
    def __init__(
        self,
        *,
        weights_path: str = None,
        data: an.AnnData = None,
        class_label: str = None,
        **kwargs,
    ) -> None:
        if weights_path is not None:
            self.model = SIMSClassifier.load_from_checkpoint(weights_path, **kwargs, strict=False)

        self.datamodule = DataModule(data=data, class_label=class_label, **kwargs)

        for att, value in self.datamodule.__dict__.items():
            setattr(self, att, value)

    def setup_model(self, *args, **kwargs):
        print('Setting up model ...')
        if 'model_size' in kwargs:
            assert kwargs['model_size'] in ["tall", "grande", "venti", "trenta"], "if specified, model_size must be one of 'big', 'medium', or 'small'"
            if kwargs['model_size'] == "trenta":
                kwargs['n_a'] = 128
                kwargs['n_d'] = 128
            if kwargs['model_size'] == "venti":
                kwargs['n_a'] = 64
                kwargs['n_d'] = 64
            if kwargs['model_size'] == "grande":
                kwargs['n_a'] = 32
                kwargs['n_d'] = 32
            if kwargs['model_size'] == "tall":
                kwargs['n_a'] = 8
                kwargs['n_d'] = 8

        self.model = SIMSClassifier(
            input_dim=self.datamodule.input_dim, 
            output_dim=self.datamodule.output_dim, 
            genes=self.datamodule.genes,
            cells=self.datamodule.cells,
            label_encoder=self.datamodule.label_encoder,
            *args, 
            **kwargs
        )

    def setup_trainer(self, early_stopping_patience: int = None, *args, **kwargs):
        print('Setting up trainer ...')
        if 'callbacks' in kwargs:
            # check if any of the list of callbacks is a modelcheckpoint or timer
            callbacks = kwargs['callbacks']
            if not any([isinstance(callback, ModelCheckpoint) for callback in callbacks]):
                callbacks.append(ModelCheckpoint())
            if not any([isinstance(callback, Timer) for callback in callbacks]):
                callbacks.append(Timer())
            if not any([isinstance(callback, EarlyStopping) for callback in callbacks]) and early_stopping_patience is not None:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience))
        else:
            kwargs['callbacks'] = [ModelCheckpoint(), Timer()]
            if early_stopping_patience is not None:
                kwargs['callbacks'].append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience))
        self._trainer = pl.Trainer(*args, **kwargs)

    def train(self, *args, **kwargs):
        print('Beginning training')
        if not hasattr(self, "_trainer"):
            self.setup_trainer(*args, **kwargs)
        if not hasattr(self, "model"):
            self.setup_model(*args, **kwargs)

        self._trainer.fit(self.model, datamodule=self.datamodule)

        print('Finished training')

    def predict(self, inference_data: an.AnnData, *args, **kwargs):
        if not hasattr(self, 'model'):
            raise UnconfiguredModelError(
                """The model attribute is not configured. This is likely 
                because you are running the predict method after re-initializing the 
                SIMS class. Reinitialize the SIMS class with the weights_path
                pointing to the .ckpt file to continue."""
            )
        self.results = self.model.predict(inference_data, *args, **kwargs)
        return self.results

    def explain(self, datafiles: an.AnnData, labelfile=None, class_label=None, *args, **kwargs):
        print('Computing explainability matrix ...')
        results = self.model.explain(datafiles, *args, **kwargs)

        return results

    def decode_predictions(self, labelfile: str, class_label: str):
        labels = pd.read_csv(labelfile)
        label_encoder = LabelEncoder()
        label_encoder.fit(labels[class_label])
        results = self.results
        try:
            pred_columns = [col for col in results.columns if 'pred' in col]
            results[pred_columns] = results[pred_columns].apply(lambda x: label_encoder.inverse_transform(x))
            #results = self.results.apply(lambda x: label_encoder.inverse_transform(x))
        except AttributeError:
            raise AttributeError(
                """The results attribute is not configured. This is likely 
                because you are running the decode_predictions method before running the predict method.
                Run the predict method first, then run the decode_predictions method."""
            )
        return results

    def set_temperature(self, dataloader):
        self.model = self.model.set_temperature(dataloader)