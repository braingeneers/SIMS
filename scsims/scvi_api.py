import pathlib
from typing import Union

import anndata as an
import pytorch_lightning as pl

from scsims.lightning_train import DataModule
from scsims.model import SIMSClassifier

here = pathlib.Path(__file__).parent.absolute()

class UnconfiguredModelError(Exception):
    pass

class SIMS:
    def __init__(
        self,
        datafiles: Union[list[str], list[an.AnnData]] = None,
        weights_path: str = None,
        *args,
        **kwargs,
    ) -> None:
        print('Setting up data module ...')
        if weights_path is not None:
            self._model = SIMSClassifier.load_from_checkpoint(weights_path, *args, **kwargs)

        self.datamodule = DataModule(
            datafiles=[datafiles] if isinstance(datafiles, an.AnnData) else datafiles, 
            *args,
            **kwargs,
        )

        for att, value in self.datamodule.__dict__.items():
            setattr(self, att, value)

    def setup_model(self, *args, **kwargs):
        print('Setting up model ...')
        self._model = SIMSClassifier(self.datamodule.input_dim, self.datamodule.output_dim, *args, **kwargs)

    def setup_trainer(self, *args, **kwargs):
        print('Setting up trainer ...')
        self._trainer = pl.Trainer(*args, **kwargs)

    def train(self, *args, **kwargs):
        print('Beginning training')
        if not hasattr(self, "_trainer"):
            self.setup_trainer(*args, **kwargs)
        if not hasattr(self, "_model"):
            self.setup_model(*args, **kwargs)

        self._trainer.fit(self._model, datamodule=self.datamodule)

        print('Finished training')

    def predict(self, inference_data: an.AnnData, *args, **kwargs):
        if not hasattr(self, '_model'):
            raise UnconfiguredModelError(
                """The model attribute is not configured. This is likely 
                because you are running the predict method after re-initializing the 
                SIMS class. Reinitialize the SIMS class with the weights_path
                pointing to the .ckpt file to continue."""
            )
        results = self._model.predict(inference_data, *args, **kwargs)
        try:
            results = results.apply(lambda x: self.label_encoder.inverse_transform(x))
        except Exception as e:
            if "has no attribute" in str(e):
                print("""
                    Unable to encoder numeric predictions back to class labels, since the original
                    labelfile and class_label column were not passed upon initialization. Alternatively, use 
                    SIMS.decode_predictions([labelfiles]) to convert the numeric labels to string names.
                """)
        self.results = results 

        print('Finished prediction, returning results and storing in results attribute ...')
        return results

    def explain(self, datafiles: an.AnnData, *args, **kwargs):
        print('Computing explainability matrix ...')
        results = self._model.explain(datafiles, *args, **kwargs)
        
        return results

    def decode_predictions(predictions, labelfiles, class_labels, datafiles, sep=None):
        labelencoder, _ = DataModule.get_unique_targets(labelfiles, sep, class_labels, datafiles)

        return labelencoder.inverse_transform(predictions)
