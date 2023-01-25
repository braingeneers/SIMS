import pathlib
from typing import Union

import anndata as an
import pytorch_lightning as pl

from scsims.lightning_train import DataModule
from scsims.model import SIMSClassifier

here = pathlib.Path(__file__).parent.absolute()


class SIMS:
    def __init__(
        self,
        adata: Union[an.AnnData, list[an.AnnData]],
        labels_key: str,
        verbose=True,
        *args,
        **kwargs,
    ) -> None:
        self.adata = adata
        self.labels_key = labels_key
        self.verbose = verbose

        self.datamodule = DataModule(
            datafiles=[self.adata]
            if isinstance(self.adata, an.AnnData)
            else self.adata,  # since datamodule expects a list of data always
            label_key=labels_key,
            class_label=self.labels_key,
            *args,
            **kwargs,
        )

        for att, value in self.datamodule.__dict__.items():
            setattr(self, att, value)

    def setup_model(self, *args, **kwargs):
        self.model = SIMSClassifier(self.datamodule.input_dim, self.datamodule.output_dim, *args, **kwargs)

    def setup_trainer(self, *args, **kwargs):
        self.trainer = pl.Trainer(
            *args,
            **kwargs,
        )

    def train(self, *args, **kwargs):
        if not hasattr(self, "trainer"):
            self.setup_trainer()
        if not hasattr(self, "model"):
            self.setup_model()

        self.trainer.fit(self.model, datamodule=self.datamodule)

    def predict(self, adata: an.AnnData, *args, **kwargs):
        results = self.model.predict(adata, *args, **kwargs)
        results = results.apply(lambda x: self.label_encoder(x))

        return results

    def explain(self, adata: an.AnnData, *args, **kwargs):
        results = self.model.explain(adata, *args, **kwargs)
        
        return results