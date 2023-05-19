import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=None,
        output_dim=None,
        layers=None,
    ) -> None:
        super().__init__()

        if all((input_dim, output_dim, layers)) == None:
            raise ValueError("If layers aren't specified, input and output dimensions must be")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encode = (
            layers
            if layers is not None
            else nn.Sequential(
                nn.Linear(input_dim, 10000),
                nn.ReLU(),
                nn.Linear(10000, 5000),
                nn.ReLU(),
                nn.Linear(5000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, output_dim),
            )
        )

    def forward(self, x):
        return self.encode(x)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim=None,
        input_dim=None,
        layers=None,
    ) -> None:
        super().__init__()

        if all((output_dim, input_dim, layers)) == None:
            raise ValueError("If layers aren't specified, output_dim and input_dim must be.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.decode = (
            layers
            if layers is not None
            else nn.Sequential(
                nn.Linear(input_dim, 500),
                nn.ReLU(),
                nn.Linear(500, 1000),
                nn.ReLU(),
                nn.Linear(1000, 5000),
                nn.ReLU(),
                nn.Linear(5000, 10000),
                nn.ReLU(),
                nn.Linear(10000, output_dim),
            )
        )

    def forward(self, x):
        return self.decode(x)


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        data_shape=None,
        encoder_layers=None,
        decoder_layers=None,
        optim_params=None,
        scheduler_params=None,
        loss=None,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(input_dim=data_shape, layers=encoder_layers)
        self.decoder = Decoder(output_dim=data_shape, layers=decoder_layers)

        self.loss = loss if loss is not None else nn.MSELoss()
        self.optim_params = (
            optim_params
            if optim_params is not None
            else {
                "optimizer": torch.optim.Adam,
                "lr": 0.001,
                "weight_decay": 0.001,
            }
        )

        self.scheduler_params = (
            scheduler_params
            if scheduler_params is not None
            else {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "factor": 0.75,  # Reduce LR by 25% on plateau
            }
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def configure_optimizers(self):
        if "optimizer" in self.optim_params:
            optimizer = self.optim_params.pop("optimizer")
            optimizer = optimizer(self.parameters(), **self.optim_params)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-5)

        if self.scheduler_params is not None:
            scheduler = self.scheduler_params.pop("scheduler")
            scheduler = scheduler(optimizer, **self.scheduler_params)

        if self.scheduler_params is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def _step(self, tag, batch):
        x_hat = self.forward(batch)
        loss = self.loss(batch, x_hat)

        self.log(tag, loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train_loss", batch)

    def validation_step(self, batch, batch_idx):
        return self._step("val_loss", batch)

    def test_step(self, batch, batch_idx):
        return self._step("test_loss", batch)
