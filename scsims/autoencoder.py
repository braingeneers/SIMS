import torch
import pytorch_lightning as pl
import torch.nn as nn
import warnings
from torch.utils.data import Dataset
from scipy.sparse import issparse
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim=250, layers=None) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim > 500:
            warnings.warn(
                "Output_dim < 500, final layer has increasing dimensionality in Encoder")

        self.encode = layers if layers is not None else nn.Sequential(
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

    def forward(self, x):
        return self.encode(x)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        input_dim=250,
        layers=None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if output_dim < 10000:
            warnings.warn(
                f"Output_dim < 10000 ({output_dim, input_dim}) in Decoder")

        self.decode = layers if layers is not None else nn.Sequential(
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

    def forward(self, x):
        return self.decode(x)


class AutoEncoder(pl.LightningModule):
    def __init__(
        self, 
        data_shape, 
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
        self.optim_params = optim_params if optim_params is not None else {
            'optimizer': torch.optim.Adam,
            'lr': 0.001,
            'weight_decay': 0.001,
        }

        self.scheduler_params = scheduler_params if scheduler_params is not None else {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'factor': 0.75,  # Reduce LR by 25% on plateau
        }

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def configure_optimizers(self):
        if 'optimizer' in self.optim_params:
            optimizer = self.optim_params.pop('optimizer')
            optimizer = optimizer(self.parameters(), **self.optim_params)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=0.01, weight_decay=1e-5
            )

        if self.scheduler_params is not None:
            scheduler = self.scheduler_params.pop('scheduler')
            scheduler = scheduler(optimizer, **self.scheduler_params)

        if self.scheduler_params is None:
            return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
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

class AEDataset(Dataset):
    def __init__(self, matrix) -> None:
        super().__init__()
        self.matrix = np.asarray(matrix.todense()) if issparse(matrix) else matrix

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            it = list(range(idx.start or 0, idx.stop or len(self), idx.step or 1))
            return [self[i] for i in it]

        data = self.matrix[idx]
        return torch.from_numpy(data)

    def __len__(self):
        return self.matrix.shape[0]