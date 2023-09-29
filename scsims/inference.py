import linecache
from functools import cached_property
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset
import anndata as an
from typing import Union

class DatasetForInference(Dataset):
    def __init__(self, adata: Union[an.AnnData, np.ndarray]) -> None:
        super().__init__()
        self.data = adata.X if isinstance(adata, an.AnnData) else adata

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            it = list(range(idx.start or 0, idx.stop or len(self), idx.step or 1))
            return [self[i] for i in it]

        data = self.data[idx, :]
        if issparse(data):
            data = data.todense()

        data = torch.from_numpy(data)
        return data

    def __len__(self):
        return self.data.shape[0]

    @property
    def shape(self):
        return self.data.shape