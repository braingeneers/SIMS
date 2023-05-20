import linecache
from functools import cached_property
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset


class MatrixDatasetWithLabelsFile(Dataset):
    def __init__(
        self,
        datafile: str,
        indexfile: str = None,
        index_col: str = None,
        skip=3,
        cast=True,
        sep=",",
        columns: List[Any] = None,
        *args,
        **kwargs,  # To handle extraneous inputs
    ):
        self.datafile = datafile
        self.skip = skip
        self.cast = cast
        self.sep = sep
        self._cols = columns

        if indexfile is not None:
            self.index_df = pd.read_csv(indexfile, index_col=index_col).reset_index()

    def __getitem__(self, idx: int):
        """Get sample at index

        :param idx: Numerical index between 0, len(self) -1
        :type idx: int
        :raises ValueError: Errors in the case of unbounded slicing, which is normally supported in this method
        :return: Returns a data, label sample
        :rtype: Tuple[torch.Tensor, Any]
        """
        # Handle slicing
        if isinstance(idx, slice):
            if idx.start is None or idx.stop is None:
                raise ValueError(
                    f"Error: Unlike other iterables, {self.__class__.__name__} does not support unbounded slicing since samples are being read as needed from disk, which may result in memory errors."
                )

            step = 1 if idx.step is None else idx.step
            idxs = range(idx.start, idx.stop, step)
            return [self[i] for i in idxs]

        # The actual line in the datafile to get, corresponding to the number in the self.index_col values, otherwise the line at index "idx"
        data_index = self._labeldf.loc[idx, self.index_col] if self.index_col is not None else idx

        # get gene expression for current cell from csv file
        # We skip some lines because we're reading directly from
        line = linecache.getline(self.datafile, data_index + self.skip)

        if self.cast:
            data = torch.from_numpy(np.array(line.split(self.sep), dtype=np.float32)).float()
        else:
            data = np.array(line.split(self.sep))

        return data

    def __len__(self):
        return len(self._labeldf)  # number of total samples

    @cached_property
    def columns(self):  # Just an alias...
        return self.features

    # Worth caching, since this is a list comprehension on up to 50k strings. Annoying.
    @cached_property
    def features(self):
        if self._cols is not None:
            return self._cols
        else:
            data = linecache.getline(self.datafile, self.skip - 1)
            data = [x.split("|")[0].upper().strip() for x in data.split(self.sep)]
            return data

    @property
    def shape(self):
        return (self.__len__(), len(self.features))


class MatrixDatasetWithoutLabels(Dataset):
    def __init__(self, matrix, transforms=None) -> None:
        super().__init__()
        self.matrix = np.asarray(matrix.todense()) if issparse(matrix) else matrix
        self.transforms = transforms

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            it = list(range(idx.start or 0, idx.stop or len(self), idx.step or 1))
            return [self[i] for i in it]

        data = self.matrix[idx]
        data = torch.from_numpy(data)
        return data if self.transforms is None else self.transforms(data)

    def __len__(self):
        return self.matrix.shape[0]
    
    @property
    def shape(self):
        return self.matrix.shape

class AnnDatasetForInference(Dataset):
    def __init__(self, adata) -> None:
        super().__init__()
        self.adata = adata

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            it = list(range(idx.start or 0, idx.stop or len(self), idx.step or 1))
            return [self[i] for i in it]

        data = self.adata.X[idx, :]
        data = torch.from_numpy(data)
        return data

    def __len__(self):
        return self.adata.shape[0]

    @property
    def shape(self):
        return self.adata.shape