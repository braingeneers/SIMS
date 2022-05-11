import linecache
from multiprocessing.sharedctypes import Value 
from typing import *
from functools import cached_property, partial
from itertools import chain 
import inspect
import warnings 
import pathlib

import pandas as pd 
import torch
import numpy as np
import scanpy as sc 

from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import issparse
import pytorch_lightning as pl 

import sys, os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

class DelimitedTestData(Dataset):
    def __init__(
        self, 
        datafile: str,
        indexfile: str=None,
        index_col: str=None,
        skip=3,
        cast=True,
        sep=',',
        columns: List[any]=None,
        *args,
        **kwargs, # To handle extraneous inputs
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

            step = (1 if idx.step is None else idx.step)
            idxs = range(idx.start, idx.stop, step)
            return [self[i] for i in idxs]

        # The actual line in the datafile to get, corresponding to the number in the self.index_col values, otherwise the line at index "idx"
        data_index = (
            self._labeldf.loc[idx, self.index_col] if self.index_col is not None else idx
        )

        # get gene expression for current cell from csv file
        # We skip some lines because we're reading directly from 
        line = linecache.getline(self.datafile, data_index + self.skip)
        
        if self.cast:
            data = torch.from_numpy(np.array(line.split(self.sep), dtype=np.float32)).float()
        else:
            data = np.array(line.split(self.sep))

        return data 

    def __len__(self):
        return len(self._labeldf) # number of total samples 

    @cached_property
    def columns(self): # Just an alias...
        return self.features

    @cached_property # Worth caching, since this is a list comprehension on up to 50k strings. Annoying. 
    def features(self):
        if self._cols is not None:
            return self._cols 
        else:
            data = linecache.getline(self.datafile, self.skip - 1)
            data = [x.split('|')[0].upper().strip() for x in data.split(self.sep)]
            return data

    @property
    def shape(self):
        return (self.__len__(), len(self.features))
    
class TestAnndatasetMatrix(Dataset):
    def __init__(self,
        matrix: np.ndarray,
    ) -> None:
        super().__init__()

        self.data = matrix
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            step = (1 if idx.step is None else idx.step)
            idxs = range(idx.start, idx.stop, step)
            return [self[i] for i in idxs]

        data = self.data[idx, :]

        if issparse(data):
            data = data.todense()
            data = np.squeeze(np.asarray(data)) # Need to get first row of 1xp matrix, weirdly this is how to do it :shrug:

        return torch.from_numpy(data)
    
    def __len__(self):
        return (
            self.data.shape[0] if issparse(self.data) else len(self.data) # sparse matrices dont have len :shrug:
        )