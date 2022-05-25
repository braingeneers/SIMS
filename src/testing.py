import linecache
from typing import *
from functools import cached_property
import pathlib 

import pandas as pd 
import torch
import numpy as np
import torch
import scanpy as sc 
import anndata as an

from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse
import pytorch_lightning as pl 

from .data import *

class TestDelimitedData(Dataset):
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
        indices: Collection[int]=None,
    ) -> None:
        super().__init__()
        self.data = (matrix if indices is None else matrix[indices, :])
        
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

def generate_single_test_loader(
    datafile: str,
    sep: str=',',
    *args,
    **kwargs,
):
    dataset = generate_single_test_dataset(
        datafile,
        sep,
        *args,
        **kwargs,
    )

    return CollateLoader(
        dataset=dataset,
        *args,
        **kwargs
    )

def generate_test_loaders(
    datafiles: List[str],
    labelfiles: List[str],
    class_label: str,
    index_col: str,
    sep: str,
    *args,
    **kwargs
) -> SequentialLoader:
    
    loaders = []
    for datafile, labelfile in zip(datafiles, labelfiles):
        dataset = generate_single_test_dataset(
            datafile,
            labelfile,
            class_label,
            index_col,
            sep,
            *args,
            **kwargs,
        )

        loaders.append(
            CollateLoader(
                dataset=dataset,
                *args,
                **kwargs,
            )
        )

    return SequentialLoader(loaders)

def generate_single_test_dataset(
    datafile: str,
    labelfile: str,
    class_label: str,
    index_col: str,
    sep: str=',',
    *args,
    **kwargs,
) -> Union[TestDelimitedData, TestAnndatasetMatrix]:
    """
    Generate a single dataset without any splitting, if we want to run prediction at inference time 

    :param datafiles: List of absolute paths to csv files under data_path/ that define cell x expression matrices
    :type datafiles: List[str]
    :param labelfiles: ist of absolute paths to csv files under data_path/ that define cell x class matrices
    :type labelfiles: List[str]
    :param class_label: Column in label files to train on. Must exist in all datasets, this should throw a natural error if it does not. 
    :type class_label: str
    :raises ValueError: Errors if user requests to combine datasets but there is only one. This is probability misinformed and should raise an error.
    :return: Training, validation and test datasets, respectively
    :rtype: Tuple[GeneExpressionData, AnnDatasetFile, AnnDatasetMatrix]
    """

    suffix = pathlib.Path(datafile).suffix 

    if suffix == '.h5ad':
        data = sc.read_h5ad(datafile)
        dataset = TestAnndatasetMatrix(
            matrix=data.X,
            *args,
            **kwargs,
        )
    else:
        if suffix != '.csv' and suffix != '.tsv':
            print(f'Extension {suffix} not recognized, interpreting as .csv. To silence this warning, pass in explicit file types.')

        dataset = TestDelimitedData(
                filename=datafile,
                sep=sep,
                *args,
                **kwargs,
            )

    return dataset
