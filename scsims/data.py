import inspect
import linecache
import pathlib
import warnings
from functools import cache, cached_property, partial
from itertools import chain
from typing import *

import anndata as an
import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class TransformSequence:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

class RandomNoise:
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std + self.mean

class RandomDropout:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, x):
        return torch.nn.functional.dropout(x, p=self.p, training=True)

class AnnDatasetMatrix(Dataset):
    def __init__(
        self,
        matrix: np.ndarray,
        labels: List[any],
        split: Collection[int] = None,
        transform: TransformSequence = None, 
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data = matrix
        self.labels = labels
        self.split = split
        self.transform = transform

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            step = 1 if idx.step is None else idx.step
            idxs = range(idx.start, idx.stop, step)
            return [self[i] for i in idxs]

        data = self.data[idx, :]

        if issparse(data):
            data = data.todense()
            # Need to get first row of 1xp matrix, weirdly this is how to do it :shrug:
            data = np.squeeze(np.asarray(data))

        if self.transform is not None:
            data = self.transform(data)

        return (torch.from_numpy(data), self.labels[idx])

    def __len__(self):
        return self.data.shape[0] if issparse(self.data) else len(self.data)

    @property
    def shape(self):
        return self.data.shape


class CollateLoader(DataLoader):
    def __init__(
        self,
        dataset: Type[Dataset],
        batch_size: int,
        num_workers: int,
        transpose: bool = False,
        normalize: bool = True,
        *args,
        **kwargs,
    ) -> None:
        collate_fn = partial(
            _standard_collate,
            normalize=normalize,
            transpose=transpose,
        )

        # This is awkward, but Dataloader init doesn't handle optional keyword arguments
        # So we have to take the intersection between the passed **kwargs and the DataLoader named arguments
        allowed_args = inspect.signature(super().__init__).parameters
        new_kwargs = {}

        for key in allowed_args:
            name = allowed_args[key].name
            if name in kwargs:
                new_kwargs[key] = kwargs[key]

        # Finally, initialize the DataLoader
        super().__init__(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            **new_kwargs,
        )


class SequentialLoader:
    """
    Class to sequentially stream samples from an arbitrary number of DataLoaders.

    :param dataloaders: List of DataLoaders or DataLoader derived class, such as the CollateLoader from above
    :type dataloaders: List[Union[DataLoader, SequentialLoader]]
    """

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    def __iter__(self):
        yield from chain(*self.dataloaders)

def _standard_collate(
    sample: List[tuple],
    normalize: bool,
    transpose: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate minibatch of samples, optionally normalizing and transposing.

    :param sample: List of DelimitedDataset items to collate
    :type sample: List[tuple]
    :param normalize: boolean, indicates if we should transpose the minibatch (in the case of incorrectly formatted .csv data)
    :type normalize: bool
    :param transpose: boolean, indicates if we should normalize the minibatch
    :type transpose: bool
    :return: Collated samples and labels, respectively
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    if len(sample[0]) == 2:
        data = torch.stack([x[0] for x in sample])
        labels = torch.tensor([x[1] for x in sample])

        return _transform_sample(data, normalize, transpose), labels
    else:  # len == 1
        return _transform_sample(torch.stack(sample), normalize, transpose)


def _transform_sample(data: torch.Tensor, normalize: bool, transpose: bool) -> torch.Tensor:
    if transpose:
        data = data.T
    if normalize:
        data = torch.nn.functional.normalize(data)

    return data

def clean_sample(
    sample: torch.Tensor,
    refgenes: List[str],
    currgenes: List[str],
) -> torch.Tensor:

    indices = np.intersect1d(currgenes, refgenes, return_indices=True)[1]

    if sample.ndim == 2:
        sample = sample[:, indices]
    else:
        sample = sample[indices]  # in the case of a 1d array (single row)

    return sample


def generate_dataloaders(
    data: an.AnnData,
    class_label: str,
    test_prop=0.2,
    stratify=True,
    batch_size: int = 16,
    num_workers: int = 0,
    split: bool = True,
    deterministic: bool = True,
    *args,
    **kwargs,
) -> Union[Tuple[CollateLoader, List[CollateLoader]], Tuple[SequentialLoader]]:
    if isinstance(data, str):
        data = an.read_h5ad(data)

    current_labels = data.obs.loc[:, class_label]

    # make sure data can be stratified
    if stratify:
        if len(current_labels.unique()) < 3:
            warnings.warn(
                "One class has less than 3 samples, disabling stratification"
            )
            stratify = False

    datasets = []
    if split:
        trainsplit, valsplit = train_test_split(
            current_labels,
            stratify=(current_labels if stratify else None),
            test_size=test_prop,
            random_state=(42 if deterministic else None),
        )

        trainsplit, testsplit = train_test_split(
            trainsplit,
            stratify=(trainsplit if stratify else None),
            test_size=test_prop,
            random_state=(42 if deterministic else None),
        )

        for split in [trainsplit, valsplit, testsplit]:
            datasets.append(
                AnnDatasetMatrix(
                    matrix=data[split.index, :].X,
                    labels=split.values,
                    split=split.index,
                )
            )
    else:
        train = AnnDatasetMatrix(
            matrix=data.X,
            labels=current_labels.values,
            split=current_labels.index,
            *args,
            **kwargs,
        )
        datasets.append(train)

    loaders = [
        CollateLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            *args,
            **kwargs,
        )
        for dataset in datasets
    ]

    return loaders

