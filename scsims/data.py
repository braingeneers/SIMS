import inspect
import linecache
import pathlib
import warnings
from functools import cache, cached_property, partial
from itertools import chain
from typing import *

import anndata as an
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class AnnDatasetMatrix(Dataset):
    def __init__(
        self,
        matrix: np.ndarray,
        labels: List[any],
        split: Collection[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data = matrix
        self.labels = labels
        self.split = split

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
        refgenes: List[str] = None,
        currgenes: List[str] = None,
        transpose: bool = False,
        normalize: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes a CollateLoader for efficient numerical batch-wise transformations

        :param dataset: DelimitedDatasetset to create DataLoader from
        :type dataset: Type[Dataset]
        :param refgenes: Optional, list of columns to take intersection with , defaults to None
        :type refgenes: List[str], optional
        :param currgenes: Optional, list of current dataset columns, defaults to None
        :type currgenes: List[str], optional
        :param transpose: Boolean indicating whether to tranpose the batch data , defaults to False
        :type transpose: bool, optional
        :param normalize: Boolean indicating whether to normalize the batch data, defaults to False
        :type normalize: bool, optional
        :raises ValueError: If refgenes are passed, currgenes also have to be passed otherwise we dont know what to align with
        """

        if refgenes is None and currgenes is not None or refgenes is not None and currgenes is None:
            raise ValueError(
                "If refgenes is passed, currgenes must be passed too." "If currgenes is passed, refgenes must be passed too."
            )

        # Create collate_fn via a partial of the possible collators, depending on if columns intersection is being calculated
        if refgenes is not None:
            collate_fn = partial(
                _collate_with_refgenes,
                refgenes=refgenes,
                currgenes=currgenes,
                transpose=transpose,
                normalize=normalize,
            )
        else:
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


def _collate_with_refgenes(
    sample: List[tuple],
    refgenes: List[str],
    currgenes: List[str],
    transpose: bool,
    normalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate minibatch of samples where we're intersecting the columns between refgenes and currgenes,

    :param sample: List of samples from DelimitedDataset object
    :type sample: List[tuple]
    :param refgenes: List of reference genes
    :type refgenes: List[str]
    :param currgenes: List of current columns from sample
    :type currgenes: List[str]
    :param transpose: boolean, indicates if we should transpose the minibatch (in the case of incorrectly formatted .csv data)
    :type transpose: bool
    :param normalize: boolean, indicates if we should normalize the minibatch
    :type normalize: bool
    :return: Two torch.Tensors containing the data and labels, respectively
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    if len(sample[0]) == 2:
        data = clean_sample(
            sample=torch.stack([x[0] for x in sample]),
            refgenes=refgenes,
            currgenes=currgenes,
        )

        labels = torch.tensor([x[1] for x in sample])

        # Add Gaussian noise if noise function isn't specificed, otherwise use tht
        # Assumes compatability with the data tensor
        return _transform_sample(data, normalize, transpose), labels
    else:  # len == 1
        data = clean_sample(
            sample=torch.stack(sample),
            refgenes=refgenes,
            currgenes=currgenes,
        )

        return _transform_sample(data, normalize, transpose)


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
    """
    Optionally normalize and tranpose a torch.Tensor

    :param data: Input sample
    :type data: torch.Tensor
    :param normalize: To normalize sample or not
    :type normalize: bool
    :param transpose: to transpose sample or not
    :type transpose: bool
    :return: Modified sample
    :rtype: torch.Tensor
    """
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
    """
    Remove uneeded gene columns for given sample.

    :param sample: n samples to clean
    :type sample: torch.Tensor
    :param refgenes: list of reference genes from helper.generate_intersection(), contains the genes we want to keep
    :type refgenes: List[str]
    :param currgenes: list of reference genes for the current sample
    :type currgenes: List[str]
    :return: Sample reordered and intersected with the list of refgenes
    :rtype: torch.Tensor
    """

    indices = np.intersect1d(currgenes, refgenes, return_indices=True)[1]
    
    if sample.ndim == 2:
        sample = sample[:, indices]
    else:
        sample = sample[indices]  # in the case of a 1d array (single row)

    return sample


def generate_split_dataloaders(
    datafile: Union[str, an.AnnData],
    labelfile: Union[str, None],
    class_label: str,
    index_col: str,
    test_prop: float,
    sep: str,
    subset: Collection[Any],
    stratify: bool,
    batch_size: int,
    num_workers: int,
    currgenes: Collection[Any] = None,
    refgenes: Collection[Any] = None,
    preprocess: bool = False,
    split: bool = True,
    deterministic: bool = True,
    *args,
    **kwargs,
) -> Union[Tuple[CollateLoader], Tuple[DataLoader]]:
    """
    Generates a train, val, test CollateLoader

    :param datafile: Path to dataset csv file
    :type datafile: str
    :param labelfile: Path to label csv file
    :type labelfile: str
    :param class_label: Column (label) in labelfile to train on
    :type class_label: str
    :param test_prop: Proportion of dataset to use in val/test, defaults to 0.2
    :type test_prop: float, optional
    :return: train, val, test loaders
    :rtype: Tuple[CollateLoader, CollateLoader, CollateLoader]
    """

    if isinstance(datafile, an.AnnData):
        data = datafile
    else:
        suffix = pathlib.Path(datafile).suffix

        if suffix == ".h5ad":
            data = an.read_h5ad(datafile)
            if preprocess and refgenes is not None:
                # Do the entire minibatch preprocessing on the input data
                data = clean_sample(
                    sample=data.X,
                    refgenes=refgenes,
                    currgenes=currgenes,
                )
    # if we are using a path to a labelfile, read it in, otherwise
    # just grab the dataframe from the .obs attribute of the anndata object
    if labelfile is not None:
        current_labels = pd.read_csv(labelfile, sep=sep, index_col=index_col)
    else:
        current_labels = datafile.obs.copy().set_index(index_col) if index_col is not None else datafile.obs.copy().reset_index()

    if subset is not None:
        current_labels = current_labels.iloc[subset, :]

    current_labels = current_labels.loc[:, class_label]

    if split:
        # Make stratified split on labels
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

        train, val, test = (
            AnnDatasetMatrix(
                matrix=(data[split.index] if preprocess else data.X[split.index, :]),
                labels=split.values,
                split=split.index,
            )
            for split in [trainsplit, valsplit, testsplit]
        )
    else:
        train = AnnDatasetMatrix(
            matrix=(data if preprocess else data.X),
            labels=current_labels.values,
            split=current_labels.index,
            *args,
            **kwargs,
        )

    if split:
        loaders = [
            CollateLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                refgenes=(None if preprocess else refgenes),
                currgenes=(None if preprocess else currgenes),
                *args,
                **kwargs,
            )
            for dataset in [train, val, test]
        ]
    else:
        loaders = [
            CollateLoader(
                dataset=train,
                batch_size=batch_size,
                num_workers=num_workers,
                refgenes=(None if preprocess else refgenes),
                currgenes=(None if preprocess else currgenes),
                *args,
                **kwargs,
            )
        ]

    return loaders


def generate_dataloaders(
    datafiles: Union[List[str], List[an.AnnData]],
    labelfiles: Union[List[str], List[an.AnnData]],
    class_label: str,
    collocate: bool = True,
    index_col: str = None,
    test_prop=0.2,
    sep=",",
    subset=None,
    stratify=True,
    batch_size: int = 4,
    num_workers: int = 0,
    refgenes: Collection = None,
    currgenes: Collection = None,
    split: bool = True,
    *args,
    **kwargs,
) -> Union[Tuple[CollateLoader, List[CollateLoader]], Tuple[SequentialLoader]]:
    """
    Generates DataLoaders for training, either as a combined list from each datafile or a SequentialLoader to allow sequentially sampling between DataLoaders or DataLoader derived classes.

    :param datafiles: List of absolute paths to datafiles
    :type datafiles: List[str]
    :param labelfiles: List of absolute paths to labelfiles
    :type labelfiles: List[str]
    :param collocate: Whether to combine DataLoaders into one SequentialLoader to allow sequential sampling from all continuously, defaults to True
    :type collocate: bool, optional
    :raises ValueError: Errors if num(labelfiles) != num(datafiles)
    :raises ValueError: Errors if user requests to collocate but only one loader is passed -- probably misinformed
    :return: Either lists containing train, val, test or SequentialLoader's for train, val, test
    :rtype: Union[Tuple[List[CollateLoader], List[CollateLoader], List[CollateLoader]], Tuple[SequentialLoader, SequentialLoader, SequentialLoader]]
    """
    if not collocate and len(datafiles) > 1:
        warnings.warn(
            f"collocate={collocate}, so multiple files will return multiple DataLoaders and cannot be trained sequentially with PyTorch-Lightning"
        )

    train, val, test = [], [], []

    # we need to handle labelfiles being None since it can happen when we're using a .obs key for the labels
    if labelfiles is None:
        labelfiles = [None] * len(datafiles)

    for datafile, labelfile in zip(datafiles, labelfiles):
        loaders = generate_split_dataloaders(
            datafile=datafile,
            labelfile=labelfile,
            class_label=class_label,
            index_col=index_col,
            test_prop=test_prop,
            sep=sep,
            subset=subset,
            stratify=stratify,
            batch_size=batch_size,
            num_workers=num_workers,
            currgenes=currgenes,
            refgenes=refgenes,
            split=split,
            *args,
            **kwargs,
        )

        if len(loaders) == 1:  # no split, just 1 dataloader
            train.append(loaders[0])
        else:
            train.append(loaders[0])
            val.append(loaders[1])
            test.append(loaders[2])

    if len(datafiles) == 1:
        train = train[0]
        if split:
            val = val[0]
            test = test[0]

    if collocate and len(datafiles) > 1:
        train = SequentialLoader(train)
        if split:
            # Join these together into sequential loader if requested, shouldn't error if only one training file passed, though
            val, test = (
                SequentialLoader(val),
                SequentialLoader(test),
            )

    return [train, val, test] if split else [train]


def compute_class_weights(
    labelfiles: list[str],
    class_label: str,
    datafiles: list[an.AnnData] = None,
    sep: str = None,
    device: str = None,
) -> torch.Tensor:
    """
    Compute class weights for the entire label set

    :param labelfiles: List of absolute paths to label files
    :type labelfiles: List[str]
    :param class_label: Target label to calculate weight distribution on
    :type class_label: str
    :return: Tensor of class weights for model training
    :rtype: torch.Tensor
    """
    comb = []
    if labelfiles is not None:  # if we use labelfiles they are dataframes, otherwise use the column in the obs
        for file in labelfiles:
            comb.extend(pd.read_csv(file, sep=sep).loc[:, class_label].values)
    else:
        for file in datafiles:
            comb.extend(file.obs.loc[:, class_label].values)

    weights = torch.from_numpy(
        compute_class_weight(
            y=comb,
            classes=np.unique(comb),
            class_weight="balanced",
        )
    ).float()

    return weights.to(device) if device is not None else weights
