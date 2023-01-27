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


class DelimitedDataset(Dataset):
    def __init__(
        self,
        filename: str,
        labelname: str,
        class_label: str,
        indices: Collection[int] = None,
        skip: int = 3,
        cast: bool = True,
        sep: bool = ",",
        index_col: str = None,
        columns: List[any] = None,
        **kwargs,  # To handle extraneous inputs
    ):
        """
        Initialization method for DelimitedDataset.

        The filename contains a delimited text file where ROWS are cells and the COLUMNS are the genes measured. The labelname is a delimited text file
        containing the class_label column, and optionally an index_col.

        Since dropping rows we don't want to train on is nontrivial in the case of the csv, since the csv would have to be read into memory and modified (often this is not computationaly feasible),
        we instead just drop the rows in the labelname file. Then the index_col column specifies the actual row numbers of the samples we want. Equivalently, the index_col is the numeric equivalent to
        the labelname index after dropping the unwanted rows.

        For this reason, index_col must be purely numeric, as the i-th entry of labelfile.loc[:, index_col] contains the actual line number of the i-th sample in the data file.

        :param filename: Path to csv data file, where rows are samples and columns are features
        :type filename: str
        :param labelname: Path to label file, where column '# labels' defines classification labels
        :type labelname: str
        :param class_label: Label to train on, must be in labelname file
        :type class_label: str
        :param indices: List of indices to use in dataset. If None, all indices given in labelname are used., defaults to None
        :type indices: Iterable[int], optional
        :param skip: number of lines to skip in datafile csv, since header lines are often unneeded, defaults to 3
        :type skip: int, optional
        :param cast: cast sample to float32, defaults to True
        :type cast: bool, optional
        :param index_col: column in labelfile that contains numerical indices of datafiles, defaults to 'cell'
        :type index_col: str, optional
        :param sep: separator for labelfile and datafile, defaults to ','
        :type sep: str, optional
        """

        self.filename = filename
        self.labelname = labelname  # alias
        self.class_label = class_label
        self.index_col = index_col
        self.skip = skip
        self.cast = cast
        self.indices = indices
        self.sep = sep
        self._cols = columns

        if indices is None:
            self._labeldf = pd.read_csv(labelname, sep=self.sep)
        else:
            self._labeldf = pd.read_csv(labelname, sep=self.sep).loc[indices, :].reset_index(drop=True)

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
        line = linecache.getline(self.filename, data_index + self.skip)

        if self.cast:
            data = torch.from_numpy(np.array(line.split(self.sep), dtype=np.float32)).float()
        else:
            data = np.array(line.split(self.sep))

        label = self._labeldf.loc[idx, self.class_label]

        return data, label

    def __len__(self):
        return len(self._labeldf)  # number of total samples

    def columns(self):  # Just an alias...
        return self.features

    def features(self):
        if self._cols is not None:
            return self._cols
        else:
            data = linecache.getline(self.filename, self.skip - 1)
            data = [x.split("|")[0].upper().strip() for x in data.split(self.sep)]
            return data

    @cached_property
    def labels(self):
        return self._labeldf.loc[:, self.class_label].unique()

    @property
    def shape(self):
        return (self.__len__(), len(self.features))

    def class_weights(self):
        labels = self._labeldf.loc[:, self.class_label].values

        return compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filename={self.filename}, labelname={self.labelname})"

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(filename={self.filename}, "
            f"labelname={self.labelname}, "
            f"skip={self.skip}, "
            f"cast={self.cast}, "
            f"indices={self.indices})"
        )


class AnnDatasetFile(Dataset):
    def __init__(
        self,
        matrix: np.ndarray,
        labelfile: str,
        class_label: str,
        index_col=None,
        subset=None,
        sep=",",
        columns: List[any] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        # If labelfile is passed, then we need an associated column to pull the class_label from
        if labelfile is not None and class_label is None:
            raise ValueError("If labelfile is passed, column to corresponding class must be passed in class_label.")

        if columns is None:
            warnings.warn(
                f"{self.__class__.__name__} initialized without columns. This will error if training with multiple Datasets with potentially different columns."
            )

        self.data = matrix
        self.labelfile = labelfile
        self.class_label = class_label
        self.index_col = index_col
        self.sep = sep
        self.columns = columns

        if subset is not None:
            self.labels = pd.read_csv(labelfile, sep=self.sep, index_col=index_col).loc[subset, class_label].values
        else:
            self.labels = pd.read_csv(labelfile, sep=self.sep, index_col=index_col).loc[:, class_label].values

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            it = list(range(idx.start or 0, idx.stop or len(self), idx.step or 1))
            return [self[i] for i in it]

        data = self.data[idx]

        # If matrix is sparse, then densify it for training
        if issparse(data):
            data = data.todense()

        return (torch.from_numpy(data), self.labels[idx])

    def __len__(self):
        return len(self.data)


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
        # TODO: add support for delimiteddataset
        # elif suffix == '.csv' or suffix == '.tsv':
        #     data = DelimitedDataset(
        #         filename=datafile,
        #         labelname=labelfile,
        #         class_label=class_label,
        #         sep=sep,
        #     )
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
                # because if we preprocess data becomes a matrix, not an anndata object
                matrix=(data[split.index] if preprocess else data.X[split.index, :]),
                labels=split.values,
                split=split.index,
                *args,
                **kwargs,
            )
            for split in [trainsplit, valsplit, testsplit]
        )
    else:
        train = AnnDatasetMatrix(
            matrix=(data if preprocess else data.X),
            labels=current_labels.values,
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
