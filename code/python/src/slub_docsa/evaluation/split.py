"""Methods for splitting data into training and test sets."""

import logging

from typing import Callable, Sequence, Iterator, TypeVar, Iterable, Tuple
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold
from skmultilearn.model_selection import IterativeStratification

from slub_docsa.common.dataset import Dataset
from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets, unique_subject_order

logger = logging.getLogger(__name__)

SequenceType = TypeVar("SequenceType")
DatasetSplitFunction = Callable[[Dataset], Iterable[Tuple[Dataset, Dataset]]]


class IndexedSequence(Sequence[SequenceType]):
    """Provides indexed access to a sequence."""

    def __init__(self, sequence: Sequence[SequenceType], idx: Sequence[int]):
        """Initialize indexed sequence."""
        if len(sequence) < len(idx):
            raise ValueError("index can not be larger than sequence")
        self.sequence = sequence
        self.idx = idx

    def __contains__(self, x: object) -> bool:
        """Return true if indexed sequence contains object."""
        for i in self.idx:
            if self.sequence[i] == x:
                return True
        return False

    def __iter__(self) -> Iterator[SequenceType]:
        """Iterate over indexed sequence."""
        for i in self.idx:
            yield self.sequence[i]

    def __len__(self) -> int:
        """Return length of index."""
        return self.idx.__len__()

    def __getitem__(self, i) -> SequenceType:
        """Return item at position of index."""
        return self.sequence[self.idx[i]]


def scikit_base_folder_splitter(
    n_splits: int,
    folder: _BaseKFold,
    use_random_targets: bool = False,
) -> DatasetSplitFunction:
    """Apply a scikit `folder` (which implements BaseKFold) to a dataset."""

    def split_function(dataset: Dataset) -> Iterable[Tuple[Dataset, Dataset]]:
        # features should not have any effect on splitting, use artificial features
        features = np.zeros((len(dataset.documents), 1))
        # if targets are not provided, use random targets
        if use_random_targets:
            targets = np.random.randint(n_splits, size=(len(dataset.documents), 1))
        else:
            targets = subject_incidence_matrix_from_targets(dataset.subjects, unique_subject_order(dataset.subjects))

        for train_idx_list, test_idx_list in folder.split(features, targets):

            train_dataset = Dataset(
                documents=IndexedSequence(dataset.documents, train_idx_list),
                subjects=IndexedSequence(dataset.subjects, train_idx_list),
            )

            test_dataset = Dataset(
                documents=IndexedSequence(dataset.documents, test_idx_list),
                subjects=IndexedSequence(dataset.subjects, test_idx_list),
            )

            yield train_dataset, test_dataset

    return split_function


def scikit_kfold_splitter(
    n_splits: int,
    random_state: float = None,
) -> DatasetSplitFunction:
    """Split dataset randomly into `n_splits` many training and test datasets using scikit's KFold class."""
    folder = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    return scikit_base_folder_splitter(n_splits, folder, use_random_targets=True)


def skmultilearn_iterative_stratification_splitter(
    n_splits: int,
    random_state: float = None,
) -> DatasetSplitFunction:
    """Split dataset random into `n_splits` using skmultilearn's IterativeStratification class."""
    folder = IterativeStratification(n_splits=n_splits, order=1, random_state=random_state)
    return scikit_base_folder_splitter(n_splits, folder, use_random_targets=False)


def scikit_kfold_train_test_split(ratio: float, dataset: Dataset, random_state=None) -> Tuple[Dataset, Dataset]:
    """Return a single training and test split with a rough ratio of samples."""
    n_splits = round(1 / (1 - ratio))
    return next(iter(scikit_kfold_splitter(n_splits, random_state=random_state)(dataset)))
