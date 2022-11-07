"""Methods for splitting data into training and test sets."""

import logging

from typing import Callable, Sequence, Iterator, TypeVar, Iterable, Tuple
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold
from skmultilearn.model_selection import IterativeStratification

from slub_docsa.common.dataset import SimpleDataset, Dataset
from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets, unique_subject_order

logger = logging.getLogger(__name__)

ST = TypeVar("ST")
"""An arbitrary type used by `IndexedSequence`."""

DatasetSplitFunction = Callable[[Dataset], Iterable[Tuple[Dataset, Dataset]]]
"""A splitting function that, given a dataset, returns an iterator over multiple training and test subsets."""


class IndexedSequence(Sequence[ST]):
    """Provides indexed access to an arbitrary sequence.

    Is used to provide transparent access to training and test sets without copying any data.
    """

    def __init__(self, sequence: Sequence[ST], idx: Sequence[int]):
        """Initialize indexed sequence.

        Parameters
        ----------
        sequence: Sequence[ST]
            the sequence that is supossed to be access via an index
        idx: Sequence[int]
            the index at which sequence is being accessed
        """
        if len(sequence) < len(idx):
            raise ValueError("index can not be larger than sequence")
        self.sequence = sequence
        self.idx = idx

    def __contains__(self, item: object) -> bool:
        """Return true if indexed sequence contains object."""
        for i in self.idx:
            if self.sequence[i] == item:
                return True
        return False

    def __iter__(self) -> Iterator[ST]:
        """Iterate over indexed sequence."""
        for i in self.idx:
            yield self.sequence[i]

    def __len__(self) -> int:
        """Return length of index."""
        return self.idx.__len__()

    def __getitem__(self, i) -> ST:
        """Return item at position of index."""
        return self.sequence[self.idx[i]]


def scikit_base_folder_splitter(
    n_splits: int,
    folder: _BaseKFold,
    use_random_targets: bool = False,
) -> DatasetSplitFunction:
    """Return a function that can be used for splitting data in training and test sets using a scikit `folder`.

    The scikit `folder` has to be an implementation of the `_BaseKFold` class of scikit-learn.

    Parameters
    ----------
    n_splits: int
        the number of cross-validation splits
    folder: _BaseKFold
        the scikit-learn implementation of `_BaseKFold`
    use_random_targets: bool = False
        whether to provide the true subject targets or random targets to the folder

    Returns
    -------
    DatasetSplitFunction
        a function that splits a dataset (only argument) into multiple training and test subsets according to the
        splitting strategy implemented by the provided `folder`
    """

    def split_function(dataset: Dataset) -> Iterable[Tuple[Dataset, Dataset]]:
        # features should not have any effect on splitting, use artificial features
        features = np.zeros((len(dataset.documents), 1))
        # if targets are not provided, use random targets
        if use_random_targets:
            targets = np.random.randint(n_splits, size=(len(dataset.documents), 1))
        else:
            targets = subject_incidence_matrix_from_targets(dataset.subjects, unique_subject_order(dataset.subjects))

        for train_idx_list, test_idx_list in folder.split(features, targets):

            train_dataset = SimpleDataset(
                documents=IndexedSequence(dataset.documents, train_idx_list),
                subjects=IndexedSequence(dataset.subjects, train_idx_list),
            )

            test_dataset = SimpleDataset(
                documents=IndexedSequence(dataset.documents, test_idx_list),
                subjects=IndexedSequence(dataset.subjects, test_idx_list),
            )

            yield train_dataset, test_dataset

    return split_function


def scikit_kfold_splitter(
    n_splits: int,
    random_state: float = None,
) -> DatasetSplitFunction:
    """Return a function that splits a dataset randomly into `n_splits` using scikit's KFold class.

    Parameters
    ----------
    n_splits: int
        the number of cross-validation splits
    random_state: float = None
        the random state that is passed along to the KFold class

    Returns
    -------
    DatasetSplitFunction
        a function that given a dataset returns an iterator over tuples of training and test subsets
    """
    folder = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return scikit_base_folder_splitter(n_splits, folder, use_random_targets=True)


def skmultilearn_iterative_stratification_splitter(
    n_splits: int,
    random_state: float = None,
) -> DatasetSplitFunction:
    """Return a function that splits a dataset randomly into `n_splits` using skmultilearn's IterativeStratification.

    .. note::
        The random state does not seem to have any effect on the folder. The resulting splits are not deterministic
        given a specific random state and the same dataset.

    Parameters
    ----------
    n_splits: int
        the number of cross-validation splits
    random_state: float = None
        the random state that is passed along to the skmultilearn's IterativeStratification class (but does not seem to
        have any effect on it)

    Returns
    -------
    DatasetSplitFunction
        a function that given a dataset returns an iterator over tuples of training and test subsets
    """
    folder = IterativeStratification(n_splits=n_splits, order=1, random_state=random_state)
    return scikit_base_folder_splitter(n_splits, folder, use_random_targets=False)


def scikit_kfold_train_test_split(ratio: float, dataset: Dataset, random_state=None) -> Tuple[Dataset, Dataset]:
    """Return a single training and test split with a rough ratio of samples."""
    n_splits = round(1 / (1 - ratio))
    return next(iter(scikit_kfold_splitter(n_splits, random_state=random_state)(dataset)))
