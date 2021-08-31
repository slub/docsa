"""Methods for splitting data into training and test sets."""

from typing import Sequence, Iterator, TypeVar, Iterable, Tuple
import numpy as np

from sklearn.model_selection import KFold

from slub_docsa.common.dataset import Dataset

SequenceType = TypeVar("SequenceType")


class IndexedSequence(Sequence[SequenceType]):
    """Provides indexed access to a sequence."""

    def __init__(self, sequence: Sequence[SequenceType], idx: Sequence[int]):
        """Initialize indexed sequence."""
        if len(sequence) < len(idx):
            raise ValueError("index can not be larger than sequence")
        self.sequence = sequence
        self.idx = idx

    def __contains__(self, x: object) -> bool:
        """Return true of indexed sequence contains object."""
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


def cross_validation_split(n_splits: int, dataset: Dataset, random_state=0) -> Iterable[Tuple[Dataset, Dataset]]:
    """Split dataset into `n_splits` many training and test datasets."""
    virtual_features = np.zeros((len(dataset.documents), 1))
    virtual_targets = np.random.randint(n_splits, size=(len(dataset.documents), 1))

    folder = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    for train_idx_list, test_idx_list in folder.split(virtual_features, virtual_targets):

        train_dataset = Dataset(
            documents=IndexedSequence(dataset.documents, train_idx_list),
            subjects=IndexedSequence(dataset.subjects, train_idx_list),
        )

        test_dataset = Dataset(
            documents=IndexedSequence(dataset.documents, test_idx_list),
            subjects=IndexedSequence(dataset.subjects, test_idx_list),
        )

        yield train_dataset, test_dataset
