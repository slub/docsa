"""Base class describing a dataset consisting of documents and target subjects.

A dataset is simply the combination of documents and their respective target subject annotations (or multi-label
classes). An implementation is required to provide fast random access to both documents and subjects.

In contrast to the notion of a `slub_docsa.common.sample.Sample`, documents and subjects are accessed by two separate
properties `documents` and `subjects`, which follow the same ordering such that `subjects[i]` contains the list of
target subjects for the document stored in `documents[i]`.

This design was chosen in order to be able to access and process the sequence of subject annotations independently
from their corresponding documents, which allows for much better performance in case documents are not needed for
processing, e.g., when evaluating hierarchical relationships between subjects.

Usually, a dataset implementation should interface with some kind of database that allows to randomly access large
amounts of documents and their respective subject annotations without loading all of them into main memory. A simple
database implementation is provided via `slub_docsa.data.store.dataset.DatasetSqliteStore`.
"""

# pylint: disable=too-few-public-methods

from typing import Iterator, Sequence, cast

from slub_docsa.common.document import Document
from slub_docsa.common.sample import Sample
from slub_docsa.common.subject import SubjectTargets


class Dataset:
    """Represents a dataset consisting of documents and their annotated subjects.

    Both documents and their respective annotated target subjects are accessible via a sequence interface.
    Their ordering or indexing has to match.
    """

    documents: Sequence[Document]
    subjects: SubjectTargets


class SimpleDataset(Dataset):
    """Simply keeps track of arbitrary documents and subjects provided during initialization."""

    def __init__(self, documents: Sequence[Document], subjects: SubjectTargets):
        """Remember arbitrary documents and subjects sequences."""
        self.documents = documents
        self.subjects = subjects


def dataset_from_samples(samples: Iterator[Sample]) -> Dataset:
    """Return dataset from an iterator over samples, stored as simple python lists.

    .. note::

        This will fail for large datasets that do not fit in main memory!
    """
    zipped = list(zip(*samples))
    return SimpleDataset(documents=cast(Sequence[Document], zipped[0]), subjects=cast(SubjectTargets, zipped[1]))


def samples_from_dataset(dataset: Dataset) -> Iterator[Sample]:
    """Return an iterator over each sample of a dataset."""
    for i, document in enumerate(dataset.documents):
        yield Sample(document, dataset.subjects[i])
