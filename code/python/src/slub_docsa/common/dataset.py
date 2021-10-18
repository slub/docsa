"""Base classes describing a dataset."""

# pylint: disable=too-few-public-methods

from typing import Sequence, cast

from slub_docsa.common.document import Document
from slub_docsa.common.sample import SampleIterator
from slub_docsa.common.subject import SubjectTargets


class Dataset:
    """Represents a dataset consisting of documents and their annotated subjects."""

    documents: Sequence[Document]
    subjects: SubjectTargets


class SimpleDataset(Dataset):
    """Stores documents and subjects as simple lists."""

    def __init__(self, documents: Sequence[Document], subjects: SubjectTargets):
        """Initialize a dataset."""
        self.documents = documents
        self.subjects = subjects


def dataset_from_samples(samples: SampleIterator):
    """Return dataset from an iterator over samples, which are tuples of documents and their annotated subjects."""
    zipped = list(zip(*samples))
    return SimpleDataset(documents=cast(Sequence[Document], zipped[0]), subjects=cast(SubjectTargets, zipped[1]))
