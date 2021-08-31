"""Base classes describing a dataset."""

# pylint: disable=too-few-public-methods

from typing import Sequence, Iterable

from slub_docsa.common.document import Document


class Dataset:
    """Represents a dataset consisting of documents and their annotated subjects."""

    documents: Sequence[Document]
    subjects: Sequence[Iterable[str]]

    def __init__(self, documents: Sequence[Document], subjects: Sequence[Iterable[str]]):
        """Initialize a dataset."""
        self.documents = documents
        self.subjects = subjects
