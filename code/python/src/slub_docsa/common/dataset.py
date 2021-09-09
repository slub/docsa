"""Base classes describing a dataset."""

# pylint: disable=too-few-public-methods

from typing import Sequence

from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectTargets


class Dataset:
    """Represents a dataset consisting of documents and their annotated subjects."""

    documents: Sequence[Document]
    subjects: SubjectTargets

    def __init__(self, documents: Sequence[Document], subjects: SubjectTargets):
        """Initialize a dataset."""
        self.documents = documents
        self.subjects = subjects
