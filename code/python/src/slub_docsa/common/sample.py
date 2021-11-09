"""Type definitions for a sample.

A sample is defined as a tuple of a document and its target subjects.
"""

# pylint: disable=too-few-public-methods

from typing import Iterator, Tuple

from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectUriList


class Sample(Tuple[Document, SubjectUriList]):
    """A Sample combines a single document and its subject annotations as a tuple."""


class SampleIterator(Iterator[Sample]):
    """A SampleIterator describes an iterator over a set of samples."""
