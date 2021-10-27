"""Type definitions for a sample.

A sample is defined as a tuple of a document and its target subjects.

"""

from typing import Iterator, Tuple

from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectUriList

Sample = Tuple[Document, SubjectUriList]
"""A Sample combines a single document and its subject annotations as a tuple."""

SampleIterator = Iterator[Sample]
"""A SampleIterator describes an iterator over a set of samples."""
