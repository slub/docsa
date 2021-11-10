"""Type definitions for a sample.

A sample is defined as a tuple of a document and its target subjects.
"""

from typing import NamedTuple

from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectUriList


class Sample(NamedTuple):
    """A Sample combines a single document and its subject annotations as a tuple."""

    document: Document
    subjects: SubjectUriList
