"""Type definitions for samples, meaning tuples of documents and their subjects annotations."""

from typing import Iterator, Tuple

from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectUriList

Sample = Tuple[Document, SubjectUriList]
SampleIterator = Iterator[Sample]
