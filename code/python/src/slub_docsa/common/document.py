"""Base classes describing a document consisting of an URI, title, abstract, etc."""

# pylint: disable=too-few-public-methods, too-many-arguments

from typing import Iterable, Any, Optional, Sequence


class Document:
    """Represents a document and its meta data."""

    uri: str
    """A uri referecing this document."""

    title: str
    """The title of the document."""

    authors: Iterable[str]
    """A list of author names for this document."""

    abstract: Optional[str]
    """An optional abstract text of this document."""

    fulltext: Optional[Any]
    """An optional fulltext of this document."""

    def __init__(self, uri, title, authors=None, abstract=None, fulltext=None):
        """Initialize document."""
        self.uri = uri
        self.title = title
        self.authors = authors if authors is not None else list([])
        self.abstract = abstract
        self.fulltext = fulltext


DocumentList = Sequence[Document]
"""An ordered list of documents"""
