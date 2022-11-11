"""Base class describing a document consisting of an URI, title, abstract, etc."""

# pylint: disable=too-few-public-methods, too-many-arguments

from typing import Sequence, Optional


class Document:
    """Represents a document and its meta data."""

    uri: str
    """A uri referencing this document."""

    title: str
    """The title of the document."""

    authors: Optional[Sequence[str]]
    """A list of author names for this document."""

    abstract: Optional[str]
    """An optional abstract text of this document."""

    fulltext: Optional[str]
    """An optional fulltext of this document."""

    def __init__(self, uri, title, authors=None, abstract=None, fulltext=None):
        """Initialize document."""
        self.uri = uri
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.fulltext = fulltext

    def __str__(self):
        """Return document as string representation."""
        title = None if self.title is None else self.title[:100]
        abstract = None if self.abstract is None else self.abstract[:100]
        fulltext = None if self.fulltext is None else self.fulltext[:100]
        return f"<Document title=\"{title}\" abstract=\"{abstract}\" fulltext=\"{fulltext}\">"
