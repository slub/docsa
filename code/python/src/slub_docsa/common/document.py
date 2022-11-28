"""Base class describing a document consisting of an URI, title, abstract, etc."""

# pylint: disable=too-few-public-methods, too-many-arguments

from typing import Sequence, Optional


def _prettify_text(text, max_length=100):
    if text is not None:
        return text.replace("\n", " ").replace("\r", "").strip()[:max_length]
    return None


class Document:
    """Represents a document and its meta data."""

    uri: str
    """A uri referencing this document."""

    title: str
    """The title of the document."""

    language: Optional[str]
    """The ISO 639 language code."""

    authors: Optional[Sequence[str]]
    """A list of author names for this document."""

    abstract: Optional[str]
    """An optional abstract text of this document."""

    toc: Optional[str]
    """An optional table of context as string."""

    fulltext: Optional[str]
    """An optional fulltext of this document."""

    def __init__(self, uri, title, language=None, authors=None, abstract=None, toc=None, fulltext=None):
        """Initialize document."""
        self.uri = uri
        self.title = title
        self.language = language
        self.authors = authors
        self.abstract = abstract
        self.toc = toc
        self.fulltext = fulltext

    def __str__(self):
        """Return document as string representation."""
        return f"<Document uri=\"{self.uri}\" title=\"{_prettify_text(self.title, 100)}\" " \
            + f"language=\"{self.language}\" abstract=\"{_prettify_text(self.abstract, 30)}\" " \
            + f"toc=\"{_prettify_text(self.toc, 30)}\" fulltext=\"{_prettify_text(self.fulltext, 30)}\">"
