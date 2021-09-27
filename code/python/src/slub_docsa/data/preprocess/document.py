"""Methods that process documents."""

from slub_docsa.common.document import Document


def document_as_concatenated_string(doc: Document) -> str:
    """Convert a document to a string by simple concatenation."""
    text = doc.title
    if doc.abstract is not None:
        text += " " + doc.abstract
    return text
