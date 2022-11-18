"""Common methods to parse and encode json for REST endpoints."""

from typing import Sequence

from slub_docsa.common.document import Document


def parse_documents_from_request_body(body) -> Sequence[Document]:
    """Read documents from request body."""
    return [
        Document(
            uri=d.get("uri", str(i)),
            title=d.get("title"),
            abstract=d.get("abstract"),
            fulltext=d.get("fulltext")
        ) for i, d in enumerate(body)
    ]
