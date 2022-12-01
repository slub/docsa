"""Common methods to parse and encode json for REST endpoints."""

from typing import Any, Mapping, Sequence, Optional

from slub_docsa.common.document import Document
from slub_docsa.serve.common import PublishedSubjectInfo


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


def encode_subject_info(subject_info: Optional[PublishedSubjectInfo]) -> Optional[Mapping[str, Any]]:
    """Encode subject information as a dictionary object to be transformed to json."""
    if subject_info is None:
        return None
    return {
        "labels": subject_info.labels,
        "breadcrumb": subject_info.breadcrumb,
        "parent_subject_uri": subject_info.parent_subject_uri,
        "children_subject_uris": subject_info.children_subject_uris,
    }
