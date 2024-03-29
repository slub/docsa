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
        "subject_uri": subject_info.subject_uri,
        "labels": subject_info.labels,
        "ancestors": [{
            "subject_uri": ancestor.subject_uri,
            "labels": ancestor.labels,
        } for ancestor in subject_info.ancestors],
        "children": [{
            "subject_uri": child.subject_uri,
            "labels": child.labels,
        } for child in subject_info.children],
    }
