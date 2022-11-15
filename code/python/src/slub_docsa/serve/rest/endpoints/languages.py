"""REST handlers for language detection."""

from slub_docsa.common.document import Document

from slub_docsa.serve.app import rest_service
from slub_docsa.serve.common import LanguagesRestService


def _service() -> LanguagesRestService:
    return rest_service().get_languages_service()


def find():
    """List all supported languages."""
    return _service().find_languages()


def detect(body):
    """Detect the language for a document."""
    # read documents from request body
    documents = [
        Document(
            uri=str(i),
            title=d.get("title"),
            abstract=d.get("abstract"),
            fulltext=d.get("fulltext")
        ) for i, d in enumerate(body)
    ]
    return _service().detect(documents)
