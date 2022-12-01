"""REST handlers for schema inforamtion."""

import urllib.parse

from typing import Iterable
from flask.wrappers import Response

from slub_docsa.serve.app import rest_service
from slub_docsa.serve.common import SchemasRestService
from slub_docsa.serve.rest.endpoints.common import encode_subject_info


def _service() -> SchemasRestService:
    return rest_service().get_schemas_service()


def find() -> Iterable[str]:
    """List all available schemas."""
    return _service().find_schemas()


def get():
    """Get information about a schema."""
    return Response("not implemented yet", status=200, mimetype="text/plain")


def subjects_find(schema_id: str, root_only: bool = True):
    """List all available subjects for a schema."""
    return _service().find_subjects(schema_id, root_only)


def subjects_get(schema_id: str, subject_uri: str):
    """Get information about a subject of a schema."""
    decoded_subject_uri = urllib.parse.unquote(subject_uri)
    subject_info = _service().subject_info(schema_id, decoded_subject_uri)
    return encode_subject_info(subject_info)
