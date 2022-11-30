"""REST handlers for model discovery and classification."""

import time
import logging
import json

from typing import Dict, Sequence, Optional

from flask.wrappers import Response
from slub_docsa.serve.app import rest_service
from slub_docsa.serve.common import ClassificationModelsRestService
from slub_docsa.serve.rest.endpoints.common import parse_documents_from_request_body


logger = logging.getLogger(__name__)


def _service() -> ClassificationModelsRestService:
    return rest_service().get_classification_models_service()


def find(
    supported_languages: Optional[str] = None,
    schema_id: Optional[str] = None,
    tags: Optional[str] = None,
):
    """List all available models."""
    language_list = None if supported_languages is None else list(map(str.strip, supported_languages.split(",")))
    tag_list = None if tags is None else list(map(str.strip, tags.split(",")))
    return _service().find_models(language_list, schema_id, tag_list)


def add():
    """Add a model."""
    return Response("not implemented", status=500, mimetype="text/plain")


def get(model_id):
    """Get information about a model."""
    model_info = _service().model_info(model_id)
    return {
        "model_id": model_info.model_id,
        "model_type": model_info.model_type,
        "model_version": model_info.model_version,
        "schema_id": model_info.schema_id,
        "creation_date": model_info.creation_date,
        "description": model_info.description,
        "supported_languages": model_info.supported_languages,
        "tags": model_info.tags,
        "slub_docsa_version": model_info.slub_docsa_version
    }


def delete():
    """Delete a model."""
    return Response("not implemented", status=500, mimetype="text/plain")


def classify(model_id, body: Sequence[Dict[str, str]], limit: int = 10, threshold: float = 0.0):
    """Classify a document using a model optimized for performance."""
    started = time.time() * 1000
    documents = parse_documents_from_request_body(body)

    # do classification
    results = _service().classify(model_id, documents, limit, threshold)

    logger.debug("rest service model.classify took %d ms", ((time.time() * 1000) - started))
    return Response(
        json.dumps(results),
        status=200,
        content_type="application/json"
    )


def classify_and_describe(model_id, body: Sequence[Dict[str, str]], limit: int = 10, threshold: float = 0.0):
    """Classify a document using a model and provide detailed results."""
    # read documents from request body
    documents = parse_documents_from_request_body(body)

    # do classification
    results = _service().classify_and_describe(model_id, documents, limit, threshold)

    # convert results to json response format
    response = [
        {
            "document_uri": result.document_uri,
            "predictions": [
                {"score": r.score, "subject_uri": r.subject_uri} for r in result.predictions
            ]
        } for result in results
    ]
    return response


def subjects(model_id):
    """Return subjects supported by a model."""
    return _service().subjects(model_id)
