"""REST handlers for model discovery and classification."""

from typing import Dict, Sequence, Optional

from flask.wrappers import Response
from slub_docsa.common.document import Document
from slub_docsa.serve.app import rest_service
from slub_docsa.serve.common import ClassificationModelsRestService


def _service() -> ClassificationModelsRestService:
    return rest_service().get_classification_models_service()


def find(
    languages: Optional[str] = None,
    schema_id: Optional[str] = None,
    find_best: bool = False  # pylint: disable=unused-argument
):
    """List all available models."""
    language_list = None if languages is None else list(map(str.strip, languages.split(",")))
    return _service().find_models(language_list, schema_id)


def add():
    """Add a model."""
    return Response("not implemented", status=500, mimetype="text/plain")


def get(model_id):
    """Get information about a model."""
    model_info = _service().model_info(model_id)
    return {
        "model_id": model_info.model_id,
        "model_type": model_info.model_type,
        "schema_id": model_info.schema_id,
        "creation_date": model_info.creation_date,
        "description": model_info.description,
        "supported_languages": model_info.supported_languages,
        "tags": model_info.tags,
        "statistics": {},
    }


def delete():
    """Delete a model."""
    return Response("not implemented", status=500, mimetype="text/plain")


def classify(model_id, body: Sequence[Dict[str, str]], limit: int = 10, threshold: float = 0.0):
    """Classify a document using a model."""
    # read documents from request body
    documents = [
        Document(
            uri=str(i),
            title=d.get("title"),
            abstract=d.get("abstract"),
            fulltext=d.get("fulltext")
        ) for i, d in enumerate(body)
    ]

    # do classification
    results = _service().classify(model_id, documents, limit, threshold)

    # convert results to json response format
    response = [
        [{"score": r.score, "subject_uri": r.subject_uri} for r in result]
        for result in results
    ]
    return response
