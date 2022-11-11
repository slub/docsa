"""REST handlers for model discovery and classification."""

import os
from typing import Dict, Sequence

from flask.wrappers import Response
from slub_docsa.common.document import Document

from slub_docsa.common.paths import get_serve_dir
from slub_docsa.serve.setup.classify import generate_classification_model_load_and_classify_function
from slub_docsa.serve.setup.classify import get_classification_model_type_map
from slub_docsa.serve.store.models import find_classification_model_directories


CURRENT_LOAD_AND_CLASSIFY = generate_classification_model_load_and_classify_function(
    find_classification_model_directories(os.path.join(get_serve_dir(), "classification_models")),
    get_classification_model_type_map(),
)


def find():
    """List all available models."""
    return Response("not implemented yet", status=200, mimetype="text/plain")


def add():
    """Add a model."""
    return Response("not implemented yet", status=200, mimetype="text/plain")


def get(model_id):
    """Get information about a model."""
    return Response("not implemented yet: " + model_id, status=200, mimetype="text/plain")


def delete():
    """Delete a model."""
    return Response("not implemented yet", status=200, mimetype="text/plain")


def classify(model_id, body: Sequence[Dict[str, str]], limit, threshold):
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
    results = CURRENT_LOAD_AND_CLASSIFY(model_id, documents, limit, threshold)

    # convert results to json response format
    response = [
        [{"score": r.score, "subject_uri": r.subject_uri} for r in result]
        for result in results
    ]
    return response
