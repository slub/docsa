"""REST handlers for model discovery and classification."""

from flask.wrappers import Response


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


def classify(model_id):
    """Classify a document using a model."""
    return Response("not implemented yet: " + model_id, status=200, mimetype="text/plain")
