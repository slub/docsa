"""REST handlers for language detection."""

from flask.wrappers import Response


def find():
    """List all supported languages."""
    return Response("not implemented yet", status=200, mimetype="text/plain")


def detect():
    """Detect the language for a document."""
    return Response("not implemented yet", status=200, mimetype="text/plain")
