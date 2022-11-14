"""REST handlers for pdf processing."""

from flask.wrappers import Response


def extract():
    """Extract text form a pdf."""
    return Response("not implemented yet", status=200, mimetype="text/plain")
