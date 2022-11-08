"""REST handlers for schema inforamtion."""

from flask.wrappers import Response


def find():
    """List all available schemas."""
    return Response("not implemented yet", status=200, mimetype="text/plain")


def get():
    """Get information about a schema."""
    return Response("not implemented yet", status=200, mimetype="text/plain")


def subjects_find():
    """List all available subjects for a schema."""
    return Response("not implemented yet", status=200, mimetype="text/plain")


def subjects_get():
    """Get information about a subject of a schema."""
    return Response("not implemented yet", status=200, mimetype="text/plain")


def subjects_children():
    """List all children of a subject."""
    return Response("not implemented yet", status=200, mimetype="text/plain")
