"""Custom validation methods."""

import logging

from flask import request, Response


logger = logging.getLogger(__name__)

CONFIG = {
    "MAX_CONTENT_LENGTH": 16 * 1024 * 1024
}


def validate_max_request_body_length():
    """Check that request body does not exceed certain maximum size."""
    if request.content_length is not None and request.content_length > CONFIG["MAX_CONTENT_LENGTH"]:
        return Response(status=413)
    return None
