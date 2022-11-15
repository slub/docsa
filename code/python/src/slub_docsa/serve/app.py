"""Method that creates the web application."""

import os
import logging

from typing import Dict, Any, cast

from connexion import FlaskApp
from connexion.resolver import RelativeResolver

from slub_docsa.common.paths import ROOT_DIR
from slub_docsa.serve.exceptions import generate_connexion_exception_handler
from slub_docsa.serve.rest.service.models import ModelNotFoundException
from slub_docsa.serve.routes import app as routes
from slub_docsa.serve.common import RestService
from slub_docsa.serve.validate import validate_max_request_body_length


logger = logging.getLogger(__name__)

OPENAPI_DIRECTORY = os.path.join(ROOT_DIR, "code/python/")

CONTEXT: Dict[str, Any] = {
    "rest_service": None
}


def rest_service() -> RestService:
    """Return the current rest service implementation."""
    service = CONTEXT.get("rest_service")
    if service is None:
        raise RuntimeError("rest service not initialized yet, call `create_webapp` before accessing context")
    return cast(RestService, service)


def create_webapp(service: RestService):
    """Create root web application."""
    # remember webapp context
    if CONTEXT.get("rest_service") is not None:
        logger.warning("overwriting rest service means `create_webapp` was called twice!")
    CONTEXT["rest_service"] = service

    app = FlaskApp(__name__, specification_dir=OPENAPI_DIRECTORY, debug=True)
    app.app.before_request(validate_max_request_body_length)
    app.add_api(
        specification="openapi.yaml",
        resolver=RelativeResolver("slub_docsa.serve.rest.endpoints"),
        validate_responses=True
    )

    app.add_error_handler(ModelNotFoundException, generate_connexion_exception_handler("model not found", 404))
    app.app.register_blueprint(routes)  # type: ignore

    return app
