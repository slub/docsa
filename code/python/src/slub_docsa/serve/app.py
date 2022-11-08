"""Method that creates the web application."""

import os
import connexion

from slub_docsa.common.paths import ROOT_DIR
from slub_docsa.serve.routes import app as routes

OPENAPI_DIRECTORY = os.path.join(ROOT_DIR, "code/python/")


def create_webapp():
    """Create root web application."""
    app = connexion.FlaskApp(__name__, specification_dir=OPENAPI_DIRECTORY)
    app.add_api("openapi.yaml")
    app.app.register_blueprint(routes)  # type: ignore
    return app
