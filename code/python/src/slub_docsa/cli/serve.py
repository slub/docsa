"""Serve command."""

# pylint: disable=too-many-locals

import argparse
import logging
import os

import waitress

from slub_docsa.cli.common import add_logging_arguments, setup_logging_from_args
from slub_docsa.common.paths import get_serve_dir
from slub_docsa.data.load.subjects.common import default_schema_generators
from slub_docsa.serve.app import create_webapp
from slub_docsa.serve.common import SimpleRestService
from slub_docsa.serve.models.classification.common import get_all_classification_model_types
from slub_docsa.serve.rest.service.languages import LangidLanguagesRestService
from slub_docsa.serve.rest.service.models import AllStoredModelRestService, SingleStoredModelRestService
from slub_docsa.serve.rest.service.schemas import SimpleSchemaRestService

logger = logging.getLogger(__name__)


def serve_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for serve command."""
    parser.set_defaults(func=_run_rest_service)
    add_logging_arguments(parser)
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        default=False,
        help="whether to load all models into memory (default: load only one model at a time)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="whether to start rest service in debug mode"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SLUB_DOCSA_SERVE_HOST", "0.0.0.0"),  # nosec
        help="the host address that is used to start the REST service (default 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("SLUB_DOCSA_SERVE_PORT", 5000)),
        help="the port number that is used to start the REST service (default 5000)"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=4,
        help="""the number of application threads that are used process REST requests
            (only in production mode, default 4)"""
    )


def _run_rest_service(args):
    """Start rest service when serve command is executed via cli."""
    setup_logging_from_args(args)

    serve_directory = args.serve_dir if args.serve_dir is not None else get_serve_dir()
    load_all_models = args.all
    debug_mode = args.debug
    host = args.host
    port = args.port
    threads = args.threads

    models_dir = os.path.join(serve_directory, "classification_models")
    model_types = get_all_classification_model_types()
    schema_generators = default_schema_generators()
    schema_rest_service = SimpleSchemaRestService(schema_generators)

    model_rest_service_cls = AllStoredModelRestService if load_all_models else SingleStoredModelRestService
    model_rest_service = model_rest_service_cls(models_dir, model_types, schema_rest_service, schema_generators)
    lang_rest_service = LangidLanguagesRestService()

    rest_service = SimpleRestService(model_rest_service, schema_rest_service, lang_rest_service)
    app = create_webapp(rest_service, debug=debug_mode)

    if debug_mode:
        app.run(host=host, port=port)
    else:
        logger.info("start waitress production server on host=%s, port=%d with threads=%d", host, port, threads)
        waitress.serve(app, host=host, port=port, threads=threads)
