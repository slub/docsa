"""Serve command."""

import argparse
import logging
import os

from slub_docsa.cli.common import add_logging_arguments, setup_logging_from_args
from slub_docsa.common.paths import get_serve_dir
from slub_docsa.serve.app import create_webapp
from slub_docsa.serve.common import SimpleRestService
from slub_docsa.serve.rest.service.models import AllStoredModelRestService, SingleStoredModelRestService

logger = logging.getLogger(__name__)


def serve_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for serve command."""
    parser.set_defaults(func=_run_rest_service)
    add_logging_arguments(parser)
    parser.add_argument(
        "--directory", "-d",
        help="""directory to load models from, default is <data_dir>/runtime/serve/classification_models or
            environment variable SLUB_DOCSA_SERVE_DIR/classification_models"""
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        default=False,
        help="whether to load all models into memory (default: load only one model at a time)"
    )


def _run_rest_service(args):
    """Start rest service when serve command is executed via cli."""
    setup_logging_from_args(args)

    default_directory = os.path.join(get_serve_dir(), "classification_models")
    serve_directory = args.directory if args.directory is not None else default_directory
    load_all_models = args.all

    model_rest_service = AllStoredModelRestService(serve_directory) \
        if load_all_models else SingleStoredModelRestService(serve_directory)

    rest_service = SimpleRestService(model_rest_service, None, None)
    app = create_webapp(rest_service)
    app.run(host="0.0.0.0", port=5000)  # nosec
