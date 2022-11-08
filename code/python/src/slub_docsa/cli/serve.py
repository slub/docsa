"""Serve command."""

import argparse
import logging

from slub_docsa.cli.common import add_logging_arguments, setup_logging_from_args
from slub_docsa.serve.app import create_webapp

logger = logging.getLogger(__name__)


def serve_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for serve command."""
    parser.set_defaults(func=_run_rest_service)
    add_logging_arguments(parser)


def _run_rest_service(args):
    """Start rest service when serve command is executed via cli."""
    setup_logging_from_args(args)

    app = create_webapp()
    app.run(host="0.0.0.0", port=5000)  # nosec
