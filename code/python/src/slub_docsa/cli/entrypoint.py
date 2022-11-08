"""Entrypoint for the command line interface.

common arguments:
  --data /path/to/data/
  --resources /path/to/data/resources/
  --cache /path/to/data/cache/

classify qucosa train / classify tsv train
  --dataset <name>
  --model <name>
  --model_dir /path/to/storage

classify qucosa predict_one / classify tsv predict_one
  --dataset <name>
  --model <name>
  --model_dir /path/to/storage
  <stdin> # document as any text
  <stdout> # prediction as sorted list of subjects with scores

classify qucosa predict_many / classify tsv predict_many
  --model_dir /path/to/storage
  --input_dir /path/to/input/documents
  --output_dir /path/to/output/predictions


experiments qucosa classify_many
experiments qucosa classify_one
experiments qucosa cluster_many
experiments qucosa cluster_one

"""

import argparse
import logging
from slub_docsa.cli.classify import classify_subparser
from slub_docsa.cli.experiments import experiments_subparser
from slub_docsa.cli.serve import serve_subparser

logger = logging.getLogger(__name__)


def main():
    """Process command line arguments."""
    root_parser = argparse.ArgumentParser()

    root_parser_subparsers = root_parser.add_subparsers(dest="command")

    classify_subparser(root_parser_subparsers.add_parser("classify", help="train and predict classification models"))
    experiments_subparser(root_parser_subparsers.add_parser("experiments", help="run pre-built experiments"))
    serve_subparser(root_parser_subparsers.add_parser("serve", help="run rest service"))

    args = root_parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        root_parser.print_help()
