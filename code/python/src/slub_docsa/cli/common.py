"""Common methods used to process command line arguments."""

import logging
import sys
import io
import argparse

LOG_LEVEL_MAP = {
    0: logging.WARN,
    1: logging.INFO,
    2: logging.DEBUG,
}


def add_logging_arguments(parser: argparse.ArgumentParser):
    """Add default arguments controlling log verbosity to a argument parser."""
    parser.add_argument("-v", action="count", default=1, help="increase logging verbosity")
    parser.add_argument("-s", action="count", default=0, help="only print error messages")


def setup_logging_from_args(args):
    """Set up logging depending on the arguments controlling verbosity."""
    # setup logging based on verbose level and silent option
    log_arg_level = 0 if args.s else args.v
    log_level = LOG_LEVEL_MAP[log_arg_level] if log_arg_level in LOG_LEVEL_MAP else logging.DEBUG
    logging.basicConfig(level=log_level)


def read_uft8_from_stdin():
    """Return utf-8 encoded text that is read from stdin."""
    input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    return input_stream.read()
