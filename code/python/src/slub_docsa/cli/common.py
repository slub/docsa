"""Common methods used to process command line arguments."""

import logging
import sys
import io
import argparse
import select
import os

from slub_docsa.common import paths

logger = logging.getLogger(__name__)

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
    logging.getLogger("elasticsearch").setLevel(logging.INFO if not args.s else logging.WARNING)


def read_uft8_from_stdin():
    """Return utf-8 encoded text that is read from stdin."""
    rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
    if rlist:
        input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        return input_stream.read()
    return None


def add_storage_directory_arguments(parser: argparse.ArgumentParser):
    """Add common arguments for modifying storage directories."""
    parser.add_argument(
        "--data_dir",
        help="""path to the data directory, default is the current working directory or the environment variable
            SLUB_DOCSA_DATA_DIR""",
        default=os.environ.get("SLUB_DOCSA_DATA_DIR", "./")
    )

    parser.add_argument(
        "--resources_dir",
        help="""path to the resoruces directory that is used to store downloaded resources, default is
            <data_dir>/resources or environment variable SLUB_DOCSA_RESOURCES_DIR""",
    )

    parser.add_argument(
        "--cache_dir",
        help="""path to the cache directory that is used for caching various processing steps, default is
            <data_dir>/runtime/cache or environment variable SLUB_DOCSA_CACHE_DIR""",
    )

    parser.add_argument(
        "--figures_dir",
        help="""path to the figures directory, where plots are saved, default is <data_dir>/runtime/figures or
            environment variable SLUB_DOCSA_FIGURES_DIR""",
    )


def setup_storage_directories(args):
    """Modify storage directories based on provided command line arguments."""
    paths.DIRECTORIES["data"] = args.data_dir
    logger.debug("use data directory: %s", paths.get_data_dir())

    if args.resources_dir is not None:
        paths.DIRECTORIES["resources"] = args.resources_dir
    else:
        paths.DIRECTORIES["resources"] = os.environ.get(
            "SLUB_DOCSA_RESOURCES_DIR",
            os.path.join(paths.get_data_dir(), "resources/")
        )
    logger.debug("use resources directory: %s", paths.get_resources_dir())

    if args.cache_dir is not None:
        paths.DIRECTORIES["cache"] = args.cache_dir
    else:
        paths.DIRECTORIES["cache"] = os.environ.get(
            "SLUB_DOCSA_CACHE_DIR",
            os.path.join(paths.get_data_dir(), "runtime/cache/")
        )
    logger.debug("use cache directory: %s", paths.get_cache_dir())

    if args.figures_dir is not None:
        paths.DIRECTORIES["figures"] = args.figures_dir
    else:
        paths.DIRECTORIES["figures"] = os.environ.get(
            "SLUB_DOCSA_FIGURES_DIR",
            os.path.join(paths.get_data_dir(), "runtime/figures/")
        )
    logger.debug("use figures directory: %s", paths.get_figures_dir())
