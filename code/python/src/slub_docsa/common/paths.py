"""Default paths to various resources files, cache and output directories.

The following directories serve as default paths that are used throughout this library. In most cases, these paths
can be specified directly when calling a particular method. However, for convenience, default paths are used based
on the following directory structure.

This default directory structure is based on a common data directory, which is then divided further:

- `data/` - default parent directory for all default directories and default paths
- `data/resources/` - static resources that won't change (e.g. downloaded files)
- `data/runtime/cache/` - various volatile files that are generated from static resources (e.g. indices)
- `data/runtime/figures/` - output directory for generated plots (e.g. performance plots)
- `data/runtime/container/annif/data` - data directory of an Annif installation

All paths can be overwritten by specifying environment variables:

- `SLUB_DOCSA_DATA_DIR`
- `SLUB_DOCSA_RESOURCES_DIR`
- `SLUB_DOCSA_CACHE_DIR`
- `SLUB_DOCSA_FIGURES_DIR`
- `SLUB_DOCSA_ANNIF_DIR`

Alternatively, paths can be overwritten directly, e.g.,

>>> from slub_docsa.common import paths
>>> paths.DIRECTORIES["data"] = "/my/custom/data/dir"
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))

DEFAULT_DATA_DIR = os.environ.get("SLUB_DOCSA_DATA_DIR", os.path.join(ROOT_DIR, "data/"))
"""The main data directory storing all runtime data.
It can be customized by specifying the environment variable `SLUB_DOCSA_DATA_DIR`.
"""

DEFAULT_RESOURCES_DIR = os.environ.get("SLUB_DOCSA_RESOURCES_DIR", os.path.join(DEFAULT_DATA_DIR, "resources/"))
"""The directory storing resource files downloaded from various thrid-parties, e.g. dbpedia.
It can be customized by specifying the environment variable `SLUB_DOCSA_RESOURCES_DIR`.
"""

DEFAULT_CACHE_DIR = os.environ.get("SLUB_DOCSA_CACHE_DIR", os.path.join(DEFAULT_DATA_DIR, "runtime/cache/"))
"""The directory used for various cache data, e.g., partially processed documents.
It can be customized by specifying the environment variable `SLUB_DOCSA_CACHE_DIR`.
"""

DEFAULT_FIGURES_DIR = os.environ.get("SLUB_DOCSA_FIGURES_DIR", os.path.join(DEFAULT_DATA_DIR, "runtime/figures/"))
"""The output directory for figures and plots.
It can be customized by specifying the environment variable `SLUB_DOCSA_FIGURES_DIR`.
"""

DEFAULT_ANNIF_DIR = os.environ.get(
    "SLUB_DOCSA_ANNIF_DIR",
    os.path.join(DEFAULT_DATA_DIR, "runtime/container/annif/data")
)
"""The main data directory used by Annif.
It can be customized by specifying the environment variable `SLUB_DOCSA_ANNIF_DIR`.
"""

DIRECTORIES = {
    "data": DEFAULT_DATA_DIR,
    "resources": DEFAULT_RESOURCES_DIR,
    "cache": DEFAULT_CACHE_DIR,
    "figures": DEFAULT_FIGURES_DIR,
    "annif": DEFAULT_ANNIF_DIR,
}


def get_data_dir():
    """Return the main data directory storing all runtime data."""
    return DIRECTORIES["data"]


def get_resources_dir():
    """Return the directory storing resource files downloaded from various thrid-parties, e.g. dbpedia."""
    return DIRECTORIES["resources"]


def get_cache_dir():
    """Return the directory used for various cache data, e.g., partially processed documents."""
    return DIRECTORIES["cache"]


def get_figures_dir():
    """Return the output directory for figures and plots."""
    return DIRECTORIES["figures"]


def get_annif_dir():
    """Return the main data directory used by Annif."""
    return DIRECTORIES["annif"]
