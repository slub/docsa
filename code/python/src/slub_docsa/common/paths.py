"""Path definitions to various resources files, cache and output directories."""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
"""The root directory of the python package."""

DATA_DIR = os.path.join(ROOT_DIR, "data/")
"""The main data directory storing all runtime data."""

RESOURCES_DIR = os.path.join(DATA_DIR, "resources/")
"""The directory storing resource files downloaded from various thrid-parties, e.g. dbpedia."""

CACHE_DIR = os.path.join(DATA_DIR, "runtime/cache/")
"""The directory used for various cache data, e.g., partially processed documents."""

FIGURES_DIR = os.path.join(DATA_DIR, "runtime/figures/")
"""The output directory for figures and plots."""

ANNIF_DIR = os.path.join(DATA_DIR, "runtime/container/annif/data")
"""The main data directory used by Annif."""

if __name__ == "__main__":
    print(f"root dir: {ROOT_DIR}")
    print(f"data dir: {DATA_DIR}")
    print(f"resources dir: {RESOURCES_DIR}")
    print(f"resources dir: {CACHE_DIR}")
