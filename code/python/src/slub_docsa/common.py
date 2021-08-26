"""Contains common definitions."""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
DATA_DIR = os.path.join(ROOT_DIR, "data/")
RESOURCES_DIR = os.path.join(DATA_DIR, "resources/")
CACHE_DIR = os.path.join(DATA_DIR, "runtime/cache/")
FIGURES_DIR = os.path.join(DATA_DIR, "runtime/figures/")
ANNIF_DIR = os.path.join(DATA_DIR, "runtime/container/annif/data")

if __name__ == "__main__":
    print(f"root dir: {ROOT_DIR}")
    print(f"data dir: {DATA_DIR}")
    print(f"resources dir: {RESOURCES_DIR}")
    print(f"resources dir: {CACHE_DIR}")
