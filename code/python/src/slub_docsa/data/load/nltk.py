"""Manages nltk resource loading."""

import logging
import os

import nltk

from slub_docsa.common.paths import RESOURCES_DIR

logger = logging.getLogger(__name__)

NLTK_PATH = os.path.join(RESOURCES_DIR, "nltk/")
nltk.data.path.append(NLTK_PATH)

NLTK_ALREADY_DOWNLOADED = {}


def download_nltk(name: str):
    """Download a nltk resource.

    Parameters
    ----------
    name: str
        The name of the nltk resources that is being downloaded.
    """
    if name not in NLTK_ALREADY_DOWNLOADED:
        logger.info("nltk download of %s", name)
        os.makedirs(NLTK_PATH, exist_ok=True)
        nltk.download(name, download_dir=NLTK_PATH, quiet=True)
        NLTK_ALREADY_DOWNLOADED[name] = True
