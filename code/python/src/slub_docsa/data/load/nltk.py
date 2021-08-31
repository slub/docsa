"""Manages nltk resource loading."""

import logging
import os

import nltk

from slub_docsa.common.paths import RESOURCES_DIR

logger = logging.getLogger(__name__)

NLTK_PATH = os.path.join(RESOURCES_DIR, "nltk/")
nltk.data.path.append(NLTK_PATH)


def download_nltk(name: str):
    """Download a nltk resource.

    Parameters
    ----------
    name: str
        The name of the nltk resources that is being downloaded.
    """
    logger.info("nltk download of %s", name)
    nltk.download(name, download_dir=NLTK_PATH)
