"""Manages nltk resource loading."""

import logging
import os

import nltk

from slub_docsa.common import RESOURCES_DIR

logger = logging.getLogger(__name__)

NLTK_PATH = os.path.join(RESOURCES_DIR, "nltk/")
nltk.data.path.append(NLTK_PATH)


def download_nltk(name):
    """Download a nltk resource."""
    logger.info("nltk download of %s", name)
    nltk.download(name, download_dir=NLTK_PATH)
