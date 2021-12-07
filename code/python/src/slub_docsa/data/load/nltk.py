"""Manages nltk resource loading."""

# pylint: disable=import-outside-toplevel

import logging
import os

from slub_docsa.common.paths import get_resources_dir

logger = logging.getLogger(__name__)

NLTK_ALREADY_DOWNLOADED = {}


def download_nltk(name: str):
    """Download a nltk resource.

    Parameters
    ----------
    name: str
        The name of the nltk resources that is being downloaded.
    """
    import nltk
    nltk_download_dir = os.path.join(get_resources_dir(), "nltk/")
    if nltk_download_dir not in nltk.data.path:
        nltk.data.path.append(nltk_download_dir)

    if name not in NLTK_ALREADY_DOWNLOADED:
        logger.info("nltk download of %s", name)
        os.makedirs(nltk_download_dir, exist_ok=True)
        nltk.download(name, download_dir=nltk_download_dir, quiet=True)
        NLTK_ALREADY_DOWNLOADED[name] = True
