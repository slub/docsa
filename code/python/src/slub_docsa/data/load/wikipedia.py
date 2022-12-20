"""Download and parse wikipedia Cirrus data."""

import logging
import os
import json
from typing import Iterator, Optional

from slub_docsa.common.paths import get_resources_dir
from slub_docsa.data.load.common import download_file, read_gzip_text_lines

logger = logging.getLogger(__name__)

# see https://dumps.wikimedia.org/other/cirrussearch/
WIKIPEDIA_CIRRUS_DOWNLOAD_PREFIX = "https://dumps.wikimedia.org/other/cirrussearch/20221205/"


def default_wikipedia_cirrus_download_url(language: str):
    """Return the default download url for wikipedia cirrus files."""
    return WIKIPEDIA_CIRRUS_DOWNLOAD_PREFIX + f"{language}wiki-20221205-cirrussearch-content.json.gz"


def default_wikipedia_cirrus_filepath(language: str):
    """Return the default filepath where wikipedia cirrus files are stored."""
    return os.path.join(get_resources_dir(), f"wikipedia/{language}-cirrus.json.gz")


def download_wikipedia_cirrus_file(
    language: str,
    url: Optional[str] = None,
    filepath: Optional[str] = None
):
    """Download a wikipedia cirrus file for the specified language.

    Parameters
    ----------
    language : str
        the language of the requested wikipedia cirrus texts
    url : Optional[str], optional
        the download URL for the wikipedia cirrus file; if None, a default URL is used
    filepath : Optional[str], optional
        the filepath where the wikipedia cirrus file is stored; if None, a default filepath is used

    Returns
    -------
    None
        as soon as the file was downloaded; or immediately if the file already exists
    """
    if filepath is None:
        filepath = default_wikipedia_cirrus_filepath(language)
    if url is None:
        url = default_wikipedia_cirrus_download_url(language)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if not os.path.exists(filepath):
        logger.info("download wikipedia cirrus file")
        download_file(url, filepath)


def load_wikipedia_cirrus_texts(
    language: str,
    url: Optional[str] = None,
    filepath: Optional[str] = None,
    limit: Optional[int] = None
) -> Iterator[str]:
    """Extract text for wikipedia cirrus file.

    Parameters
    ----------
    language : str
        the language of the requested wikipedia cirrus texts
    url : Optional[str], optional
        the download URL for the wikipedia cirrus file; if None, a default URL is used
    filepath : Optional[str], optional
        the filepath where the wikipedia cirrus file is stored; if None, a default filepath is used
    limit : Optional[int], optional
        the maximum number of texts that are loaded, if None, all texts are loaded

    Yields
    ------
    Iterator[str]
        the individual texts from the wikipedia cirrus file
    """
    if filepath is None:
        filepath = default_wikipedia_cirrus_filepath(language)

    download_wikipedia_cirrus_file(language, url, filepath)

    logger.debug("read text from wikipedia cirrus file")
    count = 0
    for line in read_gzip_text_lines(filepath):
        if limit is not None and count >= limit:
            return
        data = json.loads(line)
        if "text" in data:
            count += 1
            yield data["text"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    for text in load_wikipedia_cirrus_texts(language="de", limit=100):
        print(text)
