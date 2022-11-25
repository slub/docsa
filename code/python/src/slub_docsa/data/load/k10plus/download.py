"""Download k10plus data."""

import os
import urllib.request
import re
import logging

from typing import Iterator

from slub_docsa.common.paths import get_resources_dir
from slub_docsa.data.load.common import download_file


K10PLUS_DEFAULT_BASE_URL = "https://swblod.bsz-bw.de/od/"

logger = logging.getLogger(__name__)


def download_k10plus_marc21_files(
    base_url: str = K10PLUS_DEFAULT_BASE_URL,
    target_dir: str = None
):
    """Download k10plus MARC21 xml files.

    Parameters
    ----------
    base_url : str, optional
        the URL corresponding to the online directory where k10plus marc21 xml files are hosted; if None, the URL
        `https://swblod.bsz-bw.de/od/` is used
    target_dir : str, optional
        the directory where files are stored at; if None, the directory `SLUB_RESOURCE_DIR/k10plus/marc21` is used

    Returns
    -------
    None
        after all files have been downloaded (or verified to already exist)
    """
    if target_dir is None:
        target_dir = os.path.join(get_resources_dir(), "k10plus/marc21/")
    os.makedirs(target_dir, exist_ok=True)

    # find links
    with urllib.request.urlopen(base_url) as response:  # nosec
        html = str(response.read())
        filenames = re.findall(r'href="(od-full_bsz-tit_\d+\.xml\.gz)"', html)

    for filename in filenames:
        filepath = os.path.join(target_dir, filename)
        if not os.path.exists(filepath):
            logger.info("download k10plus marc21 xml dump '%s'", filename)
            url = base_url + filename
            download_file(url, filepath)


def k10plus_marc21_xml_files(
    directory: str = None,
) -> Iterator[str]:
    """Return the list of k10plus xml files contained in a directory.

    Parameters
    ----------
    directory : str, optional
        the directory where files are stored at; if None, the directory `SLUB_RESOURCE_DIR/k10plus/marc21` is used

    Yields
    ------
    str
        the list of filepaths to k10plus marc21 xml files
    """
    if directory is None:
        directory = os.path.join(get_resources_dir(), "k10plus/marc21/")
    for entry in os.scandir(directory):
        if not entry.is_file():
            continue

        filepath = entry.path

        if not filepath.endswith(".xml.gz"):
            continue

        yield filepath


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    download_k10plus_marc21_files()
