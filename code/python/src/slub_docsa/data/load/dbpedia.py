"""Loads Dbpedia resources."""

import logging
import os
import bz2
import re
from typing import Iterator
import urllib.request
import urllib.parse
import shutil

from slub_docsa.common.paths import RESOURCES_DIR

logger = logging.getLogger(__name__)

RESOURCE_DIR = os.path.join(RESOURCES_DIR, "dbpedia")


def _get_dbpedia_download_url(lang_code):
    filename = f"short-abstracts_lang={lang_code}.ttl.bz2"
    return f"https://databus.dbpedia.org/dbpedia/text/short-abstracts/2020.07.01/{filename}"


def _get_dbpedia_abstracts_filepath(lang_code):
    return os.path.join(RESOURCE_DIR, f"short-abstracts_lang={lang_code}.ttl.bz2")


def _download_dbpedia_abstracts(lang_code: str, filepath: str = None):
    # create resources dir if not exists
    os.makedirs(RESOURCE_DIR, exist_ok=True)

    if filepath is None:
        filepath = _get_dbpedia_abstracts_filepath(lang_code)

    if not os.path.exists(filepath):
        logging.info("downloading dbpedia abstracts, this may take a while ... ")
        url = _get_dbpedia_download_url(lang_code)
        with urllib.request.urlopen(url) as request, open(filepath, 'wb') as file:  # nosec
            shutil.copyfileobj(request, file)


def read_dbpedia_abstracts(lang_code: str, limit=None, filepath: str = None) -> Iterator[str]:
    """Return an iteator of dbpedia abstracts.

    Parameters
    ----------
    lang_code: str
        The language code of DBpedia resources to load.
    limit: int | None
        The maximum number of abstracts to return. Returns all abstracts if None.
    filepath: str | None
        The path to the dbpedia abstracts resource file. Is set to a default path relative to the
        `slub_docsa.common.paths.RESOURCES_DIR` if set to None.
        File will be downloaded unless it already exists.

    Returns
    -------
    Iterator[str]
        Each abstract as string.
    """
    _download_dbpedia_abstracts(lang_code, filepath)

    line_pattern_str = r"^<([^>]+)> <http://www.w3.org/2000/01/rdf-schema#comment> \"(.*)\"@" + lang_code + r" .$"
    line_pattern = re.compile(line_pattern_str)

    n_abstracts = 0
    with bz2.open(_get_dbpedia_abstracts_filepath(lang_code), "rt", encoding="utf-8") as file:
        while True:
            line = file.readline()

            if not line:
                break

            if limit is not None and n_abstracts > limit:
                break

            line_match = line_pattern.match(line)
            if line_match:
                abstract = line_match.group(2).replace("\\", "")
                n_abstracts += 1
                yield abstract
