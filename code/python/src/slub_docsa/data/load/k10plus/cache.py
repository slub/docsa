"""Cache k10plus data as json files.

This package allows to iterate through the full k10plus marc21 xml dumps and generate json documents with relevant
information (title, language, subjects). The xml processing is parallized to achieve high performance.
"""

from itertools import islice
import os
import logging
import multiprocessing
import gzip
import json

from typing import Optional

from slub_docsa.common.paths import get_cache_dir, get_resources_dir
from slub_docsa.data.load.common import read_gzip_text_lines
from slub_docsa.data.load.languages import load_language_codes
from slub_docsa.data.preprocess.language import load_fasttext_language_detector
from slub_docsa.data.load.k10plus.download import download_k10plus_marc21_files, k10plus_marc21_xml_files
from slub_docsa.data.load.k10plus.marc21 import parse_single_k10plus_marc21_xml_file, parse_k10plus_marc_xml_to_json

logger = logging.getLogger(__name__)


def k10plus_json_cache_filepath_from_xml_filepath(
    xml_filepath: str,
    json_directory: str,
) -> str:
    """Return path to json cache file for a k10plus marc21 xml dump file.

    Parameters
    ----------
    xml_filepath : str
        the filepath to an k10plus marc21 xml dump file
    json_directory : str
        the directory where json cache files are supposed to be stored

    Returns
    -------
    str
        the filepath to the equivalent json cache file
    """
    return os.path.join(json_directory, os.path.splitext(os.path.basename(xml_filepath))[0] + ".json.gz")


def k10plus_transform_and_save_xml_to_json(
    xml_filepath: str,
    json_filepath: str,
    line_batch_size: int = 1000,
):
    """Read a single k10plus xml dump file and saves a single corresponding json cache file after transformation.

    Parameters
    ----------
    xml_filepath : str
        the filepath to the k10plus marc21 xml dump file
    json_filepath : str
        the filepath to the corresponding json cache file that is generated
    line_batch_size : int, optional
        the number of lines that are read in one batch to improve performance, by default 1000
    """
    if os.path.exists(json_filepath):
        logger.info("k10plus json cache file already exists at, skipping: %s", json_filepath)
        return
    os.makedirs(os.path.dirname(json_filepath), exist_ok=True)

    logger.info("save k10plus xml dump as json cache file to '%s'", json_filepath)
    language_detector = load_fasttext_language_detector(0.99)
    language_code_table = load_language_codes()
    with gzip.open(json_filepath + ".tmp", "wt") as file:
        for xml_str in parse_single_k10plus_marc21_xml_file(xml_filepath, line_batch_size):
            json_object = parse_k10plus_marc_xml_to_json(xml_str, language_detector, language_code_table)
            json_str = json.dumps(json_object)
            file.write(json_str + "\n")
    os.rename(json_filepath + ".tmp", json_filepath)


def k10plus_build_json_cache(
    xml_directory: Optional[str] = None,
    json_directory: Optional[str] = None,
    download: bool = True,
    workers: Optional[int] = None,
    line_batch_size: int = 1000,
):
    """Read all k10plus marc21 xml dump files and create corresponding json cache files.

    Parameters
    ----------
    xml_directory : Optional[str]
        the directory where k10plus marc21 xml dump files are stored; if None, the directory
        `SLUB_RESOURCE_DIR/k10plus/marc21` is used
    json_directory : Optional[str]
        the directory where generated json cache files are stored at; if None, the directory
        `SLUB_CACHE_DIR/k10plus/json` is used
    download : bool, optional
        whether to download k10plus marc21 xml dump files if they do not exist yet, by default True
    workers : Optional[int]
        the number of worker threads to process marc21 xml files; if None, the number of available cpu cores is used
    line_batch_size : int
        the number of lines that are read in one batch to improve performance, by default 1000
    """
    if xml_directory is None:
        xml_directory = os.path.join(get_resources_dir(), "k10plus/marc21/")
    if json_directory is None:
        json_directory = os.path.join(get_cache_dir(), "k10plus/json/")
    if workers is None:
        workers = len(os.sched_getaffinity(0))

    if download:
        download_k10plus_marc21_files()
        load_fasttext_language_detector()

    with multiprocessing.Pool(workers) as pool:
        xml_filepaths = list(k10plus_marc21_xml_files(xml_directory))
        json_filepaths = [k10plus_json_cache_filepath_from_xml_filepath(fp, json_directory) for fp in xml_filepaths]
        args = [(xfp, jfp, line_batch_size) for xfp, jfp in list(zip(xml_filepaths, json_filepaths))]
        pool.starmap(k10plus_transform_and_save_xml_to_json, args)


def k10plus_read_from_json_cache(
    xml_directory: str = None,
    json_directory: str = None,
    download: bool = True,
    workers: Optional[int] = None,
    line_batch_size: int = 1000,
):
    """Read from all k10plus json cache files (and create json cache if missing).

    Parameters
    ----------
    xml_directory : Optional[str]
        the directory where k10plus marc21 xml dump files are stored; if None, the directory
        `SLUB_RESOURCE_DIR/k10plus/marc21` is used
    json_directory : Optional[str]
        the directory where generated json cache files are stored at; if None, the directory
        `SLUB_CACHE_DIR/k10plus/json` is used
    download : bool, optional
        whether to download k10plus marc21 xml dump files if they do not exist yet, by default True
    workers : Optional[int]
        the number of worker threads to process marc21 xml files; if None, the number of available cpu cores is used
    line_batch_size : int
        the number of lines that are read in one batch to improve performance, by default 1000
    """
    if json_directory is None:
        json_directory = os.path.join(get_cache_dir(), "k10plus/json/")

    xml_filepaths = list(k10plus_marc21_xml_files(xml_directory))
    json_filepaths = [k10plus_json_cache_filepath_from_xml_filepath(fp, json_directory) for fp in xml_filepaths]

    # create cache if json files are missing
    all_json_exist = False
    if json_filepaths:
        all_json_exist = min(os.path.exists(fp) for fp in json_filepaths)
    if not all_json_exist:
        k10plus_build_json_cache(xml_directory, json_directory, download, workers, line_batch_size)

    # iterate over all json files
    for filepath in json_filepaths:
        logger.debug("reading k10plus json file %s", filepath)
        line_generator = read_gzip_text_lines(filepath)
        while True:
            chunk = list(islice(line_generator, line_batch_size))

            if not chunk:
                break

            for line in chunk:
                yield json.loads(line)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    k10plus_build_json_cache(workers=8)
