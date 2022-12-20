"""Download and load subject hierarchies from jskos json files."""

# pylint: disable=too-many-arguments

import logging
import os
import gzip
import json
import urllib

from typing import Any, Mapping, Optional

from slub_docsa.common.paths import get_resources_dir, get_cache_dir
from slub_docsa.common.subject import SimpleSubjectHierarchy, SubjectHierarchy, print_subject_hierarchy
from slub_docsa.data.load.common import download_file
from slub_docsa.data.preprocess.subject import root_subjects_from_subject_parent_map
from slub_docsa.data.preprocess.subject import children_map_from_subject_parent_map
from slub_docsa.data.store.subject import SqliteSubjectHierarchy
from slub_docsa.data.load.subjects.ddc import ddc_parent_from_uri

logger = logging.getLogger(__name__)

JSKOS_DOWNLOAD_URI_BY_SCHEMA = {
    "ddc": "https://coli-conc.gbv.de/api/voc/concepts?uri=http://dewey.info/scheme/edition/e23/&download=json",
    "bk": "https://api.dante.gbv.de/export/download/bk/default/bk__default.jskos.jsonld",
    "rvk": "https://coli-conc.gbv.de/rvk/data/2022_3/rvko_2022_3.raw.ndjson",
    # "gnd": "http://bartoc.org/en/node/430"
}


def default_jskos_download_url(
    schema: str
) -> str:
    """Return default download url for supported schema.

    Parameters
    ----------
    schema : str
        the schema ("rvk", "bk", "ddc")

    Returns
    -------
    str
        the download url for the jskos file
    """
    return JSKOS_DOWNLOAD_URI_BY_SCHEMA[schema]


def default_jskos_schema_json_filepath(
    schema: str
) -> str:
    """Return default filepath for json schema file.

    Parameters
    ----------
    schema : str
        the schema ("rvk", "bk", "ddc")

    Returns
    -------
    str
        the default filepath that is used to store the downloaded jskos json file
    """
    return os.path.join(get_resources_dir(), f"jskos/{schema}.json.gz")


def default_jskos_schema_cache_filepath(
    schema: str
) -> str:
    """Return default filepath to sqlite cache for subject hierarchy.

    Parameters
    ----------
    schema : str
        the schema ("rvk", "bk", "ddc")

    Returns
    -------
    str
        the default filepath to the generated Sqlite file that caches the subject hierarchy
    """
    return os.path.join(get_cache_dir(), f"jskos/{schema}.sqlite")


def download_jskos_json_file(
    schema: str,
    url: Optional[str] = None,
    filepath: Optional[str] = None,
):
    """Download a schema json file.

    Parameters
    ----------
    schema : str
        the schema ("rvk", "bk", "ddc")
    url : Optional[str], optional
        the download url for the jskos json file, if None, the default download url for the requested schema is
        determined via `default_jskos_download_url`
    filepath : Optional[str], optional
        the filepath where the downloaded jskos json file is stored, if None, the default storage location for the
        requested schema is determined via `default_jskos_schema_json_filepath`

    Returns
    -------
    None
        as soon as the file is downloaded; or in case the file already exists

    Raises
    ------
    ValueError
        if no download url and filepath is provided for a schema that is not already known ("rvk", "ddc", "bk")
    """
    if filepath is None:
        filepath = default_jskos_schema_json_filepath(schema)

    if os.path.exists(filepath):
        return

    logger.debug("download jskos file for schema '%s'", schema)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if url is None:
        if schema not in JSKOS_DOWNLOAD_URI_BY_SCHEMA:
            raise ValueError(f"no jskos download url known for schema '{schema}', please provide one")
        url = default_jskos_download_url(schema)

    download_file(url, filepath, gzip_compress=True)


def _repair_ddc_subject_hierarchy_from_coliconc(subject_parent: Mapping[str, str]):
    """Repair the ddc subject hierarchy downloaded from the Coli-Conc API.

    The DDC subject hierarchy from Coli-Conc contains a few subjects that have no parent or incorrect parent subject
    URIs. These cases are repaired by generating artificial parent subject URIs by removing the last digit from the
    DDC notation until the resulting DDC notation is found amongst all valid DDC subjects.

    Parameters
    ----------
    subject_parent : Mapping[str, str]
        the mapping of DDC subject URIs to their parents subject URI that is repaired
    """
    # repair incorrect ddc parents
    incorrect_parents = ["http://dewey.info/class/617.4-617.483/e23/", "http://dewey.info/class/355.01/e23/"]
    for parent_uri in incorrect_parents:
        del subject_parent[parent_uri]
    missing_parents = set(subject_parent.values()).difference(subject_parent.keys()).union(incorrect_parents)
    # add missing parents
    for parent_uri in missing_parents:
        while parent_uri is not None and parent_uri not in subject_parent:
            logger.debug("add ddc parent for '%s'", parent_uri)
            subject_parent[parent_uri] = ddc_parent_from_uri(parent_uri)
            parent_uri = ddc_parent_from_uri(parent_uri)


def _repair_jskos_subject_hierarchy(
    subject_parent: Mapping[str, str],
    subject_labels: Mapping[str, Mapping[str, str]],
    subject_notation: Mapping[str, str],
):
    """Check and repair any jskos subject hierarchy.

    Verifies that all information (parent, labels, notation) is available for all subject URIs.
    If anything is missing, the data structure for this subject is initialized with its default values (no parent,
    no labels, no notation).
    """
    # make sure that referenced subjects are available
    # such that there will not be any broken links
    all_subjects = set(subject_parent.keys()) \
        .union(subject_parent.values()) \
        .union(subject_labels.keys()) \
        .union(subject_notation.keys()) \
        .difference([None])
    for missing_uri in all_subjects:
        if missing_uri not in subject_parent:
            logger.debug("add missing parent for subject uri '%s'", missing_uri)
            subject_parent[missing_uri] = None
        if missing_uri not in subject_labels:
            logger.debug("add missing labels for subject uri '%s'", missing_uri)
            subject_labels[missing_uri] = {}
        if missing_uri not in subject_notation:
            logger.debug("add missing notation for subject uri '%s'", missing_uri)
            subject_notation[missing_uri] = None


def _parse_subject_from_jskos_object(
    subject_data: Mapping[str, Any],
    subject_parent: Mapping[str, str],
    subject_labels: Mapping[str, Mapping[str, str]],
    subject_notation: Mapping[str, str],
):
    """Transfer information (parent, label, notation) from a jskos object to the subject hierarchy."""
    subject_uri = subject_data["uri"]
    subject_labels[subject_uri] = subject_data.get("prefLabel", {})
    if not subject_labels[subject_uri]:
        subject_labels[subject_uri] = subject_data.get("http://www.w3.org/2004/02/skos/core#prefLabel", {})
    subject_notation[subject_uri] = subject_data["notation"]
    if "broader" in subject_data and len(subject_data["broader"]) > 0:
        parent_uri = subject_data["broader"][0]["uri"]
        subject_parent[subject_uri] = parent_uri
    else:
        subject_parent[subject_uri] = None


def build_jskos_subject_hierarchy(
    schema: str,
    download_url: Optional[str] = None,
    json_filepath: Optional[str] = None,
    download: bool = True,
) -> SubjectHierarchy:
    """Build a subject hierarchy by parsing a JSKOS file optionally downloaded from a URL.

    Parameters
    ----------
    schema : str
        the schema that is represented in the JSKOS file ("rvk", "ddc", "bk")
    download_url : Optional[str], optional
        the download url for the jskos json file, if None, the default download url for the requested schema is
        determined via `default_jskos_download_url`
    json_filepath : Optional[str], optional
        the filepath where the downloaded jskos json file is stored, if None, the default storage location for the
        requested schema is determined via `default_jskos_schema_json_filepath`
    download : bool, optional
        whether to do the download in case the file does not exist yet, by default True

    Returns
    -------
    SubjectHierarchy
        the subject hierarchy that is generated by parsing the JSKOS file
    """
    if json_filepath is None:
        json_filepath = default_jskos_schema_json_filepath(schema)

    if download:
        download_jskos_json_file(schema, download_url, json_filepath)

    def _parse_full_file(json_file):
        for subject_data in json.load(json_file):
            yield subject_data

    def _parse_file_line_by_line(json_file):
        for line in json_file:
            yield json.loads(line)

    def _is_ndjson_file(json_file):
        try:
            for line in json_file:
                json.loads(line)
                break
            return True
        except json.decoder.JSONDecodeError:
            return False
        finally:
            json_file.seek(0)

    subject_parent: Mapping[str, str] = {}
    subject_labels: Mapping[str, Mapping[str, str]] = {}
    subject_notation: Mapping[str, str] = {}
    with gzip.open(json_filepath, "rt") as json_file:
        if _is_ndjson_file(json_file):
            subject_data_generator = _parse_file_line_by_line(json_file)
        else:
            subject_data_generator = _parse_full_file(json_file)

        # iterate over all subject objects in a JSKOS file and collect information
        for subject_data in subject_data_generator:
            _parse_subject_from_jskos_object(subject_data, subject_parent, subject_labels, subject_notation)

    # repair subject hierarchies
    if schema == "ddc":
        _repair_ddc_subject_hierarchy_from_coliconc(subject_parent)
    _repair_jskos_subject_hierarchy(subject_parent, subject_labels, subject_notation)

    root_subjects = root_subjects_from_subject_parent_map(subject_parent)
    subject_children = children_map_from_subject_parent_map(subject_parent)
    return SimpleSubjectHierarchy(root_subjects, subject_labels, subject_parent, subject_children, subject_notation)


def load_jskos_subject_hierarchy_from_sqlite(
    schema: str,
    cache_filepath: Optional[str] = None,
    download_url: Optional[str] = None,
    json_filepath: Optional[str] = None,
    download: bool = True,
    preload_contains: bool = False,
) -> SubjectHierarchy:
    """Load subject hierarchy from sqlite file and generate it from jskos json if it does not exist yet.

    Parameters
    ----------
    schema : str
        the schema that is supposed to be loaded ("rvk", "ddc", "bk")
    cache_filepath: str = None
        The path that is used to cache a loaded subject hierarchy; if None, the cache filepath for the requested schema
        is determined via `default_jskos_schema_cache_filepath`
    download_url: str = None
        the download url for the jskos json file, if None, the default download url for the requested schema is
        determined via `default_jskos_download_url`
    json_filepath: str = None
        the filepath where the downloaded jskos json file is stored, if None, the default storage location for the
        requested schema is determined via `default_jskos_schema_json_filepath`
    download : bool, optional
        whether to do the download in case the file does not exist yet, by default True
    preload_contains: bool, optional
        wether to preload all available subject URIs into memory such that the contains operation
        (subject_uri in subject_hierarchy) can be executed much faster (in comparison to the Sqlite database)

    Returns
    -------
    SubjectHierarchy
        The subject hierarchy loaded from the sqlite file
    """
    if cache_filepath is None:
        cache_filepath = default_jskos_schema_cache_filepath(schema)

    if not os.path.exists(cache_filepath):
        logger.debug("create and build subject store from jskos file for schema '%s'", schema)
        SqliteSubjectHierarchy.save(
            build_jskos_subject_hierarchy(schema, download_url, json_filepath, download),
            cache_filepath
        )
    return SqliteSubjectHierarchy(cache_filepath, preload_contains)


def _print_coliconc_concept_schema_uris():
    """Print all available schema or vocabularies using the Coli-Conc API."""
    url = "https://coli-conc.gbv.de/api/voc"
    with urllib.request.urlopen(url) as response:  # nosec
        data = json.load(response)
        for concept_data in data:
            print(concept_data["uri"], concept_data.get("notation"), concept_data["prefLabel"].get("de"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # _print_coliconc_concept_schema_uris()

    _schema = "bk"  # pylint: disable=invalid-name
    _subject_hierarchy = load_jskos_subject_hierarchy_from_sqlite(_schema)
    print(f"schema '{_schema}' has", sum(1 for _ in _subject_hierarchy), "total subjects")
    print_subject_hierarchy("de", _subject_hierarchy, depth=2)
