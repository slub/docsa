"""Reads and processes RVK classes loaded from the official xml."""

# pylint: disable=too-few-public-methods

import os
import urllib.parse
import urllib.request
import shutil
import zipfile
import io
import logging

from typing import Iterable, Any, List, NamedTuple, Optional, Tuple, Mapping

import rdflib
from lxml import etree  # nosec
from rdflib.namespace import SKOS
from sqlitedict import SqliteDict

from slub_docsa.common.paths import get_resources_dir, get_cache_dir
from slub_docsa.common.subject import SimpleSubjectHierarchy, SubjectHierarchy
from slub_docsa.data.store.subject import SqliteSubjectHierarchy
from slub_docsa.data.preprocess.subject import children_map_from_subject_parent_map, subject_label_breadcrumb_as_string
from slub_docsa.data.preprocess.subject import root_subjects_from_subject_parent_map

logger = logging.getLogger(__name__)

RVK_XML_URL = "https://rvk.uni-regensburg.de/downloads/rvko_xml.zip"
"""URL used to download the RVK xml file"""


def _get_rvk_xml_filepath():
    """Return filepath where downloaded xml file is stored."""
    return os.path.join(get_resources_dir(), "rvk/rvko_xml.zip")


def _get_rvk_subject_store_path(depth: Optional[int] = None):
    """Return filepath where processed RVK hierarchy is stored as cache."""
    if depth is None:
        return os.path.join(get_cache_dir(), "rvk/rvk_store.sqlite")
    return os.path.join(get_cache_dir(), f"rvk/rvk_store_depth={depth}.sqlite")


def _get_annif_tsv_filepath():
    """Return filepath where RVK subjects are exported to as TSV file."""
    return os.path.join(get_cache_dir(), "rvk/rvk_annif.tsv")


class RvkSubjectHierarchy(SubjectHierarchy):
    """Extends the base subject with an RVK notation string."""

    def subject_notation(self, subject_uri: str) -> str:
        """Return RVK notation for a subject."""
        raise NotImplementedError()


class SimpleRvkSubjectHierarchy(SimpleSubjectHierarchy):
    """Naive implementation of a RVK subject hierarchy."""

    def __init__(
        self,
        root_subjects: Iterable[str],
        subject_labels: Mapping[str, Mapping[str, str]],
        subject_parent: Mapping[str, str],
        subject_children: Mapping[str, Iterable[str]],
        subject_notation: Mapping[str, str]
    ):
        """Initialize with all required information about RVK subjects."""
        super().__init__(root_subjects, subject_labels, subject_parent, subject_children)
        self._subject_notation = subject_notation

    def subject_notation(self, subject_uri: str) -> Optional[str]:
        """Return RVK notation for a subject."""
        return self._subject_notation[subject_uri]


class SqliteRvkSubjectHierarchy(SqliteSubjectHierarchy, RvkSubjectHierarchy):
    """Sqlite store for the RVK subject hierarchy."""

    TABLE_FOR_SUBJECT_NOTATION = "subject_notation"

    def __init__(self, filepath: str):
        """Initialie new RVK subject hierarchy from stored sqlite file."""
        super().__init__(filepath)
        self.notation_store = SqliteDict(filepath, tablename=self.TABLE_FOR_SUBJECT_NOTATION, flag="r")

    def subject_notation(self, subject_uri: str) -> str:
        """Return RVK notation for a subject."""
        return self.notation_store[subject_uri]

    @staticmethod
    def save(rvk_subject_hierarchy: RvkSubjectHierarchy, filepath: str):  # pylint: disable=arguments-renamed
        """Save RVK subject hierarchy to sqlite file."""
        SqliteSubjectHierarchy.save(rvk_subject_hierarchy, filepath)

        # save rvk notation to new sqlite table
        cls = SqliteRvkSubjectHierarchy
        store = SqliteDict(filepath, tablename=cls.TABLE_FOR_SUBJECT_NOTATION, flag="w", autocommit=False)
        for subject_uri in rvk_subject_hierarchy:
            store[subject_uri] = rvk_subject_hierarchy.subject_notation(subject_uri)
        store.commit()
        store.close()


class RvkSubjectTuple(NamedTuple):
    """Tuple containing information parse from RVK xml file."""

    subject_uri: str
    """The uri of the subject."""

    labels: Mapping[str, str]
    """The mapping from ISO 639-1 language codes to labels for the subject."""

    parent_uri: Optional[str]
    """The uri of the parent subject or None if the subject does not have a parent."""

    notation: str
    """The notation for an RVK subject."""


def _download_rvk_xml(
    download_url: str = RVK_XML_URL,
    xml_filepath: Optional[str] = None,
) -> None:
    """Download the RVK xml file from rvk.uni-regensburg.de."""
    if xml_filepath is None:
        xml_filepath = _get_rvk_xml_filepath()
    os.makedirs(os.path.dirname(xml_filepath), exist_ok=True)
    if not os.path.exists(xml_filepath):
        logger.debug("download RVK classes to %s", xml_filepath)
        with urllib.request.urlopen(download_url) as f_src, open(xml_filepath, "wb") as f_dst:  # nosec
            shutil.copyfileobj(f_src, f_dst)


def _get_parent_notation(element: Any) -> Optional[str]:
    """Find notation of parent RVK class."""
    next_element = element.getparent()
    while next_element is not None:
        if next_element.tag == "node":
            return next_element.get("notation")
        next_element = next_element.getparent()


def rvk_notation_to_uri(notation: str) -> str:
    """Convert a RVK notation to an URI.

    Parameters
    ----------
    notation: str
        The notation that references the RVK subject, e.g., "AA 10600"

    Returns
    -------
    str
        An URI as string representing the RVK subject referenced by the notation, e.g.,
        "https://rvk.uni-regensburg.de/api/xml/node/AA%2010600"
    """
    notation_encoded = urllib.parse.quote(notation)
    return f"https://rvk.uni-regensburg.de/api/xml/node/{notation_encoded}"


def read_rvk_subjects_from_file(
    filepath: str,
    depth: Optional[int] = None
) -> Iterable[RvkSubjectTuple]:
    """Read classes and their labels from the official RVK xml zip archive file.

    Parameters
    ----------
    filepath: str
        The path to the RVK xml file
    depth: int
        The maximum hieararchy level at which subjects are iterated

    Returns
    -------
    Iterable[RvkSubjectTuple]
        A generator of RvkSubjectTuple as parsed from the xml file
    """
    with zipfile.ZipFile(filepath, "r") as f_zip:
        for filename in f_zip.namelist():

            # read and parse xml file
            data = f_zip.read(filename)
            node_iterator = etree.iterparse(io.BytesIO(data), events=('start', 'end'), tag="node")
            level = 0

            # collect all classes
            for event, node in node_iterator:
                if event == "start":
                    level += 1
                elif event == "end":
                    if depth is None or level <= depth:
                        notation = node.get("notation")
                        uri = rvk_notation_to_uri(notation)
                        labels = {
                            "de": node.get("benennung")
                        }
                        parent_notation = _get_parent_notation(node)
                        parent_uri = None if parent_notation is None else rvk_notation_to_uri(parent_notation)

                        yield RvkSubjectTuple(uri, labels, parent_uri, notation)
                    level -= 1


def read_rvk_subjects(
    depth: Optional[int] = None,
    download_url: str = RVK_XML_URL,
    xml_filepath: Optional[str] = None,
) -> Iterable[RvkSubjectTuple]:
    """Download and read RVK subjects and their labels.

    Subjects are directly read from the xml file, and not cached. Use `get_rvk_subject_store()` for random cached
    access of RVK subjects.

    Parameters
    ----------
    depth: int
        The maximum hieararchy level at which subjects are iterated.
    download_url: str = RVK_XML_URL
        The url that is used to download the RVK xml file
    xml_filepath: str = RVK_XML_FILE_PATH
        The filepath that is used to store the downloaded RVK xml file

    Returns
    -------
    Iterable[RvkSubjectTuple]
        A generator of RvkSubjectTuple as parsed from the xml file downloaded via `_download_rvk_xml()`.
    """
    if xml_filepath is None:
        xml_filepath = _get_rvk_xml_filepath()
    # make sure file is available and download if necessary
    _download_rvk_xml(download_url, xml_filepath)
    return read_rvk_subjects_from_file(xml_filepath, depth)


def build_rvk_subject_hierarchy(
    depth: Optional[int] = None,
    download_url: str = RVK_XML_URL,
    xml_filepath: Optional[str] = None,
):
    """Build simple RVK subject hierarchy by reading subject information from xml file.

    Parameters
    ----------
    depth: int
        The maximum hieararchy level at which subjects are iterated.
    download_url: str = RVK_XML_URL
        The url that is used to download the RVK xml file
    xml_filepath: str = RVK_XML_FILE_PATH
        The filepath that is used to store the downloaded RVK xml file

    Returns
    -------
    SubjectHierarchy
        The RVK subject hierarchy loaded in memory as simple subject hierarchy
    """
    subject_parent: Mapping[str, str] = {}
    subject_labels: Mapping[str, Mapping[str, str]] = {}
    subject_notation: Mapping[str, str] = {}
    for rvk_tuple in read_rvk_subjects(depth, download_url, xml_filepath):
        subject_parent[rvk_tuple.subject_uri] = rvk_tuple.parent_uri
        subject_labels[rvk_tuple.subject_uri] = rvk_tuple.labels
        subject_notation[rvk_tuple.subject_uri] = rvk_tuple.notation

    root_subjects = root_subjects_from_subject_parent_map(subject_parent)
    subject_children = children_map_from_subject_parent_map(subject_parent)

    return SimpleRvkSubjectHierarchy(root_subjects, subject_labels, subject_parent, subject_children, subject_notation)


def load_rvk_subject_hierarchy_from_sqlite(
    store_filepath: Optional[str] = None,
    depth: Optional[int] = None,
    download_url: str = RVK_XML_URL,
    xml_filepath: Optional[str] = None,
) -> RvkSubjectHierarchy:
    """Load RVK subject hierarchy from sqlite file and generate it if it does not exist yet.

    Parameters
    ----------
    store_filepath: str = RVK_SUBJECT_STORE_PATH
        The path that is used to cache a loaded RVK subject hierarchy
    depth: int
        The maximum hieararchy level at which subjects are iterated.
    download_url: str = RVK_XML_URL
        The url that is used to download the RVK xml file
    xml_filepath: str = RVK_XML_FILE_PATH
        The filepath that is used to store the downloaded RVK xml file

    Returns
    -------
    SubjectHierarchy
        The RVK subject hierarchy loaded from the filepath
    """
    if store_filepath is None:
        store_filepath = _get_rvk_subject_store_path(depth)
    if xml_filepath is None:
        xml_filepath = _get_rvk_xml_filepath()
    if not os.path.exists(store_filepath):
        logger.debug("create and fill RVK subject store (may take some time)")
        SqliteRvkSubjectHierarchy.save(build_rvk_subject_hierarchy(depth, download_url, xml_filepath), store_filepath)

    return SqliteRvkSubjectHierarchy(store_filepath)


def convert_rvk_classes_to_annif_tsv(
    rvk_subject_hierarchy: SubjectHierarchy,
    tsv_filepath: Optional[str] = None,
):
    """Convert RVK classes to tab-separated values file required by Annif.

    Parameters
    ----------
    rvk_subject_hierarchy: SubjectHierarchy
        The RVK subject hierarchy as loaded via e.g. `get_rvk_subject_store`
    tsv_filepath: str = RVK_ANNIF_TSV_FILE_PATH,
        The path to a file where RVK subjects are stored in tab-separated format

    Returns
    -------
    None
    """
    if tsv_filepath is None:
        tsv_filepath = _get_annif_tsv_filepath()
    if not os.path.exists(tsv_filepath):
        os.makedirs(os.path.dirname(tsv_filepath), exist_ok=True)

        logger.debug("convert RVK classes to annif tsv format")
        with open(tsv_filepath, "w", encoding="utf8") as f_tsv:
            for subject_uri in rvk_subject_hierarchy:
                breadcrumb = subject_label_breadcrumb_as_string(subject_uri, "de", rvk_subject_hierarchy)
                f_tsv.write(f"<{subject_uri}>\t{breadcrumb}\n")


def generate_rvk_custom_skos_triples(
    subject_uri: str,
    rvk_subject_hierarchy: RvkSubjectHierarchy
) -> List[Tuple[Any, Any, Any]]:
    """Return additional skos triples that should be added to an SKOS graph for each subject node.

    Is only used in combination with `slub_docsa.data.preprocess.skos.subject_hierarchy_to_skos_graph`.

    Parameters
    ----------
    subject_uri: str
        The uri of the RVK subject that is being transformed to a SKOS format
    subject_hierarchy: RvkSubjectHierarchy
        The RVK subject hierarchy that is queries for additional information about the subject

    Returns
    -------
    List[Tuple[Any, Any, Any]]
        A list of additional triples, in this case only the triple describing the subject notation.
    """
    subject_notation = rvk_subject_hierarchy.subject_notation(subject_uri)
    if subject_notation is not None:
        return [(rdflib.URIRef(subject_uri), SKOS.notation, rdflib.Literal(subject_notation))]
    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    rvk_store = load_rvk_subject_hierarchy_from_sqlite()
    print(f"RVK has {sum(1 for _ in iter(rvk_store))} subjects")
