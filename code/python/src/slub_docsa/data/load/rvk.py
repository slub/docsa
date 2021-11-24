"""Reads and processes RVK classes loaded from the official xml."""

# pylint: disable=too-few-public-methods

import os
import urllib.parse
import urllib.request
import shutil
import zipfile
import io
import logging

from typing import Iterable, Any, List, Optional, Tuple

import rdflib
from lxml import etree  # nosec
from rdflib.namespace import SKOS
from slub_docsa.common.paths import RESOURCES_DIR, CACHE_DIR
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNode
from slub_docsa.data.store.subject import SubjectHierarchyDbmStore
from slub_docsa.data.preprocess.subject import subject_label_breadcrumb

logger = logging.getLogger(__name__)

RVK_XML_URL = "https://rvk.uni-regensburg.de/downloads/rvko_xml.zip"
"""URL used to download the RVK xml file"""

RVK_XML_FILE_PATH = os.path.join(RESOURCES_DIR, "rvk/rvko_xml.zip")
"""Default filepath where downloaded xml file is stored."""

RVK_SUBJECT_STORE_PATH = os.path.join(CACHE_DIR, "rvk/rvk_store.dbm")
"""Default filepath where processed RVK hierarchy is stored as cache."""

RVK_ANNIF_TSV_FILE_PATH = os.path.join(CACHE_DIR, "rvk/rvk_annif.tsv")
"""Default filepath where RVK subjects are exported to as TSV file."""


class RvkSubjectNode(SubjectNode):
    """Extends the base subject with an RVK notation string."""

    __slots__ = ("uri", "label", "notation", "parent_uri")

    notation: str

    def __init__(self, uri: str, label: str, notation: str, parent_uri: Optional[str]):
        """Initialize RVK subject."""
        super().__init__(uri, label, parent_uri)
        self.notation = notation


def _download_rvk_xml(
    download_url: str = RVK_XML_URL,
    xml_filepath: str = RVK_XML_FILE_PATH,
) -> None:
    """Download the RVK xml file from rvk.uni-regensburg.de."""
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
    depth: int = None
) -> Iterable[RvkSubjectNode]:
    """Read classes and their labels from the official RVK xml zip archive file.

    Parameters
    ----------
    filepath: str
        The path to the RVK xml file
    depth: int
        The maximum hieararchy level at which subjects are iterated

    Returns
    -------
    Iterable[RvkSubjectNode]
        A generator of RvkSubjectNodes as parsed from the xml file
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
                        label = node.get("benennung")
                        parent_notation = _get_parent_notation(node)
                        parent_uri = None if parent_notation is None else rvk_notation_to_uri(parent_notation)

                        yield RvkSubjectNode(uri, label, notation, parent_uri)
                    level -= 1


def read_rvk_subjects(
    depth: int = None,
    download_url: str = RVK_XML_URL,
    xml_filepath: str = RVK_XML_FILE_PATH,
) -> Iterable[RvkSubjectNode]:
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
    Iterable[RvkSubjectNode]
        A generator of RvkSubjectNodes as parsed from the xml file downloaded via `_download_rvk_xml()`.
    """
    # make sure file is available and download if necessary
    _download_rvk_xml(download_url, xml_filepath)
    return read_rvk_subjects_from_file(xml_filepath, depth)


def get_rvk_subject_store(
    store_filepath: str = RVK_SUBJECT_STORE_PATH,
    depth: int = None,
    download_url: str = RVK_XML_URL,
    xml_filepath: str = RVK_XML_FILE_PATH,
) -> SubjectHierarchyType[RvkSubjectNode]:
    """Store all RVK classes in a dictionary indexed by notation.

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
    SubjectHierarchyType[RvkSubjectNode]
        The RVK subject hierarchy loaded from the filepath
    """
    if not os.path.exists(store_filepath):
        logger.debug("create and fill RVK subject store (may take some time)")
        store = SubjectHierarchyDbmStore[RvkSubjectNode](store_filepath, read_only=False)

        for i, rvk_subject in enumerate(read_rvk_subjects(depth, download_url, xml_filepath)):
            store[rvk_subject.uri] = rvk_subject
            if i % 10000 == 0:
                logger.debug("Added %d RVK subjects to store so far", i)

        store.close()

    return SubjectHierarchyDbmStore[RvkSubjectNode](store_filepath, read_only=True)


def convert_rvk_classes_to_annif_tsv(
    rvk_subject_hierarchy: SubjectHierarchyType[RvkSubjectNode],
    tsv_filepath: str = RVK_ANNIF_TSV_FILE_PATH,
):
    """Convert RVK classes to tab-separated values file required by Annif.

    Parameters
    ----------
    rvk_subject_hierarchy: SubjectHierarchyType[RvkSubjectNode]
        The RVK subject hierarchy as loaded via e.g. `get_rvk_subject_store`
    tsv_filepath: str = RVK_ANNIF_TSV_FILE_PATH,
        The path to a file where RVK subjects are stored in tab-separated format

    Returns
    -------
    None
    """
    if not os.path.exists(tsv_filepath):
        os.makedirs(os.path.dirname(tsv_filepath), exist_ok=True)

        logger.debug("convert RVK classes to annif tsv format")
        with open(tsv_filepath, "w", encoding="utf8") as f_tsv:
            for uri, rvk_subject_node in rvk_subject_hierarchy.items():
                breadcrumb = subject_label_breadcrumb(rvk_subject_node, rvk_subject_hierarchy)
                f_tsv.write(f"<{uri}>\t{breadcrumb}\n")


def generate_rvk_custom_skos_triples(subject_node: RvkSubjectNode) -> List[Tuple[Any, Any, Any]]:
    """Return additional skos triples that should be added to an SKOS graph for each subject node.

    Is only used in combination with `slub_docsa.data.preprocess.skos.subject_hierarchy_to_skos_graph`.

    Parameters
    ----------
    subject_node: RvkSubjectNode
        The RVK subject node that is being transformed to a SKOS format

    Returns
    -------
    List[Tuple[Any, Any, Any]]
        A list of additional triples, in this case only the triple describing the subject notation.
    """
    return [(rdflib.URIRef(subject_node.uri), SKOS.notation, rdflib.Literal(subject_node.notation))]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    rvk_store = get_rvk_subject_store()
    print(f"RVK has {len(rvk_store)} subjects")
