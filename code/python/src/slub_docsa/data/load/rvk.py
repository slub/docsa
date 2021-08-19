"""Reads and processes RVK classes loaded from the official xml"""

import os
import urllib.parse
import urllib.request
import shutil
import zipfile
import io
import logging

from typing import Iterable, Any, List, Dict
from typing_extensions import TypedDict
from lxml import etree  # nosec
from slub_docsa.common import RESOURCES_DIR, CACHE_DIR

logger = logging.getLogger(__name__)

RVK_XML_URL = "https://rvk.uni-regensburg.de/downloads/rvko_xml.zip"
RVK_XML_FILE_PATH = os.path.join(RESOURCES_DIR, "rvk/rvko_xml.zip")
RVK_ANNIF_TSV_FILE_PATH = os.path.join(CACHE_DIR, "rvk/rvk_annif.tsv")


class RvkClass(TypedDict):
    """Represents an RVK class"""

    uri: str
    notation: str
    label: str
    ancestors: List["RvkClass"]


def _download_rvk_xml() -> None:
    """Downloads the RVK xml file from rvk.uni-regensburg.de"""
    os.makedirs(os.path.dirname(RVK_XML_FILE_PATH), exist_ok=True)
    if not os.path.exists(RVK_XML_FILE_PATH):
        logger.debug("download RVK classes to %s", RVK_XML_FILE_PATH)
        with urllib.request.urlopen(RVK_XML_URL) as f_src, open(RVK_XML_FILE_PATH, "wb") as f_dst:  # nosec
            shutil.copyfileobj(f_src, f_dst)


def rvk_notation_to_uri(notation: str) -> str:
    """Converts a rvk notation to an URI"""
    notation_encoded = urllib.parse.quote(notation)
    return f"https://rvk.uni-regensburg.de/api/xml/node/{notation_encoded}"


def _get_ancestors(element: Any) -> List[RvkClass]:
    """Collects labels and notations from all parent classes"""

    ancestors: List[RvkClass] = []
    next_element = element.getparent()
    while next_element is not None:
        if next_element.tag == "node":
            notation = next_element.get("notation")
            label = next_element.get("benennung")
            ancestors.insert(0, {
                "uri": rvk_notation_to_uri(notation),
                "label": label,
                "notation": notation,
                "ancestors": []
            })
        next_element = next_element.getparent()

    return ancestors


def _get_breadcrumb_label(rvk_cls: RvkClass) -> str:
    return " | ".join(map(lambda c: c["label"], rvk_cls["ancestors"] + [rvk_cls]))


def read_rvk_classes(depth: int = None) -> Iterable[RvkClass]:
    """Downloads and reads RVK classes and their labels"""

    # make sure file is available and download if necessary
    _download_rvk_xml()

    return read_rvk_classes_from_file(RVK_XML_FILE_PATH, depth)


def read_rvk_classes_from_file(filepath: str, depth: int = None) -> Iterable[RvkClass]:
    """Reads classes and their labels from the official RVK xml zip archive file"""

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
                        label = node.get("benennung")
                        ancestors = _get_ancestors(node)

                        rvk_class: RvkClass = {
                            "uri": rvk_notation_to_uri(notation),
                            "notation": notation,
                            "label": label,
                            "ancestors": ancestors,
                        }

                        yield rvk_class
                    level -= 1


def load_rvk_classes_indexed_by_notation() -> Dict[str, RvkClass]:
    """Stores all RVK classes in a dictionary indexed by notation"""
    index = {}
    for rvk_cls in read_rvk_classes():
        index[rvk_cls["notation"]] = rvk_cls

    return index


def convert_rvk_classes_to_annif_tsv():
    """Converts RVK classes to Tab-separated Values files required by Annif"""
    if not os.path.exists(RVK_ANNIF_TSV_FILE_PATH):
        os.makedirs(os.path.dirname(RVK_ANNIF_TSV_FILE_PATH), exist_ok=True)

        logger.debug("convert RVK classes to annif tsv format")
        with open(RVK_ANNIF_TSV_FILE_PATH, "w") as f_tsv:
            for rvk_cls in read_rvk_classes():
                uri = rvk_cls["uri"]
                breadcrumb = _get_breadcrumb_label(rvk_cls)
                f_tsv.write(f"<{uri}>\t{breadcrumb}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # convert_rvk_classes_to_annif_tsv()

    for cls in read_rvk_classes(depth=None):
        print(cls)
