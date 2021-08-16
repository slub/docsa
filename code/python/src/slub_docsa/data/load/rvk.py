"""Reads and processes RVK classes loaded from the official xml"""

import os
import urllib.parse
import urllib.request
import shutil
import zipfile
import io
import logging

from typing import Iterable, Dict, Any
from lxml import etree  # nosec
from slub_docsa.common import RESOURCES_DIR, CACHE_DIR

logger = logging.getLogger(__name__)

RVK_XML_URL = "https://rvk.uni-regensburg.de/downloads/rvko_xml.zip"
RVK_XML_FILE_PATH = os.path.join(RESOURCES_DIR, "rvk/rvko_xml.zip")
RVK_ANNIF_TSV_FILE_PATH = os.path.join(CACHE_DIR, "rvk/rvk_annif.tsv")


def _download_rvk_xml() -> None:
    """Downloads the RVK xml file from rvk.uni-regensburg.de"""
    os.makedirs(os.path.dirname(RVK_XML_FILE_PATH), exist_ok=True)
    if not os.path.exists(RVK_XML_FILE_PATH):
        logger.debug("download RVK classes to %s", RVK_XML_FILE_PATH)
        with urllib.request.urlopen(RVK_XML_URL) as f_src, open(RVK_XML_FILE_PATH, "wb") as f_dst:  # nosec
            shutil.copyfileobj(f_src, f_dst)


def _get_label_breadcrumb(element: Any) -> str:
    """Combines labels of all parent classes into a single breadcrumb label"""

    label = element.get("benennung")
    next_element = element.getparent()
    while next_element is not None:
        if next_element.tag == "node":
            label = next_element.get("benennung") + " | " + label
        next_element = next_element.getparent()

    return label


def read_rvk_classes(depth: int = None) -> Iterable[Dict[str, str]]:
    """Reads classes and their labels from the official RVK xml"""

    # make sure file is available and download if necessary
    _download_rvk_xml()

    with zipfile.ZipFile(RVK_XML_FILE_PATH, "r") as f_zip:
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
                        notation_encoded = urllib.parse.quote(notation)
                        label = node.get("benennung")
                        breadcrumb = _get_label_breadcrumb(node)

                        yield {
                            "uri": f"https://rvk.uni-regensburg.de/api/xml/node/{notation_encoded}",
                            "notation": notation,
                            "label": label,
                            "breadcrumb": breadcrumb,
                        }
                    level -= 1


def convert_rvk_classes_to_annif_tsv():
    """Converts RVK classes to Tab-separated Values files required by Annif"""
    if not os.path.exists(RVK_ANNIF_TSV_FILE_PATH):
        os.makedirs(os.path.dirname(RVK_ANNIF_TSV_FILE_PATH), exist_ok=True)

        logger.debug("convert RVK classes to annif tsv format")
        with open(RVK_ANNIF_TSV_FILE_PATH, "w") as f_tsv:
            for rvk_cls in read_rvk_classes():
                uri = rvk_cls["uri"]
                breadcrumb = rvk_cls["breadcrumb"]
                f_tsv.write(f"<{uri}>\t{breadcrumb}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # convert_rvk_classes_to_annif_tsv()

    for cls in read_rvk_classes(depth=None):
        print(cls)
