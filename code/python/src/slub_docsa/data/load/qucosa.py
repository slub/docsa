"""Reads qucosa metadata from jsonl files"""

import os
import gzip
import json
import logging

from typing import Mapping, List, Any
from slub_docsa.data.load.rvk import load_rvk_classes_indexed_by_notation, rvk_notation_to_uri
from slub_docsa.common import CACHE_DIR, RESOURCES_DIR

logger = logging.getLogger(__name__)

QUCOSA_FULLTEXT_MAPPING_DIR = os.path.join(RESOURCES_DIR, "qucosa/fulltext_mapping/")
QUCOSA_SIMPLE_TRAINING_DATA_TSV = os.path.join(CACHE_DIR, "qucosa/simple_training_data.tsv")

ClassificationElementType = Mapping[str, Any]
ClassifcationType = List[ClassificationElementType]


def read_qucosa_metadata():
    """Reads qucosa metadata from gzip compressed jsonl files"""
    for entry in os.scandir(QUCOSA_FULLTEXT_MAPPING_DIR):
        if entry.is_file():
            with gzip.open(entry.path, "r") as f_jsonl:
                for line in f_jsonl.readlines():
                    yield json.loads(line)


def get_rvk_notations_from_qucosa_metadata(doc: Mapping[str, Any]) -> List[str]:
    """Returns list of RVK notations from qucosa metadata object"""
    classification: ClassifcationType = doc["metadata"]["classification"]
    rvk_dict = list(filter(lambda d: d["type"] == "rvk", classification))[0]
    keys = rvk_dict["keys"]
    if isinstance(keys, str):
        return list(map(lambda s: s.strip(), keys.split(",")))
    if isinstance(keys, list):
        return keys
    return []


def get_document_title_from_qucosa_metadata(doc: Mapping[str, Any]) -> str:
    """Returns the document title from a qucosa metadata object"""
    return doc["title"]["text"]


def read_qucosa_simple_rvk_traing_data():
    """Reads qucosa documents and returns only document title and rvk uri labels"""

    logger.debug("load rvk classes and index them by notation")
    rvk_classes_index = load_rvk_classes_indexed_by_notation()

    logger.debug("iterate over qucosa documents")
    for doc in read_qucosa_metadata():
        notations = get_rvk_notations_from_qucosa_metadata(doc)
        notations_filtered = list(filter(lambda n: n in rvk_classes_index, notations))

        if notations_filtered:

            yield {
                "text": get_document_title_from_qucosa_metadata(doc),
                "labels": list(map(rvk_notation_to_uri, notations_filtered)),
            }


def save_qucosa_simple_rvk_traing_data_as_annif_tsv():
    """Saves qucosa simple RVK training data as annif tsv file"""

    # make sure directory exists
    os.makedirs(os.path.dirname(QUCOSA_SIMPLE_TRAINING_DATA_TSV), exist_ok=True)

    if not os.path.exists(QUCOSA_SIMPLE_TRAINING_DATA_TSV):
        with open(QUCOSA_SIMPLE_TRAINING_DATA_TSV, "w") as f_tsv:
            for doc in read_qucosa_simple_rvk_traing_data():
                text = doc["text"]
                labels_list = doc["labels"]
                labels_str = " ".join(map(lambda l: f"<{l}>", labels_list))

                f_tsv.write(f"{text}\t{labels_str}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    save_qucosa_simple_rvk_traing_data_as_annif_tsv()
