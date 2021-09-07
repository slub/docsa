"""Reads qucosa metadata from jsonl files."""

import os
import gzip
import json
import logging

from typing import Mapping, List, Any
from slub_docsa.common.document import Document
from slub_docsa.common.dataset import Dataset
from slub_docsa.data.load.rvk import get_rvk_subject_store, rvk_notation_to_uri
from slub_docsa.common.paths import CACHE_DIR, RESOURCES_DIR

logger = logging.getLogger(__name__)

QUCOSA_FULLTEXT_MAPPING_DIR = os.path.join(RESOURCES_DIR, "qucosa/fulltext_mapping/")
QUCOSA_SIMPLE_TRAINING_DATA_TSV = os.path.join(CACHE_DIR, "qucosa/simple_training_data.tsv")

ClassificationElementType = Mapping[str, Any]
ClassifcationType = List[ClassificationElementType]


def read_qucosa_metadata():
    """Read qucosa metadata from gzip compressed jsonl files."""
    for entry in os.scandir(QUCOSA_FULLTEXT_MAPPING_DIR):
        if entry.is_file():
            with gzip.open(entry.path, "r") as f_jsonl:
                for line in f_jsonl.readlines():
                    yield json.loads(line)


def get_rvk_notations_from_qucosa_metadata(doc: Mapping[str, Any]) -> List[str]:
    """Return list of RVK notations from qucosa metadata object."""
    classification: ClassifcationType = doc["metadata"]["classification"]
    rvk_dict = list(filter(lambda d: d["type"] == "rvk", classification))[0]
    keys = rvk_dict["keys"]
    if isinstance(keys, str):
        return list(map(lambda s: s.strip(), keys.split(",")))
    if isinstance(keys, list):
        return keys
    return []


def get_document_title_from_qucosa_metadata(doc: Mapping[str, Any]) -> str:
    """Return the document title from a qucosa metadata object."""
    if isinstance(doc["title"]["text"], list):
        return doc["title"]["text"][0]
    return doc["title"]["text"]


def get_document_id_from_qucosa_metadate(doc: Mapping[str, Any]) -> str:
    """Return the document id from a qucosa metadata object."""
    return doc["id"]


def read_qucosa_simple_rvk_training_dataset() -> Dataset:
    """Read qucosa documents and return full training data."""
    logger.debug("load rvk classes and index them by notation")
    rvk_subject_store = get_rvk_subject_store()

    documents = []
    subjects = []
    logger.debug("read qucosa meta data json")
    for doc in read_qucosa_metadata():
        notations = get_rvk_notations_from_qucosa_metadata(doc)
        notations_filtered = list(filter(lambda n: rvk_notation_to_uri(n) in rvk_subject_store, notations))
        subject_uris_filtered = list(map(rvk_notation_to_uri, notations_filtered))
        doc_uri = "uri://" + get_document_id_from_qucosa_metadate(doc)
        doc_title = get_document_title_from_qucosa_metadata(doc)

        if len(doc_title) < 10:
            logger.warning("qucosa document with short title '%s': %s", doc_title, doc_uri)
            continue

        if len(subject_uris_filtered) < 1:
            logger.debug("qucosa document with not rvk subjects: %s", doc_uri)
            continue

        documents.append(
            Document(uri=doc_uri, title=doc_title)
        )
        subjects.append(
            subject_uris_filtered
        )

    return Dataset(documents=documents, subjects=subjects)


def save_qucosa_simple_rvk_training_data_as_annif_tsv():
    """Save qucosa simple RVK training data as annif tsv file."""
    # make sure directory exists
    os.makedirs(os.path.dirname(QUCOSA_SIMPLE_TRAINING_DATA_TSV), exist_ok=True)

    if not os.path.exists(QUCOSA_SIMPLE_TRAINING_DATA_TSV):
        with open(QUCOSA_SIMPLE_TRAINING_DATA_TSV, "w", encoding="utf8") as f_tsv:
            dataset = read_qucosa_simple_rvk_training_dataset()
            for i, doc in enumerate(dataset.documents):
                text = doc.title
                labels_list = dataset.subjects[i]
                labels_str = " ".join(map(lambda l: f"<{l}>", labels_list))

                f_tsv.write(f"{text}\t{labels_str}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    save_qucosa_simple_rvk_training_data_as_annif_tsv()
