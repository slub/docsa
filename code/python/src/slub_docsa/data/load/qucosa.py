"""Reads qucosa metadata from jsonl files."""

import os
import gzip
import json
import logging

from typing import Callable, Iterable, Mapping, List, Any, Optional
from slub_docsa.common.document import Document
from slub_docsa.common.dataset import Dataset
from slub_docsa.data.load.rvk import get_rvk_subject_store, rvk_notation_to_uri
from slub_docsa.common.paths import CACHE_DIR, RESOURCES_DIR

logger = logging.getLogger(__name__)

QUCOSA_FULLTEXT_MAPPING_DIR = os.path.join(RESOURCES_DIR, "qucosa/fulltext_mapping/")
QUCOSA_SIMPLE_TRAINING_DATA_TSV = os.path.join(CACHE_DIR, "qucosa/simple_training_data.tsv")

ClassificationElementType = Mapping[str, Any]
ClassifcationType = List[ClassificationElementType]

QucosaDocument = Mapping[str, Any]


def read_qucosa_metadata() -> Iterable[QucosaDocument]:
    """Read qucosa metadata from gzip compressed jsonl files."""
    for entry in os.scandir(QUCOSA_FULLTEXT_MAPPING_DIR):
        if entry.is_file():
            with gzip.open(entry.path, "r") as f_jsonl:
                for line in f_jsonl.readlines():
                    yield json.loads(line)


def get_rvk_notations_from_qucosa_metadata(doc: QucosaDocument) -> List[str]:
    """Return list of RVK notations from qucosa metadata object."""
    classification: ClassifcationType = doc["metadata"]["classification"]
    rvk_dict = list(filter(lambda d: d["type"] == "rvk", classification))[0]
    keys = rvk_dict["keys"]
    if isinstance(keys, str):
        return list(map(lambda s: s.strip(), keys.split(",")))
    if isinstance(keys, list):
        return keys
    return []


def get_document_title_from_qucosa_metadata(doc: QucosaDocument) -> str:
    """Return the document title from a qucosa metadata object."""
    if isinstance(doc["title"]["text"], list):
        return doc["title"]["text"][0]
    return doc["title"]["text"]


def get_document_abstract_from_qucosa_metadata(doc: QucosaDocument, lang_code: str) -> Optional[str]:
    """Return the document abstract from a qucosa metadata object."""
    for fulltext in doc["fulltext"]:
        if fulltext["type"] == "abstract" and "text" in fulltext and "language" in fulltext:
            language_entry = fulltext["language"]
            text_entry = fulltext["text"]

            # if there are multiple abstracts, find the abstract matching the current language
            if isinstance(language_entry, list) and isinstance(text_entry, list) and lang_code in language_entry:
                language_match_idx = language_entry.index(lang_code)
                return text_entry[language_match_idx]

            # if there is only one abstract, check that it is the correct language
            if isinstance(language_entry, str) and isinstance(text_entry, str) and language_entry == lang_code:
                return text_entry
    return None


def get_document_id_from_qucosa_metadate(doc: QucosaDocument) -> str:
    """Return the document id from a qucosa metadata object."""
    return doc["id"]


def read_qucosa_generic_rvk_training_dataset(
    create_document_from_qucosa: Callable[[QucosaDocument], Optional[Document]]
) -> Dataset:
    """Read qucosa data and extract documents and RVK subjects."""
    logger.debug("load rvk classes and index them by notation")
    rvk_subject_store = get_rvk_subject_store()

    documents = []
    subjects = []
    logger.debug("read qucosa meta data json")
    for doc in read_qucosa_metadata():
        notations = get_rvk_notations_from_qucosa_metadata(doc)
        notations_filtered = list(filter(lambda n: rvk_notation_to_uri(n) in rvk_subject_store, notations))
        subject_uris_filtered = list(map(rvk_notation_to_uri, notations_filtered))

        if len(subject_uris_filtered) < 1:
            logger.debug("qucosa document with no known rvk subjects: %s", get_document_id_from_qucosa_metadate(doc))
            continue

        document = create_document_from_qucosa(doc)
        if document is None:
            continue

        documents.append(document)
        subjects.append(subject_uris_filtered)

    return Dataset(documents=documents, subjects=subjects)


def read_qucosa_titles_rvk_training_dataset() -> Dataset:
    """Read qucosa documents and use document titles as training data."""
    def make_title_only_doc(doc: QucosaDocument) -> Optional[Document]:
        """Only uses title with at least 10 characters as document text."""
        doc_uri = "uri://" + get_document_id_from_qucosa_metadate(doc)
        doc_title = get_document_title_from_qucosa_metadata(doc)

        if len(doc_title) < 10:
            logger.debug("qucosa document with short title '%s': %s", doc_title, doc_uri)
            return None

        return Document(uri=doc_uri, title=doc_title)

    return read_qucosa_generic_rvk_training_dataset(make_title_only_doc)


def read_qucosa_abstracts_rvk_training_dataset() -> Dataset:
    """Read qucosa documents and use document titles and abstracts as training data."""
    def make_title_and_abstract_doc(doc: QucosaDocument) -> Optional[Document]:
        """Only uses title with at least 10 characters as document text."""
        doc_uri = "uri://" + get_document_id_from_qucosa_metadate(doc)
        doc_title = get_document_title_from_qucosa_metadata(doc)
        doc_abstract = get_document_abstract_from_qucosa_metadata(doc, "ger")

        if len(doc_title) < 10:
            logger.debug("qucosa document with too short title '%s': %s", doc_title, doc_uri)
            return None

        if doc_abstract is None:
            logger.debug("qucosa document with no abstract: %s", doc_uri)
            return None

        if len(doc_abstract) < 20:
            logger.debug("qucosa document with too short abstract '%s': %s", doc_abstract, doc_uri)
            return None

        return Document(uri=doc_uri, title=doc_title, abstract=doc_abstract)

    return read_qucosa_generic_rvk_training_dataset(make_title_and_abstract_doc)


def save_qucosa_simple_rvk_training_data_as_annif_tsv():
    """Save qucosa simple RVK training data as annif tsv file."""
    # make sure directory exists
    os.makedirs(os.path.dirname(QUCOSA_SIMPLE_TRAINING_DATA_TSV), exist_ok=True)

    if not os.path.exists(QUCOSA_SIMPLE_TRAINING_DATA_TSV):
        with open(QUCOSA_SIMPLE_TRAINING_DATA_TSV, "w", encoding="utf8") as f_tsv:
            dataset = read_qucosa_titles_rvk_training_dataset()
            for i, doc in enumerate(dataset.documents):
                text = doc.title
                labels_list = dataset.subjects[i]
                labels_str = " ".join(map(lambda l: f"<{l}>", labels_list))

                f_tsv.write(f"{text}\t{labels_str}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    save_qucosa_simple_rvk_training_data_as_annif_tsv()
