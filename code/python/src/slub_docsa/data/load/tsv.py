"""Load and save data as TSV files."""

import os
import gzip

from typing import Iterable, Mapping

from slub_docsa.common.dataset import Dataset, SimpleDataset
from slub_docsa.common.document import Document
from slub_docsa.data.preprocess.document import document_as_concatenated_string


def save_dataset_as_annif_tsv(
    dataset: Dataset,
    tsv_filepath: str,
):
    """Save both documents and subjects to two TSV files that can be loaded by Annif.

    Parameters
    ----------
    dataset: Dataset
        The dataset that is being saved as tsv file
    tsv_filepath: str
        the path to the tsv file that is being written

    Returns
    -------
    None
    """
    # write documents
    with open(tsv_filepath, "w", encoding="utf8") as f_tsv:
        for i, doc in enumerate(dataset.documents):
            text = document_as_concatenated_string(doc)
            text = text.replace("\n", " ").replace("\r", "").replace("\t", " ").replace("  ", " ")
            subject_uri_list = dataset.subjects[i]
            subject_uri_str = " ".join(map(lambda uri: f"<{uri}>", subject_uri_list))
            f_tsv.write(f"{text}\t{subject_uri_str}\n")


def save_subject_labels_as_annif_tsv(
    subject_labels: Mapping[str, str],
    tsv_filepath: str,
):
    """Write subject labels to Annif tsv file.

    Parameters
    ----------
    subject_labels: Mapping[str, str]
        A map of subects and their corresponding labels that will be saved as tsv file
    tsv_filepath: str
        The path to the tsv file that is being written

    Returns
    -------
    None
    """
    with open(tsv_filepath, "w", encoding="utf8") as f_tsv:
        for uri, label in subject_labels.items():
            f_tsv.write(f"<{uri}>\t{label}\n")


def load_dataset_from_gzipped_annif_tsv(
    tsv_filepath: str,
):
    """Load both documents and subjects from a gzipped tsv file as suggested by Annif.

    Parameters
    ----------
    tsv_filepath: str
        the path to the tsv file that is being loaded

    Returns
    -------
    SimpleDataset
        the documents and subjects loaded as SimpleDataset
    """
    # write documents
    uri_prefix = os.path.basename(tsv_filepath)
    documents: Iterable[Document] = []
    subjects: Iterable[Iterable[str]] = []
    i = 0
    with gzip.open(tsv_filepath, "rt", encoding="utf8") as f_tsv:
        while True:
            line = f_tsv.readline()

            if not line:
                break

            text, label_str = line.split("\t")
            labels = [label[1:-1] for label in label_str.strip().split(" ")]

            uri = uri_prefix + "#" + str(i)
            documents.append(Document(uri=uri, title=text))
            subjects.append(labels)

            i += 1

    return SimpleDataset(documents=documents, subjects=subjects)
