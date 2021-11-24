"""Load and save data as TSV files."""

from typing import Mapping

from slub_docsa.common.dataset import Dataset
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
            labels_list = dataset.subjects[i]
            labels_str = " ".join(map(lambda l: f"<{l}>", labels_list))
            f_tsv.write(f"{text}\t{labels_str}\n")


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
