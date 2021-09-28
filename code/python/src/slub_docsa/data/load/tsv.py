"""Load and save data as TSV files."""

from typing import Iterable

from slub_docsa.common.dataset import Dataset
from slub_docsa.data.preprocess.document import document_as_concatenated_string


def save_dataset_as_annif_tsv(
    dataset: Dataset,
    document_tsv_filepath: str,
):
    """Save both documents and subjects to two TSV files that can be loaded by Annif."""
    # write documents
    with open(document_tsv_filepath, "w", encoding="utf8") as f_tsv:
        for i, doc in enumerate(dataset.documents):
            text = document_as_concatenated_string(doc)
            text = text.replace("\n", " ").replace("\r", "").replace("\t", " ").replace("  ", " ")
            labels_list = dataset.subjects[i]
            labels_str = " ".join(map(lambda l: f"<{l}>", labels_list))
            f_tsv.write(f"{text}\t{labels_str}\n")


def save_subject_targets_as_annif_tsv(
    subject_target_list: Iterable[str],
    subject_tsv_filepath: str,
):
    """Write subject targets to Annif tsv file."""
    with open(subject_tsv_filepath, "w", encoding="utf8") as f_tsv:
        for sub in subject_target_list:
            f_tsv.write(f"<{sub}>\t{sub}\n")
