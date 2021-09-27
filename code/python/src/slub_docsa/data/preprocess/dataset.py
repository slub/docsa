"""Methods to preprocess datasets."""

import logging

from typing import Mapping, Set

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.subject import SubjectTargets

logger = logging.getLogger(__name__)


def count_number_of_samples_by_subjects(subject_targets: SubjectTargets) -> Mapping[str, int]:
    """Count the number of occurances of subjects annotations."""
    counts = {}

    for subject_list in subject_targets:
        for subject_uri in subject_list:
            counts[subject_uri] = counts.get(subject_uri, 0) + 1

    return counts


def remove_subjects_from_dataset(dataset: Dataset, subject_set: Set[str]) -> Dataset:
    """Remove subjects from dataset.

    Samples are only removed if all subject annotations will be removed, such that it can not be considered as a
    training example any more.
    """
    new_documents = []
    new_targets = []

    for i, subject_list in enumerate(dataset.subjects):
        new_subject_set = set(subject_list).difference(subject_set)
        if len(new_subject_set) > 0:
            new_documents.append(dataset.documents[i])
            new_targets.append(list(new_subject_set))
        else:
            # sample has no subject annotations left and needs to be removed
            document_uri = dataset.documents[i].uri
            logger.debug("document %s is removed since it has no subject annotations left", document_uri)

    return Dataset(documents=new_documents, subjects=new_targets)


def remove_subjects_with_insufficient_samples(dataset: Dataset, minimum_samples: int = 1) -> Dataset:
    """Remove subjects from a dataset that do not meet the minimum required number of samples.

    Samples are only removed if all subject annotations will be removed, such that it can not be considered as a
    training example any more.
    """
    # count number of samples by subjects
    subject_counts = count_number_of_samples_by_subjects(dataset.subjects)

    # determine which subjects are to be removed
    subject_set_to_be_removed = {s_uri for s_uri, c in subject_counts.items() if c < minimum_samples}

    if len(subject_set_to_be_removed) > 0:
        logger.info(
            "a total of %d subjects are removed due to the minimum requirement of %d samples",
            len(subject_set_to_be_removed),
            minimum_samples
        )
        return remove_subjects_from_dataset(dataset, subject_set_to_be_removed)
    return dataset
