"""Methods to preprocess datasets."""

import logging

from typing import Iterator, Set, Callable

from slub_docsa.common.sample import Sample
from slub_docsa.common.dataset import Dataset, SimpleDataset
from slub_docsa.data.preprocess.subject import count_number_of_samples_by_subjects

logger = logging.getLogger(__name__)


def filter_samples_by_condition(
    samples_iterator: Iterator[Sample],
    condition: Callable[[Sample], bool]
) -> Iterator[Sample]:
    """Return a new dataset that contains only samples matching a condition.

    Parameters
    ----------
    samples_iterator: Iterator[Sample]
        An iterator of samples that is being filtered for samples that match a condition
    condition: Callable[[Sample], bool]
        A function that decides whether a sample should be filtered or not. If the function returns false, the sample
        is filtered (meaning it is not passed along).

    Returns
    -------
    Iterator[Sample]
        An iterator over non-filtered samples, meaning samples for which the condition function returned true
    """
    for sample in samples_iterator:
        if condition(sample):
            yield sample


def filter_subjects_from_dataset(dataset: Dataset, subject_set: Set[str]) -> Dataset:
    """Return a new dataset which does not contain the specified subjects.

    The resulting dataset may contain less samples, in case removing the specified subjects leaves a document without
    any subject annotations, such that it can not be considered as a training example any more.

    Parameters
    ----------
    dataset: Dataset
        The dataset that is used as the source of samples
    subject_set: Set[str]
        The set of subjects, which will be removed form the dataset

    Returns
    -------
    Dataset
        A new dataset that does not contain the specified subjects.
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

    return SimpleDataset(documents=new_documents, subjects=new_targets)


def filter_subjects_with_insufficient_samples(dataset: Dataset, minimum_samples: int = 1) -> Dataset:
    """Return a new dataset which only contains subjects that meet the required minimum number of samples.

    The resulting dataset may contain less samples, in case a all subject annotations for a document need to be
    removed, such that it can not be considered as a training example any more.

    Parameters
    ----------
    dataset: Dataset
        The dataset that is the source of samples and use to generate a new filtered dataset
    minimum_samples: int = 1
        The number of minimum samples required for each subject

    Returns
    -------
        A new dataset that does not contain subjects that do not meet the required minimum number of samples.
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
        return filter_subjects_from_dataset(dataset, subject_set_to_be_removed)
    return dataset
