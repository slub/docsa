"""Methods checking various prerequisites or conditions."""

import logging
from typing import Tuple

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.subject import SubjectTargets
from slub_docsa.data.preprocess.dataset import count_number_of_samples_by_subjects

logger = logging.getLogger(__name__)


def check_dataset_subjects_have_minimum_samples(dataset: Dataset, minimum_samples: int = 1) -> bool:
    """Check and fails if some subjects do not have the required minimum number of samples.

    Parameters
    ----------
    dataset: Dataset
        the dataset to be checked for minimum number of samples per subject
    minimum_samples: int = 1
        the required minimum number of samples per subject

    Returns
    -------
    bool
        true if all subjects have the specified minimum number of samples, else false
    """
    return check_subject_targets_have_minimum_samples(dataset.subjects, minimum_samples)


def check_subject_targets_have_minimum_samples(targets: SubjectTargets, minimum_samples: int = 1) -> bool:
    """Check and fails if some subjects do not have the required minimum number of samples.

    Parameters
    ----------
    targets: SubjectTargets
        subject target list to be checked for minimum number of samples per subject
    minimum_samples: int = 1
        the required minimum number of samples per subject

    Returns
    -------
    bool
        true if all subjects have the specified minimum number of samples, else false
    """
    subject_counts = count_number_of_samples_by_subjects(targets)

    subjects_below_minimum = {s_uri for s_uri, c in subject_counts.items() if c < minimum_samples}

    if len(subjects_below_minimum) > 0:
        logger.error(
            "a total of %d subjects do not have a minimum of %d samples",
            len(subjects_below_minimum),
            minimum_samples
        )
        for subject_uri in subjects_below_minimum:
            logger.error("subject %s only has %d samples", subject_uri, subject_counts[subject_uri])

        return False
    return True


def check_subject_targets_distribution(
    train_targets: SubjectTargets,
    test_targets: SubjectTargets,
    target_ratio_interval: Tuple[float, float]
) -> bool:
    """Check whether training and test split has a good balance for each subject.

    Parameters
    ----------
    train_targets: SubjectTargets
        the training subject target list
    test_targets: SubjectTargets
        the test subject target list
    target_ratio_interval: Tuple[float, float]
        the expected target ratio interval which is checked for every subject

    Returns
    -------
    bool
        true if all ratios for all subjects are between the specified target ratio interval, else false
    """
    well_balanced = True

    train_counts = count_number_of_samples_by_subjects(train_targets)
    test_counts = count_number_of_samples_by_subjects(test_targets)

    train_subject_set = set(train_counts.keys())
    test_subject_set = set(test_counts.keys())

    subjects_only_trained = train_subject_set - test_subject_set
    subjects_only_tested = test_subject_set - train_subject_set

    for subject in subjects_only_trained:
        well_balanced = False
        logger.warning("subject %s is only trained, but not tested", subject)

    for subject in subjects_only_tested:
        well_balanced = False
        logger.warning("subject %s is only tested, but not trained", subject)

    subjects_in_both = train_subject_set.intersection(test_subject_set)

    for subject in subjects_in_both:
        ratio = test_counts[subject] / (test_counts[subject] + train_counts[subject])

        if ratio < target_ratio_interval[0] or ratio > target_ratio_interval[1]:
            well_balanced = False
            logger.warning(
                "subject '%s' is outside target split ratio interval of %s with %d training and %d test samples",
                subject,
                repr(target_ratio_interval),
                train_counts[subject],
                test_counts[subject]
            )

    return well_balanced


def check_dataset_subject_distribution(
    train_dataset: Dataset,
    test_dataset: Dataset,
    target_ratio_interval: Tuple[float, float]
) -> bool:
    """Check whether training and test dataset split has a good balance.

    Uses `check_subject_targets_distribution` to check for the ratio between training and test samples for each subject.

    Parameters
    ----------
    train_dataset: Dataset
        the training dataset to check
    test_dataset: Dataset
        the test dataset to check
    target_ratio_interval: Tuple[float, float]
        the expected target ratio interval which is checked for every subject

    Returns
    -------
    bool
        true if all ratios for all subjects are between the specified target ratio interval, else false
    """
    return check_subject_targets_distribution(
        train_dataset.subjects,
        test_dataset.subjects,
        target_ratio_interval
    )
