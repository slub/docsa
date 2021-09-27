"""Methods checking various prerequisites or conditions."""

import logging

from slub_docsa.common.dataset import Dataset
from slub_docsa.data.preprocess.dataset import count_number_of_samples_by_subjects

logger = logging.getLogger(__name__)


def check_subjects_have_minimum_samples(dataset: Dataset, minimum_samples: int = 1):
    """Check and fails if some subjects do not have the required minimum number of samples."""
    subject_counts = count_number_of_samples_by_subjects(dataset.subjects)

    subjects_below_minimum = {s_uri for s_uri, c in subject_counts.items() if c < minimum_samples}

    if len(subjects_below_minimum) > 0:
        logger.error(
            "a total of %d subjects do not have a minimum of %d samples",
            len(subjects_below_minimum),
            minimum_samples
        )
        for subject_uri in subjects_below_minimum:
            logger.debug("subject %s only has %d samples", subject_uri, subject_counts[subject_uri])

    assert len(subjects_below_minimum) == 0
