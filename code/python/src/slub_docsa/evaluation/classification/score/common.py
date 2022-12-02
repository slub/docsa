"""Common methods used to score the performance of classification models."""

from typing import Tuple

import numpy as np


def f1_score(true_positive, false_positive, false_negative):
    """Calculate f1 score."""
    precision = precision_score(true_positive, false_positive)
    recall = recall_score(true_positive, false_negative)
    return 2 * precision * recall / (precision + recall + 1e-7)


def precision_score(true_positive, false_positive):
    """Calculate f1 score."""
    return true_positive / (true_positive + false_positive + 1e-7)


def recall_score(true_positive, false_negative):
    """Calculate f1 score."""
    return true_positive / (true_positive + false_negative + 1e-7)


def absolute_confusion_from_incidence(true_incidence, predicted_incidence) -> Tuple[float, float, float, float]:
    """Return the absolute number of true positives, true negatives, false positives and false negatives.

    Parameters
    ----------
    true_incidence: numpy.ndarray
        the true target incidence matrix
    predicted_incidence: numpy.ndarray
        the predicted incidence matrix

    Returns
    -------
    Tuple[float, float, float, float]
        a tuple of four values that contain the the absolute number of true positives, true negatives, false positives
        and false negatives (in that order)
    """
    bool_true_incidence = true_incidence > 0.0
    bool_predicted_incidence = predicted_incidence > 0.0

    true_positives = (bool_true_incidence & bool_predicted_incidence)
    true_negatives = (~bool_true_incidence & ~bool_predicted_incidence)
    false_positives = (~bool_true_incidence & bool_predicted_incidence)
    false_negatives = (bool_true_incidence & ~bool_predicted_incidence)

    return np.sum(true_positives), np.sum(true_negatives), np.sum(false_positives), np.sum(false_negatives)
