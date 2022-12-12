"""Common methods used to score the performance of classification models."""

from typing import Sequence, Tuple, Union

import numpy as np


def f1_score(
    true_positive: Union[float, np.ndarray],
    false_positive: Union[float, np.ndarray],
    false_negative: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate f1 score.

    Parameters
    ----------
    true_positive : Union[float, np.ndarray]
        the number of true positive cases
    false_positive : Union[float, np.ndarray]
        the number of false positive cases
    false_negative : Union[float, np.ndarray]
        the number of false negative cases

    Returns
    -------
    Union[float, np.ndarray]
        the f1 score
    """
    precision = precision_score(true_positive, false_positive)
    recall = recall_score(true_positive, false_negative)
    return 2 * precision * recall / (precision + recall + 1e-7)


def precision_score(
    true_positive: Union[float, np.ndarray],
    false_positive: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate precison score.

    Parameters
    ----------
    true_positive : Union[float, np.ndarray]
        the number of true positive cases
    false_positive : Union[float, np.ndarray]
        the number of false positive cases

    Returns
    -------
    Union[float, np.ndarray]
        the precision score
    """
    return true_positive / (true_positive + false_positive + 1e-7)


def recall_score(
    true_positive: Union[float, np.ndarray],
    false_negative: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate recall score.

    Parameters
    ----------
    true_positive : Union[float, np.ndarray]
        the number of true positive cases
    false_negative : Union[float, np.ndarray]
        the number of false negative cases

    Returns
    -------
    Union[float, np.ndarray]
        the recall score
    """
    return true_positive / (true_positive + false_negative + 1e-7)


def absolute_confusion_from_incidence(
    true_incidence: np.ndarray,
    predicted_incidence: np.ndarray
) -> Tuple[float, float, float, float]:
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


def subject_score_distribution_from_scores(scores: Sequence[float]) -> Sequence[int]:
    """Extract score distribution from array of scores with 10 bins."""
    distribution = np.histogram(scores, bins=np.arange(0, 1.1, 0.1))[0]
    return distribution.tolist()
