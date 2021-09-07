"""Defines various scores that can be used to judge the performance of models."""

from typing import Callable, Tuple

import numpy as np

from slub_docsa.common.score import IncidenceDecisionFunctionType


def scikit_incidence_metric(
    incidence_decision_function: IncidenceDecisionFunctionType,
    metric_function,
    **kwargs
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a scikit-learn metric transformed to score lists of subject URIs."""

    def _metric(
        true_subject_incidence: np.ndarray,
        predicted_subject_probabitlies: np.ndarray,
    ) -> float:

        predicted_subject_incidence = incidence_decision_function(predicted_subject_probabitlies)

        score = metric_function(true_subject_incidence, predicted_subject_incidence, **kwargs)

        if not isinstance(score, float):
            raise RuntimeError("sklearn metric output is not a float")

        return score

    return _metric


def absolute_confusion_from_incidence(true_incidence, predicted_incidence) -> Tuple[float, float, float, float]:
    """Return the absolute number of true positives, true negatives, false positives and false negatives."""
    bool_true_incidence = true_incidence > 0.0
    bool_predicted_incidence = predicted_incidence > 0.0

    true_positives = (bool_true_incidence & bool_predicted_incidence)
    true_negatives = (~bool_true_incidence & ~bool_predicted_incidence)
    false_positives = (~bool_true_incidence & bool_predicted_incidence)
    false_negatives = (bool_true_incidence & ~bool_predicted_incidence)

    return np.sum(true_positives), np.sum(true_negatives), np.sum(false_positives), np.sum(false_negatives)
