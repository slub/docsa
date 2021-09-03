"""Defines various scores that can be used to judge the performance of models."""

from typing import Callable

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
