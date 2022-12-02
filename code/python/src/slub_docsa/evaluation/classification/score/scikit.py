"""Defines various scores that can be used to judge the classification performance of models."""

# pylint: disable=too-many-locals, unused-argument

import logging

import numpy as np
from sklearn.metrics import f1_score

from slub_docsa.common.score import BinaryClassProbabilitiesScore
from slub_docsa.common.score import IncidenceDecisionFunction
from slub_docsa.common.score import MultiClassProbabilitiesScore, MultiClassIncidenceScore
from slub_docsa.evaluation.classification.incidence import threshold_incidence_decision

logger = logging.getLogger(__name__)


def scikit_incidence_metric(
    incidence_decision_function: IncidenceDecisionFunction,
    metric_function: MultiClassIncidenceScore,
    **kwargs
) -> BinaryClassProbabilitiesScore:
    """Return a function that can be used to score a subject probability matrix against a target incidence matrix.

    This score function is based on a scikit-learn metric passes as argument `metric_function`. In order to apply this
    metric, first, subject probabilities are converted to an incidence matrix using the provided
    `incidence_decision_function`.

    Parameters
    ----------
    incidence_decision_function: IncidenceDecisionFunctionType
        a function that transforms a subject probability matrix (as numpy.ndarray) to an incidence matrix (binary
        matrix of same shape) using some decision logic, e.g., a threshold decision via
        `slub_docsa.evaluation.incidence.threshold_incidence_decision`
    metric_function: MultiClassIncidenceScoreFunction
        a function that scores the resulting incidence matrix (after applying the decision function), e.g., scikit's
        `precision_score` function
    kwargs
        any additional arguments that are passed to the `metric_function`

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
        a function with matrices `true_subject_incidence` and `predicted_subject_probabitlies` as parameters that
        scores the subject probabilitiy matrix against the true target subject incidence matrix by first applying the
        `incidence_decision_function` and then calling `metric_function` with its result

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_score
    >>> from slub_docsa.evaluation.incidence import threshold_incidence_decision
    >>> target_incidence = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    >>> subject_probabilities = np.array([[0.1, 0.2, 0.9], [0.3, 0.7, 0.1]])
    >>> threshold_function = threshold_incidence_decision(0.5)
    >>> score_function = scikit_incidence_metric(threshold_function, precision_score, average="micro")
    >>> score_function(target_incidence, subject_probabilities)
    1.0
    """

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


def scikit_metric_for_best_threshold_based_on_f1score(
    metric_function: MultiClassIncidenceScore,
    **kwargs
) -> MultiClassProbabilitiesScore:
    """Return a function that can be used to score a subject probability matrix against a target incidence matrix.

    Instead of an arbitrary incidence decision function as can be provided in `scikit_incidence_metric`, this function
    finds the best threshold that maximizes the f1 score by evaluating different thresholds. Only thresholds 0.1, 0.2,
    ..., 0.9 are checked, though.

    Parameters
    ----------
    metric_function: MultiClassIncidenceScoreFunction
        a function that scores the resulting incidence matrix (after finding the best threshold), e.g., scikit's
        `precision_score` function
    kwargs
        any additional arguments that are passed to the `metric_function`

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
        a function with matrices `true_subject_incidence` and `predicted_subject_probabitlies` as parameters that
        scores the subject probabilitiy matrix against the true target subject incidence matrix by first finding the
        best incidence threshold (by chosing a threshold that maximizes the f1 score) and then applying the metric
        function
    """

    def _decision(true_incidence, predicted_probabilities: np.ndarray) -> np.ndarray:
        best_score = -1
        best_threshold = None
        best_incidence = np.zeros((2, 2))
        for threshold in [i / 10.0 + 0.1 for i in range(9)]:
            score = scikit_incidence_metric(
                threshold_incidence_decision(threshold),
                f1_score,
                average="micro",
                zero_division=0
            )(
                true_incidence,
                predicted_probabilities
            )
            # logger.debug("score for threshold t=%f is %f", t, score)

            if score > best_score:
                best_incidence = threshold_incidence_decision(threshold)(predicted_probabilities)
                best_score = score
                best_threshold = threshold

        logger.debug("found best f1_score for incidence based on threshold t=%f", best_threshold)
        return best_incidence

    def _metric(
        true_subject_incidence: np.ndarray,
        predicted_subject_probabitlies: np.ndarray,
    ) -> float:
        predicted_subject_incidence = _decision(true_subject_incidence, predicted_subject_probabitlies)
        score = metric_function(true_subject_incidence, predicted_subject_incidence, **kwargs)

        if not isinstance(score, float):
            raise RuntimeError("sklearn metric output is not a float")

        return score

    return _metric
