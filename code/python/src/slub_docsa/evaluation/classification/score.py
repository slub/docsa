"""Defines various scores that can be used to judge the classification performance of models."""

# pylint: disable=too-many-locals, unused-argument

import logging
import math
from typing import Any, Callable, Optional, Sequence, Tuple, cast

import numpy as np
from sklearn.metrics import f1_score

from slub_docsa.common.score import IncidenceDecisionFunctionType
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.preprocess.subject import subject_ancestors_list
from slub_docsa.evaluation.classification.incidence import extend_incidence_list_to_ancestors
from slub_docsa.evaluation.classification.incidence import threshold_incidence_decision

logger = logging.getLogger(__name__)


def scikit_incidence_metric(
    incidence_decision_function: IncidenceDecisionFunctionType,
    metric_function,
    **kwargs
) -> Callable[[np.ndarray, np.ndarray], float]:
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
    metric_function
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
    metric_function,
    **kwargs
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a function that can be used to score a subject probability matrix against a target incidence matrix.

    Instead of an arbitrary incidence decision function as can be provided in `scikit_incidence_metric`, this function
    finds the best threshold that maximizes the f1 score by evaluating different thresholds. Only thresholds 0.1, 0.2,
    ..., 0.9 are checked, though.

    Parameters
    ----------
    metric_function
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


def cesa_bianchi_h_loss(
    subject_hierarchy: Optional[SubjectHierarchy],
    subject_order: Optional[Sequence[str]],
    log_factor: Optional[float] = None,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return function that calculates the h-loss according to cesa-bianchi et al.

    The h-loss is a hierarchical score that considers not only whether subjects are predicted correctly, but also uses
    the subject hierarchy to weigh mistakes based on their severness. Mistakes near the root of the hierarchy are
    judged to be more severe, and thus, add a higher loss to the overall score.

    A detailed description can be found in:

    N. Cesa-Bianchi, C. Gentile, and L. Zaniboni.
    [Hierarchical classification: Combining Bayes with SVM](
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.98.7332&rep=rep1&type=pdf) (2006)

    Parameters
    ----------
    subject_hierarchy: Optional[SubjectHierarchy]
        the subject hierarchy used to evaluate the serverness of prediction mistakes; if no subject hierarchy is
        provided, a score of `numpy.nan` is returned
    subject_order: Optional[Sequence[str]]
        an subject list mapping subject URIs to the columns of the provided target / predicted incidence matrices; if
        no subject order is provided, a score of `numpy.nan` is returned
    log_factor: Optional[float] = None
        factor used for logartihmic scaling, which helps to visualize the h-loss in plots

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
        a function that scores two incidence matrices (target incidence, predicted incidence) using the cesa-bianchi
        h-loss given the provided subject hierarchy and subject order
    """

    def _nan_results(_true_incidence: np.ndarray, _predicted_incidence: np.ndarray) -> float:
        return np.nan

    if subject_hierarchy is None or subject_order is None:
        # always return nan if there is no subject hierarchy provided
        return _nan_results

    number_of_root_nodes = sum(1 for _ in subject_hierarchy.root_subjects())

    def _find_ancestor_with_error(
            subject_uri: str,
            subject_hierarchy: SubjectHierarchy,
            incidence_list: Sequence[int],
    ) -> str:
        if subject_order is None:
            raise ValueError("can't find ancestor without subject order")
        ancestors = subject_ancestors_list(subject_uri, subject_hierarchy)
        previous_ancestor = ancestors[-1]
        for ancestor in reversed(ancestors[:-1]):
            if ancestor in subject_order:
                ancestor_id = subject_order.index(ancestor)
                if incidence_list[ancestor_id] == 1:
                    break
            previous_ancestor = ancestor
        return previous_ancestor

    def _h_loss(true_array: np.ndarray, predicted_array: np.ndarray) -> float:
        subjects_with_errors = set()
        true_list = true_array.tolist()
        pred_list = predicted_array.tolist()

        ext_true_list = extend_incidence_list_to_ancestors(subject_hierarchy, subject_order, true_list)
        ext_pred_list = extend_incidence_list_to_ancestors(subject_hierarchy, subject_order, pred_list)

        # look for false negative errors
        for i, value in enumerate(true_list):
            if value == 1 and ext_pred_list[i] == 0:
                # there is an error here, lets find out at which level
                subject_uri = subject_order[i]
                subjects_with_errors.add(_find_ancestor_with_error(
                    subject_uri,
                    subject_hierarchy,
                    ext_pred_list
                ))

        # look for false positive errors
        for i, value in enumerate(pred_list):
            if value == 1 and ext_true_list[i] == 0:
                subject_uri = subject_order[i]
                subjects_with_errors.add(_find_ancestor_with_error(
                    subject_uri,
                    subject_hierarchy,
                    ext_true_list
                ))

        nonlocal number_of_root_nodes
        loss = 0.0
        for subject_uri in subjects_with_errors:
            ancestors = subject_ancestors_list(subject_uri, subject_hierarchy)
            max_level_loss = 1.0
            for ancestor in ancestors:
                ancestor_parent = subject_hierarchy.subject_parent(ancestor)
                if ancestor_parent is None:
                    max_level_loss *= 1.0 / number_of_root_nodes
                else:
                    number_of_siblings = len(subject_hierarchy.subject_children(ancestor_parent))
                    max_level_loss *= 1.0 / number_of_siblings
            loss += max_level_loss

        return loss

    def _apply_log_factor(h_loss):
        if log_factor is None:
            raise ValueError("can not apply log factor that is None")
        return math.log(1 + h_loss * log_factor, 2) / math.log(log_factor + 1, 2)

    def _h_loss_array(true_incidence: np.ndarray, predicted_incidence: np.ndarray) -> float:
        h_losses = list(map(_h_loss, cast(Any, true_incidence), cast(Any, predicted_incidence)))
        if log_factor is not None:
            h_losses = list(map(_apply_log_factor, h_losses))
        return cast(float, np.average(h_losses))

    return _h_loss_array
