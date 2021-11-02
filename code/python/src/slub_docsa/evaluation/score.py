"""Defines various scores that can be used to judge the performance of models."""

# pylint: disable=too-many-locals

import logging
import math
from typing import Any, Callable, Optional, Sequence, Tuple, cast

import numpy as np
from sklearn.metrics import f1_score

from slub_docsa.common.score import IncidenceDecisionFunctionType
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.data.preprocess.subject import children_map_from_subject_hierarchy, subject_ancestors_list
from slub_docsa.evaluation.incidence import extend_incidence_list_to_ancestors, threshold_incidence_decision

logger = logging.getLogger(__name__)


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


def scikit_metric_for_best_threshold_based_on_f1score(
    metric_function,
    **kwargs
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a scikit-learn metric using the best threshold incidence by comparing f1_score."""

    def _decision(true_incidence, predicted_probabilities: np.ndarray) -> np.ndarray:
        best_score = -1
        best_threshold = None
        best_incidence = np.zeros((2, 2))
        for threshold in [i/10.0 + 0.1 for i in range(9)]:
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
    """Return the absolute number of true positives, true negatives, false positives and false negatives."""
    bool_true_incidence = true_incidence > 0.0
    bool_predicted_incidence = predicted_incidence > 0.0

    true_positives = (bool_true_incidence & bool_predicted_incidence)
    true_negatives = (~bool_true_incidence & ~bool_predicted_incidence)
    false_positives = (~bool_true_incidence & bool_predicted_incidence)
    false_negatives = (bool_true_incidence & ~bool_predicted_incidence)

    return np.sum(true_positives), np.sum(true_negatives), np.sum(false_positives), np.sum(false_negatives)


def cesa_bianchi_h_loss(
    subject_hierarchy: Optional[SubjectHierarchyType[SubjectNodeType]],
    subject_order: Optional[Sequence[str]],
    log_factor: Optional[float] = None,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Calculate h-loss according to cesa-bianchi et al."""

    def _nan_results(_true_incidence: np.ndarray, _predicted_incidence: np.ndarray) -> float:
        return np.nan

    if subject_hierarchy is None or subject_order is None:
        # always return nan if there is no subject hierarchy provided
        return _nan_results

    children_map = children_map_from_subject_hierarchy(subject_hierarchy)
    number_of_root_nodes = sum([1 for v in subject_hierarchy.values() if v.parent_uri is None])

    def _find_ancestor_node_with_error(
            subject_uri: str,
            subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
            incidence_list: Sequence[int],
    ) -> str:
        subject_node = subject_hierarchy[subject_uri]
        ancestors = subject_ancestors_list(subject_node, subject_hierarchy)
        previous_ancestor = ancestors[-1]
        for ancestor in reversed(ancestors[:-1]):
            if ancestor.uri in subject_order:
                ancestor_id = subject_order.index(ancestor.uri)
                if incidence_list[ancestor_id] == 1:
                    break
            previous_ancestor = ancestor

        return previous_ancestor.uri

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
                subjects_with_errors.add(_find_ancestor_node_with_error(
                    subject_uri,
                    subject_hierarchy,
                    ext_pred_list
                ))

        # look for false positive errors
        for i, value in enumerate(pred_list):
            if value == 1 and ext_true_list[i] == 0:
                subject_uri = subject_order[i]
                subjects_with_errors.add(_find_ancestor_node_with_error(
                    subject_uri,
                    subject_hierarchy,
                    ext_true_list
                ))

        nonlocal children_map
        nonlocal number_of_root_nodes
        loss = 0.0
        for subject_uri in subjects_with_errors:
            subject_node = subject_hierarchy[subject_uri]
            ancestors = subject_ancestors_list(subject_node, subject_hierarchy)
            max_level_loss = 1.0
            for ancestor in ancestors:
                if ancestor.parent_uri is None:
                    max_level_loss *= 1.0 / number_of_root_nodes
                else:
                    number_of_siblings = len(children_map[ancestor.parent_uri])
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
