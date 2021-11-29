"""Common scores that are used to evaluate results of an experiment."""

from typing import Callable, Iterable, List, NamedTuple, Optional, Sequence, Tuple, cast

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.metrics import mean_squared_error

from slub_docsa.common.score import MultiClassScoreFunctionType, BinaryClassScoreFunctionType
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.evaluation.incidence import threshold_incidence_decision, positive_top_k_incidence_decision
from slub_docsa.evaluation.score import cesa_bianchi_h_loss, scikit_incidence_metric
from slub_docsa.evaluation.score import scikit_metric_for_best_threshold_based_on_f1score


class DefaultScoreLists(NamedTuple):
    """Stores names, ranges and functions of default scores (both multi-class and binary)."""

    names: List[str]
    ranges: List[Tuple[Optional[float], Optional[float]]]
    functions: List[Callable]


def default_named_multiclass_scores(
    subject_order: Sequence[str] = None,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType] = None,
    score_name_subset: Iterable[str] = None
) -> DefaultScoreLists:
    """Return a list of default score functions for evaluation."""
    scores = [
        # ("t=0.5 accuracy", [0, 1], scikit_incidence_metric(
        #     threshold_incidence_decision(0.5),
        #     accuracy_score
        # )),
        # ("top3 accuracy", [0, 1], scikit_incidence_metric(
        #     positive_top_k_incidence_decision(3),
        #     accuracy_score
        # )),
        ("t=best f1_score micro", [0, 1], scikit_metric_for_best_threshold_based_on_f1score(
            f1_score,
            average="micro",
            zero_division=0
        )),
        ("top3 f1_score micro", [0, 1], scikit_incidence_metric(
            positive_top_k_incidence_decision(3),
            f1_score,
            average="micro",
            zero_division=0
        )),
        ("t=best precision micro", [0, 1], scikit_metric_for_best_threshold_based_on_f1score(
            precision_score,
            average="micro",
            zero_division=0
        )),
        ("top3 precision micro", [0, 1], scikit_incidence_metric(
            positive_top_k_incidence_decision(3),
            precision_score,
            average="micro",
            zero_division=0
        )),
        ("t=best recall micro", [0, 1], scikit_metric_for_best_threshold_based_on_f1score(
            recall_score,
            average="micro",
            zero_division=0
        )),
        ("top3 recall micro", [0, 1], scikit_incidence_metric(
            positive_top_k_incidence_decision(3),
            recall_score,
            average="micro",
            zero_division=0
        )),
        ("t=best h_loss", [0, 1], scikit_metric_for_best_threshold_based_on_f1score(
            cesa_bianchi_h_loss(subject_hierarchy, subject_order, log_factor=1000),
        )),
        ("top3 h_loss", [0, 1], scikit_incidence_metric(
            positive_top_k_incidence_decision(3),
            cesa_bianchi_h_loss(subject_hierarchy, subject_order, log_factor=1000),
        )),
        ("roc auc micro", [0, 1], lambda t, p: roc_auc_score(t, p, average="micro")),
        ("log loss", [0, None], log_loss),
        # ("mean squared error", [0, None], mean_squared_error)
    ]

    if score_name_subset is not None:
        scores = list(filter(lambda i: i[0] in score_name_subset, scores))

    score_names, score_ranges, score_functions = list(zip(*scores))
    score_names = cast(List[str], score_names)
    score_ranges = cast(List[Tuple[Optional[float], Optional[float]]], score_ranges)
    score_functions = cast(List[MultiClassScoreFunctionType], score_functions)

    return DefaultScoreLists(score_names, score_ranges, score_functions)


def default_named_binary_classification_scores(
    score_name_subset: Iterable[str] = None
) -> DefaultScoreLists:
    """Return a list of default per-subject score functions for evaluation."""
    scores = [
        ("t=0.5 accuracy", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            accuracy_score
        )),
        ("t=0.5 f1_score", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            f1_score,
            zero_division=0
        )),
        ("t=0.5 precision", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            precision_score,
            zero_division=0
        )),
        ("t=0.5 recall", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            recall_score,
            zero_division=0
        )),
        ("mean squared error", [0, None], mean_squared_error),
        ("# test samples", [0, None], lambda t, _: len(np.where(t > 0)[0]))
    ]

    if score_name_subset is not None:
        scores = list(filter(lambda i: i[0] in score_name_subset, scores))

    score_names, score_ranges, score_functions = list(zip(*scores))
    score_names = cast(List[str], score_names)
    score_ranges = cast(List[Tuple[Optional[float], Optional[float]]], score_ranges)
    score_functions = cast(List[BinaryClassScoreFunctionType], score_functions)

    return DefaultScoreLists(score_names, score_ranges, score_functions)
