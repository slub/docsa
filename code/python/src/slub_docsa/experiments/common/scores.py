"""Common scores that are used to evaluate results of an experiment."""

from typing import Any, Callable, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics import homogeneity_score, completeness_score
from scipy.spatial.distance import cosine

from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer
from slub_docsa.evaluation.incidence import threshold_incidence_decision, positive_top_k_incidence_decision
from slub_docsa.evaluation.score import cesa_bianchi_h_loss, scikit_incidence_metric
from slub_docsa.evaluation.score import scikit_metric_for_best_threshold_based_on_f1score
from slub_docsa.evaluation.score import clustering_membership_score_function, scikit_clustering_label_score_function
from slub_docsa.evaluation.similarity import indexed_document_distance_generator_from_vectorizer, intra_cluster_distance

ScoreTupleList = List[Tuple[str, Tuple[Optional[float], Optional[float]], Any]]


class NamedScoreLists(NamedTuple):
    """Stores names, ranges and functions of default scores (both multi-class and binary)."""

    names: List[str]
    ranges: List[Tuple[Optional[float], Optional[float]]]
    functions: List[Callable]


def initialize_named_score_tuple_list(
    score_list: ScoreTupleList,
    name_subset: Optional[Iterable[str]] = None,
) -> NamedScoreLists:
    """Covert a score tuple list to a named score list object."""
    if name_subset is not None:
        score_list = list(filter(lambda i: i[0] in name_subset, score_list))

    score_names = [i[0] for i in score_list]
    score_ranges = [i[1] for i in score_list]
    score_functions = [i[2] for i in score_list]

    return NamedScoreLists(score_names, score_ranges, score_functions)


def default_named_multiclass_score_list(
    subject_order: Sequence[str] = None,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType] = None,
) -> ScoreTupleList:
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

    return scores


def default_named_binary_class_score_list() -> ScoreTupleList:
    """Return a list of default per-subject score functions for evaluation."""
    scores: ScoreTupleList = [
        ("t=0.5 accuracy", (0, 1), scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            accuracy_score
        )),
        ("t=0.5 f1_score", (0, 1), scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            f1_score,
            zero_division=0
        )),
        ("t=0.5 precision", (0, 1), scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            precision_score,
            zero_division=0
        )),
        ("t=0.5 recall", (0, 1), scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            recall_score,
            zero_division=0
        )),
        ("mean squared error", (0, None), mean_squared_error),
        ("# test samples", (0, None), lambda t, _: len(np.where(t > 0)[0]))
    ]
    return scores


def default_named_clustering_score_list(
    vectorizer: AbstractVectorizer
) -> ScoreTupleList:
    """Return named clustering scoring functions."""
    scores: ScoreTupleList = [
        # ("mutual info", (None, None), scikit_clustering_label_score_function(mutual_info_score)),
        ("mutual info", (0, None), scikit_clustering_label_score_function(adjusted_mutual_info_score)),
        # ("rand", (None, None), scikit_clustering_label_score_function(rand_score)),
        ("rand", (0, None), scikit_clustering_label_score_function(adjusted_rand_score)),
        ("homogeneity", (0, 1), scikit_clustering_label_score_function(homogeneity_score)),
        ("completeness", (0, 1), scikit_clustering_label_score_function(completeness_score)),
        ("intra cluster tfidf cosine", (0, None), clustering_membership_score_function(
            indexed_document_distance_generator_from_vectorizer(vectorizer, cosine),
            intra_cluster_distance
        )),
    ]

    return scores
