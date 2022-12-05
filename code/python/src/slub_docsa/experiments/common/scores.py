"""Common scores that are used to evaluate results of an experiment."""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics import homogeneity_score, completeness_score
from scipy.spatial.distance import cosine

from slub_docsa.common.score import BatchedMultiClassProbabilitiesScore, BatchedPerClassProbabilitiesScore
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer
from slub_docsa.evaluation.classification.incidence import positive_top_k_incidence_decision
from slub_docsa.evaluation.classification.incidence import threshold_incidence_decision
# from slub_docsa.evaluation.classification.score.hierarchical import cesa_bianchi_h_loss
from slub_docsa.evaluation.classification.score.batched import BatchedIncidenceDecisionConfusionScore, BatchedF1Score
from slub_docsa.evaluation.classification.score.batched import BatchedIncidenceDecisionPerClassConfusionScore
from slub_docsa.evaluation.classification.score.batched import BatchedNumberOfTestExamplesPerClass
from slub_docsa.evaluation.classification.score.batched import BatchedPerClassF1Score, BatchedPerClassPrecisionScore
from slub_docsa.evaluation.classification.score.batched import BatchedPerClassRecallScore
from slub_docsa.evaluation.classification.score.batched import BatchedBestThresholdScore, BatchedPrecisionScore
from slub_docsa.evaluation.classification.score.batched import BatchedRecallScore
from slub_docsa.evaluation.clustering.score import clustering_membership_score_function
from slub_docsa.evaluation.clustering.score import scikit_clustering_label_score_function
from slub_docsa.evaluation.clustering.similarity import indexed_document_distance_generator_from_vectorizer
from slub_docsa.evaluation.clustering.similarity import intra_cluster_distance

T = TypeVar("T")

ScoreTupleList = List[Tuple[
    str,
    Tuple[Optional[float], Optional[float]],
    Callable[[], T]
]]


@dataclass
class NamedScoreLists(Generic[T]):
    """Stores names, ranges and functions of default scores (both multi-class and binary)."""

    names: List[str]
    ranges: List[Tuple[Optional[float], Optional[float]]]
    generators: List[Callable[[], T]]


def initialize_named_score_tuple_list(
    score_list: ScoreTupleList[T],
    name_subset: Optional[Iterable[str]] = None,
) -> NamedScoreLists[T]:
    """Covert a score tuple list to a named score list object."""
    if name_subset is not None:
        score_list = list(filter(lambda i: i[0] in name_subset, score_list))

    score_names = [i[0] for i in score_list]
    score_ranges = [i[1] for i in score_list]
    score_generators = [i[2] for i in score_list]

    return NamedScoreLists(score_names, score_ranges, score_generators)


def default_named_score_list(
    subject_order: Optional[Sequence[str]] = None,
    subject_hierarchy: Optional[SubjectHierarchy] = None,
) -> ScoreTupleList[BatchedMultiClassProbabilitiesScore]:
    """Return a list of default score functions for evaluation."""
    scores = [
        ("t=best f1 micro", [0, 1], lambda: BatchedBestThresholdScore(
            score_generator=BatchedF1Score
        )),
        ("top3 f1 micro", [0, 1], lambda: BatchedIncidenceDecisionConfusionScore(
            incidence_decision=positive_top_k_incidence_decision(3),
            confusion_score=BatchedF1Score()
        )),
        ("t=best precision micro", [0, 1], lambda: BatchedBestThresholdScore(
            score_generator=BatchedPrecisionScore
        )),
        ("top3 precision micro", [0, 1], lambda: BatchedIncidenceDecisionConfusionScore(
            incidence_decision=positive_top_k_incidence_decision(3),
            confusion_score=BatchedPrecisionScore()
        )),
        ("t=best recall micro", [0, 1], lambda: BatchedBestThresholdScore(
            score_generator=BatchedRecallScore
        )),
        ("top3 recall micro", [0, 1], lambda: BatchedIncidenceDecisionConfusionScore(
            incidence_decision=positive_top_k_incidence_decision(3),
            confusion_score=BatchedRecallScore()
        )),
        # ("t=0.5 accuracy", [0, 1], scikit_incidence_metric(
        #     threshold_incidence_decision(0.5),
        #     accuracy_score
        # )),
        # ("top3 accuracy", [0, 1], scikit_incidence_metric(
        #     positive_top_k_incidence_decision(3),
        #     accuracy_score
        # )),
        # ("t=best h_loss", [0, 1], scikit_metric_for_best_threshold_based_on_f1score(
        #     cesa_bianchi_h_loss(subject_hierarchy, subject_order, log_factor=1000),
        # )),
        # ("top3 h_loss", [0, 1], scikit_incidence_metric(
        #     positive_top_k_incidence_decision(3),
        #     cesa_bianchi_h_loss(subject_hierarchy, subject_order, log_factor=1000),
        # )),
        # ("roc auc micro", [0, 1], lambda t, p: roc_auc_score(t, p, average="micro")),
        # ("log loss", [0, None], log_loss),
        # ("mean squared error", [0, None], mean_squared_error)
    ]
    return scores


def default_named_per_class_score_list() -> ScoreTupleList[BatchedPerClassProbabilitiesScore]:
    """Return a list of default per-subject score functions for evaluation."""
    scores: ScoreTupleList[BatchedPerClassProbabilitiesScore] = [
        ("t=0.5 f1", (0, 1), lambda: BatchedIncidenceDecisionPerClassConfusionScore(
            incidence_decision=threshold_incidence_decision(0.5),
            confusion_score=BatchedPerClassF1Score(),
        )),
        ("t=0.5 precision", (0, 1), lambda: BatchedIncidenceDecisionPerClassConfusionScore(
            incidence_decision=threshold_incidence_decision(0.5),
            confusion_score=BatchedPerClassPrecisionScore(),
        )),
        ("t=0.5 recall", (0, 1), lambda: BatchedIncidenceDecisionPerClassConfusionScore(
            incidence_decision=threshold_incidence_decision(0.5),
            confusion_score=BatchedPerClassRecallScore(),
        )),
        ("# test samples", (0, None), BatchedNumberOfTestExamplesPerClass),
        # ("t=0.5 accuracy", (0, 1), scikit_incidence_metric(
        #     threshold_incidence_decision(0.5),
        #     accuracy_score
        # )),
        # ("mean squared error", (0, None), mean_squared_error),
    ]
    return scores


def default_named_clustering_score_list(
    vectorizer: AbstractVectorizer
) -> Any:
    """Return named clustering scoring functions."""
    scores: Any = [
        # ("mutual info", (None, None), scikit_clustering_label_score_function(mutual_info_score)),
        ("mutual info", (0, None), scikit_clustering_label_score_function(adjusted_mutual_info_score)),
        # ("rand", (None, None), scikit_clustering_label_score_function(rand_score)),
        ("rand", (0, None), scikit_clustering_label_score_function(adjusted_rand_score)),
        ("homogeneity", (0, 1), scikit_clustering_label_score_function(homogeneity_score)),
        ("completeness", (0, 1), scikit_clustering_label_score_function(completeness_score)),
        ("intra cluster tfidf cosine", (0, None), clustering_membership_score_function(
            indexed_document_distance_generator_from_vectorizer(vectorizer, cosine),  # type: ignore
            intra_cluster_distance
        )),
    ]

    return scores
