"""Defines various scores for evaluating clustering results."""

# pylint: disable=too-many-locals, unused-argument

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
from slub_docsa.common.document import Document

from slub_docsa.common.score import ClusteringScore
from slub_docsa.common.similarity import IndexedDocumentDistanceFunction, IndexedDocumentDistanceGenerator
from slub_docsa.common.subject import SubjectTargets
from slub_docsa.evaluation.classification.incidence import is_crisp_cluster_membership
from slub_docsa.evaluation.classification.incidence import membership_matrix_to_crisp_cluster_assignments
from slub_docsa.evaluation.classification.incidence import unique_subject_order

logger = logging.getLogger(__name__)


def scikit_clustering_label_score_function(
    scikit_score: Any,
    **kwargs: Any,
) -> ClusteringScore:
    """Return function that scores a clustering by comparing it to the true subject targets using scikit.

    Possible scikit scores are:
    - mutual information: `sklearn.metrics.mutual_info_score`, `sklearn.metrics.adjusted_mutual_info_score`
    - homegeneity and completeness: `sklearn.metrics.homogeneity_score`, `sklearn.metrics.completeness_score`
    - other scikit scores that take `labels_true` and `labels_pred` as arguments

    Parameters
    ----------
    scikit_score: Any
        the scikit label score function
    kwargs: Any
        additional arguments to the scikit score function

    Returns
    -------
    ClusteringScoreFunction
        a function that can be used to score clusterings
    """

    def _score(
        documents: Sequence[Document],
        membership: np.ndarray,
        subject_targets: Optional[SubjectTargets] = None,
    ) -> float:

        if not is_crisp_cluster_membership(membership):
            # can not calculate clustering scores for non-crisp clusterings
            return np.NaN

        if subject_targets is None:
            # can not calculate clustering label score without true subject targets
            return np.NaN

        # get cluster assignment list from membership matrix
        pred_labels = membership_matrix_to_crisp_cluster_assignments(membership)

        # get true assignments
        subject_order = unique_subject_order(subject_targets)
        true_labels = [subject_order.index(next(iter(subject_list))) for subject_list in subject_targets]

        return scikit_score(true_labels, pred_labels, **kwargs)

    return _score


def clustering_membership_score_function(
    indexed_distance_generator: IndexedDocumentDistanceGenerator,
    membership_score_function: Callable[[IndexedDocumentDistanceFunction, np.ndarray], float],
) -> ClusteringScore:
    """Return a function that can be used to score clusterings based on membership degrees and document distances.

    Parameters
    ---------
    document_distance: DocumentDistanceFunction
        a distance function between documents
    membership_score_function: Callable[[IndexedDocumentDistanceFunction, np.ndarray], float]
        a function that scores memberships based on distances between documents

    Returns
    -------
    ClusteringScoreFunction
        a function for scoring clusterings
    """

    def _score(
        documents: Sequence[Document],
        membership: np.ndarray,
        subject_targets: Optional[SubjectTargets] = None,
    ) -> float:
        indexed_distance = indexed_distance_generator(documents, membership, subject_targets)
        return membership_score_function(indexed_distance, membership)

    return _score
