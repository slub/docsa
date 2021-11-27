"""Various similarity and distance functions."""

# pylint: disable=unused-argument

import itertools
import logging

from typing import Callable, Dict, Optional, Sequence, Tuple, cast

import numpy as np
from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectTargets
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer
from slub_docsa.common.similarity import IndexedDocumentDistanceFunction, IndexedDocumentDistanceGenerator

logger = logging.getLogger(__name__)


def cached_indexed_document_distance(
    indexed_distance: IndexedDocumentDistanceFunction,
) -> IndexedDocumentDistanceFunction:
    """Return a function that wraps a document distance function for caching.

    Parameters
    ----------
    indexed_document_distance: IndexedDocumentDistanceFunction
        the distance function between documents to be cached

    Returns
    -------
    IndexedDocumentDistanceFunction
        a cached version of the indexed document distance function
    """
    cache: Dict[Tuple[int, int], float] = {}

    def cached_distance_function(d1_idx, d2_idx) -> float:
        key = (d1_idx, d2_idx)
        if key not in cache:
            cache[key] = indexed_distance(d1_idx, d2_idx)
        return cache[key]

    return cached_distance_function


def intra_cluster_distance(
    distance_function: IndexedDocumentDistanceFunction,
    membership: np.ndarray,
    max_documents_per_cluster: int = 500,
) -> float:
    """Calculate the intra cluster distance given a membership matrix and distance function.

    Parameters
    ----------
    distance_function: Callable[[int, int], float]
        a function that returns the distance between document `i` and `j`
    membership: numpy.ndarray
        the membership matrix of shape `(len(documents), n_clusters)`
    max_documents_per_cluster: int = 500
        the maximum number of documents to calculate pairwise distances from within a cluster; if a cluster consists
        of more documents, the cluster is randomly sampled for that amount of documents

    Returns
    -------
    float
        the intra cluster distance by averaging all distances of pairwise elements within a cluster weighted by their
        membership degree
    """
    n_clusters = membership.shape[1]
    intra_distances = []
    for cluster_idx in range(n_clusters):
        # get idxs of documents belonging to current cluster
        document_idxs = np.where(membership[:, cluster_idx] > 0.0)[0]
        logger.debug("intra cluster distance for cluster %d containing %d documents", cluster_idx, len(document_idxs))
        if max_documents_per_cluster is not None and len(document_idxs) > max_documents_per_cluster:
            document_idxs = np.random.choice(document_idxs, size=max_documents_per_cluster)
        if len(document_idxs) > 1:
            intra_distance = 0
            n_combinations = 0
            for d1_idx, d2_idx in itertools.combinations(document_idxs, 2):
                factor = membership[d1_idx, cluster_idx] * membership[d2_idx, cluster_idx]
                distance = distance_function(d1_idx, d2_idx)
                # logger.debug("distance is %f", distance)
                intra_distance += factor * distance
                n_combinations += 1
            intra_distance = intra_distance / n_combinations
            intra_distances.append(intra_distance)

    return cast(float, np.average(intra_distances))


def indexed_document_distance_generator_from_vectorizer(
    vectorizer: AbstractVectorizer,
    vector_distance: Callable[[np.ndarray, np.ndarray], float],
) -> IndexedDocumentDistanceGenerator:
    """Return a generator for indexed document distance functions based on a vectorizer and a vector distance.

    Parameters
    ----------
    vectorizer: AbstractVectorizer
        the vectorizer that is used to transform documents to a vector space representation
    vector_distance: Callable[[np.ndarray, np.ndarray], float]
        a distance function for numpy vectors

    Returns
    -------
    IndexedDocumentDistanceGenerator
        a function that returns an indexed document distance function when providing it with the current document set
        that is used for vectorization
    """

    def _generator_function(
        documents: Sequence[Document],
        membership: np.ndarray,
        subject_targets: Optional[SubjectTargets],
    ) -> IndexedDocumentDistanceFunction:
        corpus = [document_as_concatenated_string(d) for d in documents]
        vectorizer.fit(iter(corpus))
        vectors = np.array(list(vectorizer.transform(iter(corpus))))

        def _distance(idx1: int, idx2: int) -> float:
            return vector_distance(vectors[idx1], vectors[idx2])

        return cached_indexed_document_distance(_distance)

    return _generator_function
