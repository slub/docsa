"""Pipeline for evaluating clustering models."""

# pylint: disable=too-many-arguments, too-many-locals

import logging

from typing import Sequence, Optional

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectTargets
from slub_docsa.common.model import ClusteringModel
from slub_docsa.common.score import ClusteringScore

logger = logging.getLogger(__name__)


def score_clustering_models_for_documents(
    documents: Sequence[Document],
    subject_targets: Optional[SubjectTargets],
    models: Sequence[ClusteringModel],
    scores: Sequence[ClusteringScore],
    repeats: int = 10,
    max_documents: Optional[int] = None,
) -> np.ndarray:
    """Evaluate clustering models by fitting, predicting and scoring them on a single dataset.

    Parameters
    ----------
    documents: Sequence[Document]
        the list of documents that is being clustered
    subject_targets: Optional[SubjectTargets]
        an optional list of subject targets, which may be used to evaluate and score the resulting clusterings against
        known subject annotations
    models: Sequence[ClusteringModel]
        the sequence of models being evaluated
    scores: Sequence[ClusteringScoreFunction]
        the sequence of scores being calculated for each model
    repeats: int = 10
        how often each clustering model is fitted in order to analyze the variance of clustering scores
    max_documents: int = None
        if set to a number, a random of selection of documents is used instead of all documents; the random selection
        is repeated for each iteration

    Returns
    -------
    numpy.ndarray
        a score matrix of shape `(len(models), len(scores), repeats)`, which contains every score for every evaluated
        clustering model
    """
    score_matrix = np.empty((len(models), len(scores), repeats))
    score_matrix[:, :, :] = np.NaN

    for i in range(repeats):

        sampled_documents = documents
        sampled_subject_targets = subject_targets

        # choose max_documents many random documents for clustering
        if max_documents is not None:
            sampled_idx = np.random.choice(range(len(documents)), size=max_documents, replace=False)
            sampled_documents = [documents[i] for i in sampled_idx]
            if subject_targets is not None:
                sampled_subject_targets = [subject_targets[i] for i in sampled_idx]

        for j, model in enumerate(models):
            logger.info("fit clustering model %s for repetition %d", str(model), i + 1)
            model.fit(sampled_documents)

            logger.info("predict clustering model %s for repetition %d", str(model), i + 1)
            membership = model.predict(sampled_documents)

            logger.info("score clustering result from model %s for repetition %d", str(model), i + 1)
            for k, score_function in enumerate(scores):
                score = score_function(sampled_documents, membership, sampled_subject_targets)
                score_matrix[j, k, i] = score

    return score_matrix
