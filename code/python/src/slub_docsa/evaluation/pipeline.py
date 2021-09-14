"""Provides basic evaluation pipeline for multiple models."""

# pylint: disable=fixme, too-many-locals

import logging
from typing import Collection

import numpy as np

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.model import Model
from slub_docsa.common.score import ScoreFunctionType
from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets, unique_subject_order
from slub_docsa.evaluation.split import cross_validation_split
from slub_docsa.models.oracle import OracleModel

logger = logging.getLogger(__name__)


def evaluate_dataset(
    n_splits: int,
    dataset: Dataset,
    models: Collection[Model],
    score_functions: Collection[ScoreFunctionType],
    random_state=0,
):
    """Evaluate a dataset for a number of models and score functions."""
    score_matrix = np.empty((len(models), len(score_functions), n_splits))
    score_matrix[:, :, :] = np.NaN

    subject_order = unique_subject_order(dataset.subjects)

    for i, split in enumerate(cross_validation_split(n_splits, dataset, random_state=random_state)):
        logger.info("prepare %d-th cross validation split", i + 1)
        train_dataset, test_dataset = split
        train_incidence_matrix = subject_incidence_matrix_from_targets(train_dataset.subjects, subject_order)
        test_incidence_matrix = subject_incidence_matrix_from_targets(test_dataset.subjects, subject_order)

        for j, model in enumerate(models):
            logger.info("evaluate model %s for %d-th split", str(model), i + 1)

            logger.info("do training")
            model.fit(train_dataset.documents, train_incidence_matrix)

            logger.info("do prediction")
            if isinstance(model, OracleModel):
                # provide predictions to oracle model
                model.set_test_targets(test_incidence_matrix)
            predicted_subject_probabilities = model.predict_proba(test_dataset.documents)

            logger.info("do scoring")
            for k, score_function in enumerate(score_functions):
                score_matrix[j, k, i] = score_function(test_incidence_matrix, predicted_subject_probabilities)

            logger.info("scores are: %s", str(score_matrix[j, :, i]))

    return score_matrix
