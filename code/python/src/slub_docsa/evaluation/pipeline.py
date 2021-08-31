"""Provides basic evaluation pipeline for multiple models."""

# pylint: disable=fixme

import logging
from typing import Collection

import numpy as np

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.model import Model
from slub_docsa.common.score import ScoreFunctionType
from slub_docsa.evaluation.split import cross_validation_split

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

    for i, split in enumerate(cross_validation_split(n_splits, dataset, random_state=random_state)):
        train_dataset, test_dataset = split

        for j, model in enumerate(models):
            logger.info("train %s for %d-th split", str(model), i + 1)
            model.fit(train_dataset.documents, train_dataset.subjects)
            predicted_subjects = model.predict(test_dataset.documents)

            for k, score_function in enumerate(score_functions):
                score_matrix[j, k, i] = score_function(test_dataset.subjects, predicted_subjects)

    return score_matrix
