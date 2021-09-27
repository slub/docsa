"""Provides basic evaluation pipeline for multiple models."""

# pylint: disable=fixme, too-many-locals, too-many-arguments

import logging
from typing import Collection, Sequence

import numpy as np

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.model import Model
from slub_docsa.common.score import MultiClassScoreFunctionType, BinaryClassScoreFunctionType
from slub_docsa.evaluation.condition import check_subjects_have_minimum_samples
from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets
from slub_docsa.evaluation.split import cross_validation_split
from slub_docsa.models.oracle import OracleModel

logger = logging.getLogger(__name__)


def score_models_for_dataset(
    n_splits: int,
    dataset: Dataset,
    subject_order: Sequence[str],
    models: Collection[Model],
    overall_score_functions: Collection[MultiClassScoreFunctionType],
    per_class_score_functions: Collection[BinaryClassScoreFunctionType],
    random_state=0,
):
    """Evaluate a dataset for a number of models and score functions."""
    # check minimum requirements for cross-validation
    check_subjects_have_minimum_samples(dataset, n_splits)

    overall_score_matrix = np.empty((len(models), len(overall_score_functions), n_splits))
    overall_score_matrix[:, :, :] = np.NaN

    per_class_score_matrix = np.empty((len(models), len(per_class_score_functions), n_splits, len(subject_order)))
    per_class_score_matrix[:, :, :, :] = np.NaN

    for i, split in enumerate(cross_validation_split(n_splits, dataset, random_state=random_state)):
        logger.info("prepare %d-th cross validation split", i + 1)
        train_dataset, test_dataset = split
        train_incidence_matrix = subject_incidence_matrix_from_targets(train_dataset.subjects, subject_order)
        test_incidence_matrix = subject_incidence_matrix_from_targets(test_dataset.subjects, subject_order)
        logger.info(
            "evaluate %d-th cross validation split with %d training and %d test samples",
            i + 1,
            len(train_dataset.subjects),
            len(test_dataset.subjects)
        )

        for s_i, s_name in enumerate(subject_order):
            if len(np.where(test_incidence_matrix[:, s_i] > 0)[0]) == 0:
                logger.warning("no test case for subject %s in %d-th split", s_name, i+1)

        for j, model in enumerate(models):
            logger.info("evaluate model %s for %d-th split", str(model), i + 1)

            logger.info("do training")
            model.fit(train_dataset.documents, train_incidence_matrix)

            logger.info("do prediction")
            if isinstance(model, OracleModel):
                # provide predictions to oracle model
                model.set_test_targets(test_incidence_matrix)
            predicted_subject_probabilities = model.predict_proba(test_dataset.documents)

            logger.info("do global scoring")
            for k, score_function in enumerate(overall_score_functions):
                overall_score_matrix[j, k, i] = score_function(test_incidence_matrix, predicted_subject_probabilities)

            logger.info("do per-subject scoring")
            for s_i, s_name in enumerate(subject_order):
                per_class_test_incidence_matrix = test_incidence_matrix[:, [s_i]]
                per_class_predicted_subject_probabilities = \
                    predicted_subject_probabilities[:, [s_i]]

                # calculate score for subset of documents that are annotated with a subject
                for k, score_function in enumerate(per_class_score_functions):
                    per_class_score_matrix[j, k, i, s_i] = score_function(
                        per_class_test_incidence_matrix,
                        per_class_predicted_subject_probabilities
                    )

            logger.info("overall scores are: %s", str(overall_score_matrix[j, :, i]))

    return overall_score_matrix, per_class_score_matrix
