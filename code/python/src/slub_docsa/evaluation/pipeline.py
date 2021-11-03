"""Provides basic evaluation pipeline for multiple models."""

# pylint: disable=fixme, too-many-locals, too-many-arguments

import logging
from typing import Callable, Sequence

import numpy as np

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.document import Document
from slub_docsa.common.model import Model
from slub_docsa.common.score import MultiClassScoreFunctionType, BinaryClassScoreFunctionType
from slub_docsa.evaluation.condition import check_dataset_subject_distribution
from slub_docsa.evaluation.condition import check_dataset_subjects_have_minimum_samples
from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets
from slub_docsa.evaluation.split import DatasetSplitFunction
from slub_docsa.models.dummy import OracleModel

logger = logging.getLogger(__name__)

FitModelAndPredictCallable = Callable[[Model, Sequence[Document], np.ndarray, Sequence[Document]], np.ndarray]


def fit_model_and_predict_test_documents(
    model: Model,
    train_documents: Sequence[Document],
    train_incidence_matrix: np.ndarray,
    test_documents: Sequence[Document],
) -> np.ndarray:
    """Call fit and predict_proba method of model in order to generated predictions."""
    logger.info("do training")
    model.fit(train_documents, train_incidence_matrix)

    logger.info("do prediction")
    return model.predict_proba(test_documents)


def score_models_for_dataset(
    n_splits: int,
    dataset: Dataset,
    subject_order: Sequence[str],
    models: Sequence[Model],
    split_function: DatasetSplitFunction,
    overall_score_functions: Sequence[MultiClassScoreFunctionType],
    per_class_score_functions: Sequence[BinaryClassScoreFunctionType],
    fit_model_and_predict: FitModelAndPredictCallable = fit_model_and_predict_test_documents,
    stop_after_evaluating_split: int = None,
):
    """Evaluate a dataset for a number of models and score functions."""
    # check minimum requirements for cross-validation
    if not check_dataset_subjects_have_minimum_samples(dataset, n_splits):
        raise ValueError("dataset contains subjects with insufficient number of samples")

    overall_score_matrix = np.empty((len(models), len(overall_score_functions), n_splits))
    overall_score_matrix[:, :, :] = np.NaN

    per_class_score_matrix = np.empty((len(models), len(per_class_score_functions), n_splits, len(subject_order)))
    per_class_score_matrix[:, :, :, :] = np.NaN

    for i, split in enumerate(split_function(dataset)):
        logger.info("prepare %d-th cross validation split", i + 1)
        train_dataset, test_dataset = split
        check_dataset_subject_distribution(train_dataset, test_dataset, (0.5 / n_splits, 2.0 / n_splits))
        train_incidence_matrix = subject_incidence_matrix_from_targets(train_dataset.subjects, subject_order)
        test_incidence_matrix = subject_incidence_matrix_from_targets(test_dataset.subjects, subject_order)
        logger.info(
            "evaluate %d-th cross validation split with %d training and %d test samples",
            i + 1,
            len(train_dataset.subjects),
            len(test_dataset.subjects)
        )

        for j, model in enumerate(models):
            logger.info("evaluate model %s for %d-th split", str(model), i + 1)

            if isinstance(model, OracleModel):
                # provide predictions to oracle model
                model.set_test_targets(test_incidence_matrix)

            # do predictions
            predicted_subject_probabilities = fit_model_and_predict(
                model, train_dataset.documents, train_incidence_matrix, test_dataset.documents
            )

            logger.info("do global scoring")
            for k, score_function in enumerate(overall_score_functions):
                overall_score_matrix[j, k, i] = score_function(test_incidence_matrix, predicted_subject_probabilities)

            logger.info("do per-subject scoring")
            for s_i in range(len(subject_order)):
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

        if stop_after_evaluating_split is not None and i >= stop_after_evaluating_split:
            break

    return overall_score_matrix, per_class_score_matrix
