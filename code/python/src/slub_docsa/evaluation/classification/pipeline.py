"""Provides a basic evaluation pipeline for multiple models."""

# pylint: disable=fixme, too-many-locals, too-many-arguments

import logging
import time

from itertools import islice
from typing import Callable, Iterator, Optional, Sequence, Tuple

import numpy as np

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.document import Document
from slub_docsa.common.model import ClassificationModel
from slub_docsa.common.score import BatchedPerClassProbabilitiesScore, BatchedMultiClassProbabilitiesScore
from slub_docsa.evaluation.dataset.condition import check_dataset_subject_distribution
from slub_docsa.evaluation.dataset.condition import check_dataset_subjects_have_minimum_samples
from slub_docsa.evaluation.classification.incidence import subject_incidence_matrix_from_targets
from slub_docsa.evaluation.classification.split import DatasetSplitFunction
from slub_docsa.models.classification.dummy import OracleModel

logger = logging.getLogger(__name__)

SingleModelScores = Sequence[float]
SingleModelPerClassScores = Sequence[Sequence[float]]
MultiModelScores = Sequence[SingleModelScores]
MultiModelPerClassScores = Sequence[SingleModelPerClassScores]
MultiSplitScores = Sequence[MultiModelScores]
MultiSplitPerClassScores = Sequence[MultiModelPerClassScores]


def default_train_model(
    model: ClassificationModel,
    subject_order: Sequence[str],
    train_dataset: Dataset,
    validation_dataset: Optional[Dataset],
):
    logger.info("train model %s", str(model))
    # calcuate incidences
    train_incidence = subject_incidence_matrix_from_targets(train_dataset.subjects, subject_order)

    validation_documents = None
    validation_incidence = None
    if validation_dataset:
        validation_documents = validation_dataset.documents
        validation_incidence = subject_incidence_matrix_from_targets(validation_dataset.subjects, subject_order)

    model.fit(train_dataset.documents, train_incidence, validation_documents, validation_incidence)


def default_batch_predict_model(
    model: ClassificationModel,
    subject_order: Sequence[str],
    test_dataset: Dataset,
    batch_size: int = 100,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    test_document_generator = iter(test_dataset.documents)
    test_subjects_generator = iter(test_dataset.subjects)

    chunk_count = 0
    document_count = 0
    while True:
        logger.debug("evaluate chunk %d of test documents (%d so far)", chunk_count, document_count)

        loading_start = time.time()
        test_document_chunk = list(islice(test_document_generator, batch_size))
        test_subjects_chunk = list(islice(test_subjects_generator, batch_size))
        logger.debug("loading chunk %d took %d ms", chunk_count, (time.time() - loading_start) * 1000)

        if not test_document_chunk:
            break

        incidence_start = time.time()
        test_incidence_chunk = subject_incidence_matrix_from_targets(test_subjects_chunk, subject_order)
        logger.debug("incidence for chunk %d took %d ms", chunk_count, (time.time() - incidence_start) * 1000)

        if isinstance(model, OracleModel):
            # provide predictions to oracle model
            model.set_test_targets(test_incidence_chunk)

        prediction_start = time.time()
        predicted_probabilities_chunk = model.predict_proba(test_document_chunk)
        logger.debug("prediction of chunk %d took %d ms", chunk_count, (time.time() - prediction_start) * 1000)

        chunk_count += 1
        document_count += len(test_document_chunk)

        yield test_incidence_chunk, predicted_probabilities_chunk


def default_batch_evaluate_model(
    model: ClassificationModel,
    subject_order: Sequence[str],
    test_dataset: Dataset,
    score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]],
    per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]],
    batch_size: int = 100,
) -> Tuple[SingleModelScores, SingleModelPerClassScores]:
    # initialize new batched scoring functions
    batched_scores = [generator() for generator in score_generators]
    batched_per_class_scores = [generator() for generator in per_class_score_generators]

    chunk_count = 0
    test_chunk_generator = default_batch_predict_model(model, subject_order, test_dataset, batch_size)
    for test_incidence_chunk, predicted_probabilities_chunk in test_chunk_generator:
        scoring_start = time.time()
        for batched_score in batched_scores:
            batched_score.add_batch(test_incidence_chunk, predicted_probabilities_chunk)

        for batched_per_class_score in batched_per_class_scores:
            batched_per_class_score.add_batch(test_incidence_chunk, predicted_probabilities_chunk)
        logger.debug("scoring of chunk %d took %d ms", chunk_count, (time.time() - scoring_start) * 1000)
        chunk_count += 1

    logger.debug("calculate batched scores")
    scores = [batched_score() for batched_score in batched_scores]
    per_class_scores = [batched_per_class_score() for batched_per_class_score in batched_per_class_scores]

    return scores, per_class_scores


class TrainAndEvaluateModelFunction:

    def __call__(
        self,
        model: ClassificationModel,
        subject_order: Sequence[str],
        train_documents: Sequence[Document],
        train_incidence_matrix: np.ndarray,
        test_dataset: Dataset,
        score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]],
        per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]],
        validation_documents: Optional[Sequence[Document]] = None,
        validation_incidence_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[SingleModelScores, SingleModelPerClassScores]:
        """Call fit and predict_proba method of a model in order to generated predictions.

        This is the default implementation of a fit and predict logic, which can be overwritten in order to, e.g., save
        predictions to a database, load predictions from a database, etc.

        Parameters
        ----------
        model: Model
            the initialized model that can be fitted and used for predictions afterwards
        train_documents: Sequence[Document]
            the sequence of training documents
        train_incidence_matrix: np.ndarray
            the subject incidence matrix for the training documents
        test_documents: Sequence[Document],
            the sequence of test documents
        validation_documents: Optional[Sequence[Document]] = None
            an optional sequence of validation documents
            (used by the torch ann model to generate evaluation scores during training)
        validation_incidence_matrix: Optional[np.ndarray] = None
            an optional subject incidence matrix for the validation documents
            (used by the torch ann model to generate evaluation scores during training)

        Returns
        -------
        numpy.ndarray
            the subject prediction probability matrix as returned by the `predict_proba` method of the model
        """
        raise NotImplementedError()


class DefaultTrainAndEvaluateFunction(TrainAndEvaluateModelFunction):

    def __init__(self, batch_size: float = 100):
        self.batch_size = batch_size

    def __call__(
        self,
        model: ClassificationModel,
        subject_order: Sequence[str],
        train_dataset: Dataset,
        test_dataset: Dataset,
        score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]],
        per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]],
        validation_dataset: Optional[Dataset] = None,
    ):
        default_train_model(model, subject_order, train_dataset, validation_dataset)
        return default_batch_evaluate_model(
            model, subject_order, test_dataset, score_generators, per_class_score_generators, self.batch_size
        )


def score_classification_models_for_dataset_with_splits(
    n_splits: int,
    split_function: DatasetSplitFunction,
    subject_order: Sequence[str],
    dataset: Dataset,
    model_generators: Sequence[Callable[[], ClassificationModel]],
    score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]],
    per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]],
    train_and_evaluate: Optional[TrainAndEvaluateModelFunction] = None,
    stop_after_evaluating_split: Optional[int] = None,
    use_test_data_as_validation_data: bool = False,
) -> Tuple[MultiSplitScores, MultiSplitPerClassScores]:
    """Evaluate a dataset using cross-validation for a number of models and score functions.

    Parameters
    ----------
    n_splits: int
        the number of splits of the cross-validation
    dataset: Dataset
        the full dataset that is then being split into training and test sets during evaluation
    subject_order: Sequence[str]
        a subject order list describing a unique order of subjects, which is used as index for incidence and
        probability matrices (column ordering)
    models: Sequence[Model]
        the list of models that are evaluated
    split_function: DatasetSplitFunction,
        the function that can be used to split the full dataset into `n_splits` different sets of training and test
        data sets, e.g., `slub_docsa.evaluation.split.scikit_base_folder_splitter`.
    overall_score_functions: Sequence[MultiClassScoreFunctionType]
        the list of score functions that are applied to the full probability matrices after predicting the full test
        data set
    per_class_score_functions: Sequence[BinaryClassScoreFunctionType],
        the list of score functions that are applied on a per subject basis (one vs. rest) in order to evaluate the
        prediction quality of every subject
    fit_and_predict: FitClassificationModelAndPredictCallable = fit_classification_model_and_predict_test_documents,
        a function that allows to overwrite the basic fit and predict logic, e.g., by caching model predictions
    stop_after_evaluating_split: int = None,
        a flag that allows to stop the evaluation early, but still return the full score matrices (e.g. in order to
        test parameters settings without calculting all `n_splits` splits, which can take a long time)
    use_test_data_as_validation_data: bool = False
        a flag that if true provides the test data as validation data to the model's fit method, e.g., in order to
        evaluate the fitting behaviour of ann algorithms over multiple epochs of training

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of both score matrices, the overall score matrix and the per-subject score matrix.
        The overall score matrix will have a shape of `(len(models), len(overall_score_functions), n_splits)` and
        contains all score values calculated for every model over the test data of every split.
        The per subject score matrix will have shape of
        `(len(models), len(per_class_score_functions), n_splits, len(subject_order))` and contains score values for
        every subject calculated for every model, score function and split.

    """
    logger.info("check minimum requirements for cross-validation")
    if not check_dataset_subjects_have_minimum_samples(dataset, n_splits):
        logger.warning("dataset contains subjects with insufficient number of samples")

    all_split_scores = []
    all_split_per_class_scores = []

    for i, split in enumerate(split_function(dataset)):
        logger.info("prepare %d-th cross validation split", i + 1)
        train_dataset, test_dataset = split

        check_dataset_subject_distribution(train_dataset, test_dataset, (0.5 / n_splits, 2.0 / n_splits))

        logger.info(
            "evaluate %d-th cross validation split with %d training and %d test samples",
            i + 1,
            len(train_dataset.subjects),
            len(test_dataset.subjects)
        )

        validation_dataset: Dataset = None
        if use_test_data_as_validation_data:
            validation_dataset = test_dataset

        scores, per_class_scores = score_classification_models_for_dataset(
            subject_order,
            train_dataset,
            test_dataset,
            validation_dataset,
            model_generators,
            score_generators,
            per_class_score_generators,
            train_and_evaluate,
        )

        all_split_scores.append(scores)
        all_split_per_class_scores.append(per_class_scores)

        if stop_after_evaluating_split is not None and i >= stop_after_evaluating_split:
            break

    return all_split_scores, all_split_per_class_scores


def score_classification_models_for_dataset(
    subject_order: Sequence[str],
    train_dataset: Dataset,
    test_dataset: Dataset,
    validation_dataset: Optional[Dataset],
    model_generators: Sequence[Callable[[], ClassificationModel]],
    score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]],
    per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]],
    train_and_evaluate: Optional[TrainAndEvaluateModelFunction] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    all_model_scores = []
    all_model_per_class_scores = []

    for model_generator in model_generators:
        scores, per_class_scores = score_classification_model_for_dataset(
            subject_order,
            train_dataset,
            test_dataset,
            validation_dataset,
            model_generator,
            score_generators,
            per_class_score_generators,
            train_and_evaluate,
        )
        all_model_scores.append(scores)
        all_model_per_class_scores.append(per_class_scores)

    return all_model_scores, all_model_per_class_scores


def score_classification_model_for_dataset(
    subject_order: Sequence[str],
    training_dataset: Dataset,
    test_dataset: Dataset,
    validation_dataset: Optional[Dataset],
    model_generator: Callable[[], ClassificationModel],
    score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]],
    per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]],
    train_and_evaluate: Optional[TrainAndEvaluateModelFunction] = None,
    batch_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    # load default training and evaluation methods
    if train_and_evaluate is None:
        train_and_evaluate = DefaultTrainAndEvaluateFunction(batch_size)

    # create new instance of model
    model = model_generator()

    # do training
    scores, per_class_scores = train_and_evaluate(
        model,
        subject_order,
        training_dataset,
        test_dataset,
        score_generators,
        per_class_score_generators,
        validation_dataset,
    )

    return scores, per_class_scores
