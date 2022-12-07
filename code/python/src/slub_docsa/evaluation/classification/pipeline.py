"""Basic evaluation pipelines for a single or multiple classification models, including cross-validation."""

# pylint: disable=fixme, too-many-locals, too-many-arguments, too-few-public-methods

import logging
import time

from itertools import islice
from typing import Callable, Iterator, Optional, Sequence, Tuple

import numpy as np

from slub_docsa.common.dataset import Dataset
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
    """Apply datasets to the fit method of the model, meaning the default training method for a model.

    Parameters
    ----------
    model : ClassificationModel
        the model to be trained
    subject_order : Sequence[str]
        the subject order required to determine target incidences
    train_dataset : Dataset
        a training dataset provided to the fit method
    validation_dataset : Optional[Dataset]
        optional validation dataset provided to the fit method
    """
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
    """Predict test data for a model in batches, meaning the default prediciton strategy.

    Parameters
    ----------
    model : ClassificationModel
        the model to be used for prediction
    subject_order : Sequence[str]
        the subject order used to build target incidences
    test_dataset : Dataset
        the test dataset that is supposed to be predicted
    batch_size : int, optional
        the batch size, by default 100

    Yields
    ------
    Tuple[np.ndarray, np.ndarray]
        a tuple of (test_incidence_chunk, predicted_probabilities_chunk) that can be evalated further
    """
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
    """Evaluate a model given a test dataset and various score functions in batches.

    Parameters
    ----------
    model : ClassificationModel
        the model to be evaluated
    subject_order : Sequence[str]
        the subject order that is used to generate target incidence matrices
    test_dataset : Dataset
        the test dataset that is evaluated and scored
    score_generators : Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]]
        generators for batched score functions that measure the overall classification performance
    per_class_score_generators : Sequence[Callable[[], BatchedPerClassProbabilitiesScore]]
        generators for batched score functions that measure the classification performance for each subject
    batch_size : int, optional
        the batch size, by default 100

    Returns
    -------
    Tuple[SingleModelScores, SingleModelPerClassScores]
        both the overall score results, and the score matrices when measuring the classification performance
        for each subject
    """
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

    logger.info("scores for model %s are %s", str(model), scores)
    return scores, per_class_scores


class TrainAndEvaluateModelFunction:
    """An abstract interface for training and evaluating a classification model."""

    def __call__(
        self,
        model: ClassificationModel,
        subject_order: Sequence[str],
        train_dataset: Dataset,
        test_dataset: Dataset,
        score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]],
        per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]],
        validation_dataset: Dataset = None,
    ) -> Tuple[SingleModelScores, SingleModelPerClassScores]:
        """Evaluate a model by training a model and scoring its classification performance via test data.

        Parameters
        ----------
        model: Model
            a freshly initialized model that can be fitted and used for predictions afterwards
        subject_order : Sequence[str]
            the subject order that is used to generate incidence matrices
        train_dataset: Sequence[Document]
            the training dataset used to fit the model
        test_dataset: Sequence[Document],
            the test dataset used to score the classification performance of the model
        score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]]
            generators for batched score functions that measure the overall classification performance
        per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]]
            generators for batched score functions that measure the classification performance for each subject
        validation_dataset : Optional[Dataset]
            optional validation dataset provided to the fit method

        Returns
        -------
        Tuple[SingleModelScores, SingleModelPerClassScores]
            both the overall score results, and the score matrices when measuring the classification performance
            for each subject
        """
        raise NotImplementedError()


class DefaultTrainAndEvaluateFunction(TrainAndEvaluateModelFunction):
    """Default training and evaluation strategy.

    This is the default implementation of a fit and predict logic, which can be overwritten in order to, e.g., save
    predictions to a database, load predictions from a database, etc.
    """

    def __init__(self, batch_size: float = 100):
        """Initialize the default train and evaluation strategy with a batch size.

        Parameters
        ----------
        batch_size : float, optional
            the batch size at which the model is evaluated and scored, by default 100
        """
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
    ) -> Tuple[SingleModelScores, SingleModelPerClassScores]:
        """Return scores after applying the default training and evaluation strategy.

        Parameters
        ----------
        model: Model
            a freshly initialized model that can be fitted and used for predictions afterwards
        subject_order : Sequence[str]
            the subject order that is used to generate incidence matrices
        train_dataset: Dataset
            the training dataset used to fit the model
        test_dataset: Dataset
            the test dataset used to score the classification performance of the model
        score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]]
            generators for batched score functions that measure the overall classification performance
        per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]]
            generators for batched score functions that measure the classification performance for each subject
        validation_dataset : Optional[Dataset]
            optional validation dataset provided to the fit method

        Returns
        -------
        Tuple[SingleModelScores, SingleModelPerClassScores]
            both the overall score results, and the score matrices when measuring the classification performance
            for each subject
        """
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
    batch_size: int = 100,
) -> Tuple[MultiSplitScores, MultiSplitPerClassScores]:
    """Evaluate a dataset using cross-validation for a number of models and score functions.

    Parameters
    ----------
    n_splits: int
        the number of splits of the cross-validation
    split_function: DatasetSplitFunction,
        the function that can be used to split the full dataset into `n_splits` different sets of training and test
        data sets, e.g., `slub_docsa.evaluation.split.scikit_base_folder_splitter`.
    subject_order: Sequence[str]
        a subject order list describing a unique order of subjects, which is used as index for incidence and
        probability matrices (column ordering)
    dataset: Dataset
        the full dataset that is then being split into training and test sets during evaluation
    model_generators: Sequence[Callable[[], ClassificationModel]]
        the list of generators for models that will be instanciated, trained and evaluated one after another
    score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]]
        generators for batched score functions that measure the overall classification performance
    per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]]
        generators for batched score functions that measure the classification performance for each subject
    train_and_evaluate: TrainAndEvaluateModelFunction = None
        training and evaluation strategy, if None, the default strategy `DefaultTrainAndEvaluateFunction` is used
    stop_after_evaluating_split: int = None,
        a flag that allows to stop the evaluation early, but still return the full score matrices (e.g. in order to
        test parameters settings without calculting all `n_splits` splits, which can take a long time)
    use_test_data_as_validation_data: bool = False
        a flag that if true provides the test data as validation data to the model's fit method, e.g., in order to
        evaluate the fitting behaviour of ann algorithms over multiple epochs of training
    batch_size : int, optional
        the batch size at which the model is evaluated and scored, by default 100; only applies to the default
        training and evaluation strategy

    Returns
    -------
    Tuple[MultiSplitScores, MultiSplitPerClassScores]
        A tuple of both the overall scores and per-class scores. After conversion to a numpy array, the overall score
        matrix will have a shape of `(n_splits, len(model_generators), len(score_generators))` and
        contains all score values calculated for every model over the test data of every split.
        The per subject score matrix will have shape of
        `(n_splits, len(model_generators), len(per_class_score_generators), len(subject_order))` and contains score
        values for every subject calculated for every model, score function and split.
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
            "train and evaluate %d-th cross validation split with %d training and %d test samples",
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
            batch_size,
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
    batch_size: int = 100,
) -> Tuple[MultiModelScores, MultiModelPerClassScores]:
    """Train, evaluate and score multiple models for the same dataset.

    Parameters
    ----------
    subject_order : Sequence[str]
        a subject order list describing a unique order of subjects, which is used as index for incidence and
        probability matrices (column ordering)
    train_dataset: Dataset
        the training dataset used to fit the model
    test_dataset: Dataset
        the test dataset used to score the classification performance of the model
    validation_dataset : Optional[Dataset]
        optional validation dataset provided to the fit method
    model_generators: Sequence[Callable[[], ClassificationModel]]
        the list of generators for models that will be instanciated, trained and evaluated one after another
    score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]]
        generators for batched score functions that measure the overall classification performance
    per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]]
        generators for batched score functions that measure the classification performance for each subject
    train_and_evaluate: TrainAndEvaluateModelFunction = None
        training and evaluation strategy, if None, the default strategy `DefaultTrainAndEvaluateFunction` is used
    batch_size : int, optional
        the batch size at which the model is evaluated and scored, by default 100; only applies to the default
        training and evaluation strategy

    Returns
    -------
    Tuple[MultiModelScores, MultiModelPerClassScores]
        both the overall and per-class scores for each model
    """
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
            batch_size,
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
) -> Tuple[SingleModelScores, SingleModelPerClassScores]:
    """Train, evaluate and score and single model for a single dataset.

    Parameters
    ----------
    subject_order : Sequence[str]
        a subject order list describing a unique order of subjects, which is used as index for incidence and
        probability matrices (column ordering)
    train_dataset: Dataset
        the training dataset used to fit the model
    test_dataset: Dataset
        the test dataset used to score the classification performance of the model
    validation_dataset : Optional[Dataset]
        optional validation dataset provided to the fit method
    model_generators: Sequence[Callable[[], ClassificationModel]]
        the list of generators for models that will be instanciated, trained and evaluated one after another
    score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]]
        generators for batched score functions that measure the overall classification performance
    per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]]
        generators for batched score functions that measure the classification performance for each subject
    train_and_evaluate: TrainAndEvaluateModelFunction = None
        training and evaluation strategy, if None, the default strategy `DefaultTrainAndEvaluateFunction` is used
    batch_size : int, optional
        the batch size at which the model is evaluated and scored, by default 100; only applies to the default
        training and evaluation strategy

    Returns
    -------
    Tuple[SingleModelScores, SingleModelPerClassScores]
        both the overall scores and per-class scores that are the result of evaluating the model
    """
    # load default training and evaluation methods
    if train_and_evaluate is None:
        train_and_evaluate = DefaultTrainAndEvaluateFunction(batch_size)

    # create new instance of model
    model = model_generator()
    logger.info("train and evaluate model %s", str(model))

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
