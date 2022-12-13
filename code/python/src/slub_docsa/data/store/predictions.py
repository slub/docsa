"""Persistent storage for model predictions."""

# pylint: disable=too-many-arguments, too-many-function-args, too-many-locals

import logging
import hashlib
import os

from typing import Optional, Sequence, Tuple, Callable

from sqlitedict import SqliteDict

import slub_docsa

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.model import ClassificationModel, PersistableClassificationModel
from slub_docsa.common.score import BatchedMultiClassProbabilitiesScore
from slub_docsa.common.score import BatchedPerClassProbabilitiesScore
from slub_docsa.data.store.model import save_as_published_classification_model
from slub_docsa.evaluation.classification.pipeline import SingleModelPerClassScores, SingleModelScores
from slub_docsa.evaluation.classification.pipeline import TrainAndEvaluateModelFunction
from slub_docsa.evaluation.classification.pipeline import default_batch_evaluate_model, default_train_model
from slub_docsa.serve.common import PublishedClassificationModelInfo, PublishedClassificationModelStatistics
from slub_docsa.serve.common import current_date_as_model_creation_date

logger = logging.getLogger(__name__)


def _calculate_hash(
    model_type: str,
    dataset_name: str,
    split_id: str,
    score_names: Sequence[str],
    per_class_score_names: Sequence[str],
) -> str:
    # calculate hash over relevant paramters
    hasher = hashlib.sha1()  # nosec
    hasher.update(model_type.encode())
    hasher.update(dataset_name.encode())
    hasher.update(split_id.encode())
    hasher.update(",".join(score_names).encode())
    hasher.update(",".join(per_class_score_names).encode())
    return hasher.hexdigest()


def persisted_training_and_evaluation(
    score_directory: str,
    model_publish_directory: str,
    schema_id: str,
    dataset_name: str,
    model_type: str,
    supported_languages: Sequence[str],
    split_id: str,
    score_names: Sequence[str],
    per_class_score_names: Sequence[str],
    evluate_batch_size: int = 100,
    publish_model: bool = True,
    load_cached_scores: bool = True,
) -> TrainAndEvaluateModelFunction:
    """Load model evaluation scores from a sqlite database if they have been stored previously.

    Stored evaluation scores are only used if the exact same training and test data is used for the same model.
    This is checked by calculating a hash function over both the training and test data, as well as the descriptive
    string `__str__()` of the model class.

    Parameters
    ----------
    filepath: str
        The path to the sqlite database where evaluation scores are stored
    load_cached_scores: bool = False
        A flag to prevent loading scores from cache (the are still saved though)
    batch_size: int = 100
        The batch size at which models are evaluated using test data and scores are calculated

    Returns
    -------
    TrainAndEvaluateModelFunction
        the function that can be used as a replacement for the default strategy, such that calculated evaluation scores
        are persisted and loaded from the generated sqlite database
    """
    model_id = f"{dataset_name}__{model_type}"
    score_cache_filepath = os.path.join(score_directory, model_id + ".sqlite")
    score_cache = SqliteDict(score_cache_filepath, tablename="scores", autocommit=True, flag="c")
    hash_code = _calculate_hash(model_type, dataset_name, split_id, score_names, per_class_score_names)

    def train_and_evaluate(
        model_generator: Callable[[], ClassificationModel],
        subject_order: Sequence[str],
        train_dataset: Dataset,
        test_dataset: Dataset,
        score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]],
        per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]],
        validation_dataset: Optional[Dataset] = None,
    ) -> Tuple[SingleModelScores, SingleModelPerClassScores]:
        if load_cached_scores and hash_code in score_cache:
            logger.info("use cached scores, skip training and evaluation")
            scores, per_class_scores = score_cache[hash_code]
            logger.info("scores for model %s are %s", str(model_type), dict(zip(score_names, scores)))
            return scores, per_class_scores

        logger.info("score not available, train and evaluate the model")
        model = model_generator()
        default_train_model(model, subject_order, train_dataset, validation_dataset)
        scores, per_class_scores = default_batch_evaluate_model(
            model, subject_order, test_dataset, score_generators, per_class_score_generators, evluate_batch_size
        )

        logger.info("store scores to cache for hash %s", hash_code)
        score_cache[hash_code] = (scores, per_class_scores)
        score_cache.close()

        if publish_model and not isinstance(model, PersistableClassificationModel):
            logger.warning("cannot publish model of type '%s'", model_type)

        if publish_model and isinstance(model, PersistableClassificationModel):
            logger.info("save model with id '%s'", model_id)
            save_as_published_classification_model(
                directory=os.path.join(model_publish_directory, model_id + "__split=" + split_id),
                model=model,
                subject_order=subject_order,
                model_info=PublishedClassificationModelInfo(
                    model_id=model_id,
                    model_type=model_type,
                    model_version="0.0.0",
                    schema_id=schema_id,
                    creation_date=current_date_as_model_creation_date(),
                    supported_languages=supported_languages,
                    description=f"model trained for dataset variant '{dataset_name}' "
                                + f"with classifiation model '{model_type}'",
                    tags=[],
                    slub_docsa_version=slub_docsa.__version__,
                    statistics=PublishedClassificationModelStatistics(
                        number_of_training_samples=len(train_dataset.subjects),
                        number_of_test_samples=len(test_dataset.subjects),
                        scores=dict(zip(score_names, scores))
                    )
                )
            )
        return scores, per_class_scores

    return train_and_evaluate
