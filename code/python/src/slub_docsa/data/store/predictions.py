"""Persistent storage for model predictions."""

# pylint: disable=too-many-arguments, too-many-function-args

import logging
import hashlib
import time

from typing import Optional, Sequence, Tuple, Callable

from sqlitedict import SqliteDict

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.model import ClassificationModel
from slub_docsa.common.score import BatchedMultiClassProbabilitiesScore
from slub_docsa.common.score import BatchedPerClassProbabilitiesScore
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.evaluation.classification.pipeline import SingleModelPerClassScores, SingleModelScores
from slub_docsa.evaluation.classification.pipeline import TrainAndEvaluateModelFunction
from slub_docsa.evaluation.classification.pipeline import default_batch_predict_model, default_train_model

logger = logging.getLogger(__name__)


def persisted_training_and_evaluation(
    filepath: str,
    load_cached_scores: bool = False,
    batch_size: int = 100,
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
    store = SqliteDict(filepath, tablename="scores", autocommit=True, flag="c")

    def train_and_evaluate(
        model: ClassificationModel,
        subject_order: Sequence[str],
        train_dataset: Dataset,
        test_dataset: Dataset,
        score_generators: Sequence[Callable[[], BatchedMultiClassProbabilitiesScore]],
        per_class_score_generators: Sequence[Callable[[], BatchedPerClassProbabilitiesScore]],
        validation_dataset: Optional[Dataset] = None,
    ) -> Tuple[SingleModelScores, SingleModelPerClassScores]:

        # calculate hash over train and test data
        hashing_start = time.time()
        hasher = hashlib.sha1()  # nosec
        hasher.update(str(model).encode())

        for score_generator in score_generators:
            hasher.update(str(score_generator()).encode())

        for per_class_score_generator in per_class_score_generators:
            hasher.update(str(per_class_score_generator()).encode())

        for doc in train_dataset.documents:
            hasher.update(document_as_concatenated_string(doc).encode())

        for doc in test_dataset.documents:
            hasher.update(document_as_concatenated_string(doc).encode())

        hash_code = hasher.hexdigest()
        logger.info("hash code for scores is %s, needed %d ms", hash_code, (time.time() - hashing_start) * 1000)
        if load_cached_scores and hash_code in store:
            logger.info("use cached scores, skip training and evaluation")
            scores, per_class_scores = store[hash_code]
            logger.info("scores for model %s are %s", str(model), scores)
            return scores, per_class_scores

        logger.info("score not available, train and evaluate the model")
        default_train_model(model, subject_order, train_dataset, validation_dataset)
        scores, per_class_scores = default_batch_predict_model(
            model, subject_order, test_dataset, score_generators, per_class_score_generators, batch_size
        )

        logger.info("store scores to cache for hash %s", hash_code)
        store[hash_code] = (scores, per_class_scores)

        return scores, per_class_scores

    return train_and_evaluate
