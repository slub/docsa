"""Persistent storage for model predictions."""

# pylint: disable=too-many-arguments

import logging
import hashlib
import dbm
import io

from typing import Optional, Sequence, cast

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import Model
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.evaluation.pipeline import FitModelAndPredictCallable

logger = logging.getLogger(__name__)


def persisted_fit_model_and_predict(filepath: str, load_cached_predictions: bool = False) -> FitModelAndPredictCallable:
    """Load model predictions from a dbm database if they have been stored previously.

    Stored predictions are only used if the exact same training and test data is used for the same model.
    This is checked by calculating a hash function over both the training and test data, as well as the descriptive
    string `__str__()` of the model class.

    Parameters
    ----------
    filepath: str
        The path to the dbm database where predictions are stored
    load_cached_predictions: bool = False
        A flag to prevent loading predictions from cache in case it is disabled, e.g., upon user request

    Returns
    -------
    FitModelAndPredictCallable
        a function that calculates predictions based on the training data, test data and model
    """
    store = dbm.open(filepath, "c")

    def fit_model_and_predict(
        model: Model,
        train_documents: Sequence[Document],
        train_incidence_matrix: np.ndarray,
        test_documents: Sequence[Document],
        validation_documents: Optional[Sequence[Document]] = None,
        validation_incidence_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        # calculate hash over train and test data
        hasher = hashlib.sha1()  # nosec
        hasher.update(model.__str__().encode())

        for doc in train_documents:
            hasher.update(document_as_concatenated_string(doc).encode())

        for doc in test_documents:
            hasher.update(document_as_concatenated_string(doc).encode())

        hash_code = hasher.digest()
        if load_cached_predictions and hash_code in store:
            logger.info("use cached predicitons")
            data = io.BytesIO(store[hash_code])
            data.seek(0)
            return cast(np.ndarray, np.load(data))

        logger.info("do training")
        model.fit(train_documents, train_incidence_matrix, validation_documents, validation_incidence_matrix)

        logger.info("do prediction")
        predictions = model.predict_proba(test_documents)

        logger.info("store predictions to cache")
        data = io.BytesIO()
        np.save(data, predictions)
        data.seek(0)
        store[hash_code] = data.read()

        return predictions

    return fit_model_and_predict
