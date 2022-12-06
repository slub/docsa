"""Use scikit-learn methods and algorithms to provide model implementations."""

import os
import pickle  # nosec
import logging
import gzip
import time

from typing import Iterable, Optional, Sequence, Any

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import PersistableClassificationModel
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer, PersistableVectorizerMixin
from slub_docsa.evaluation.classification.incidence import subject_targets_from_incidence_matrix

logger = logging.getLogger(__name__)


class ScikitClassifier(PersistableClassificationModel):
    """Model that uses a Scikit-Learn Predictor that supports multiple labels.

    A list of supported multi-label models can be found [here](https://scikit-learn.org/stable/modules/multiclass.html).

    Example
    -------
    >>> # define a scikit-learn classifier
    >>> predictor = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1),
    >>> # define a vectorizer
    >>> vectorizer = slub_docsa.data.preprocess.vectorizer.TfidfStemmingVectorizer("en", max_features=1000)
    >>> # initialize both as a classification model
    >>> ScikitClassifier(predictor, vectorizer)
    <slub_docsa.models.scikit.ScikitClassifier object at 0x7fc4796b4a20>
    """

    def __init__(self, predictor, vectorizer: AbstractVectorizer):
        """Initialize meta classifier with scikit-learn predictor class."""
        self.predictor = predictor
        self.vectorizer = vectorizer
        self.subject_order = None
        self.fitted_once = False

    def _vectorize(self, documents: Sequence[Document]) -> Any:
        if not self.vectorizer:
            raise RuntimeError("Vectorizer not initialized, execute fit before predict!")

        features = list(self.vectorizer.transform(document_as_concatenated_string(d) for d in documents))
        features = np.array(features)

        # NaiveBayes classifier requires features as full-size numpy matrix
        # if isinstance(features, csr_matrix) and hasattr(self.predictor, "estimator") \
        #         and isinstance(self.predictor.estimator, GaussianNB):
        #     features = features.toarray()

        return features

    def fit(
        self,
        train_documents: Sequence[Document],
        train_targets: np.ndarray,
        validation_documents: Optional[Sequence[Document]] = None,
        validation_targets: Optional[np.ndarray] = None,
    ):
        """Fit model with training documents and subjects."""
        self.vectorizer.fit(document_as_concatenated_string(d) for d in train_documents)
        features = self._vectorize(train_documents)
        self.predictor.fit(features, train_targets)
        self.fitted_once = True

    def _predict(self, test_documents: Sequence[Document]) -> Iterable[Iterable[str]]:
        """Predict subjects of test documents."""
        if not self.subject_order:
            raise RuntimeError("subject list not initialized, execute fit before predict!")

        features = self._vectorize(test_documents)
        incidence_matrix = self.predictor.predict(features)
        predictions = subject_targets_from_incidence_matrix(incidence_matrix, self.subject_order)
        return predictions

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Predict subject probabilities for test documents."""
        features = self._vectorize(test_documents)
        started = time.time() * 1000
        probability_list = self.predictor.predict_proba(features)
        logger.debug("scikit predict_proba took %d ms", ((time.time() * 1000) - started))
        if len(probability_list[0].shape) > 1:
            probability_matrix = np.stack(list(map(lambda p: p[:, -1], probability_list)), axis=0)
        else:
            probability_matrix = np.stack(probability_list, axis=1)
        probability_matrix = np.transpose(probability_matrix)
        return probability_matrix

    def save(self, persist_dir):
        """Save scikit predictor class to disc using pickle."""
        if not self.fitted_once:
            raise ValueError("can not persist model that was not fitted before")

        logger.info("save scikit predictor to %s", persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        with gzip.open(os.path.join(persist_dir, "scikit_predictor.pickle.gz"), "wb") as file:
            pickle.dump(self.predictor, file)

        logger.info("save vectorizer to %s", persist_dir)
        if not isinstance(self.vectorizer, PersistableVectorizerMixin):
            raise ValueError("can not save vectorizer that is not persistable")
        self.vectorizer.save(persist_dir)

    def load(self, persist_dir):
        """Load scikit predictor class from disc using pickle."""
        predictor_path = os.path.join(persist_dir, "scikit_predictor.pickle.gz")

        if not os.path.exists(predictor_path):
            raise ValueError(f"can not load predictor state from file that does not exist at {predictor_path}")

        logger.info("load scikit predictor from %s", predictor_path)
        with gzip.open(predictor_path, "rb") as file:
            self.predictor = pickle.load(file)  # nosec

        logger.info("load vectorizer from %s", persist_dir)
        if not isinstance(self.vectorizer, PersistableVectorizerMixin):
            raise ValueError("can not load vectorizer that is not persistable")
        self.vectorizer.load(persist_dir)

    def __str__(self):
        """Return string describing meta classifier."""
        return f"<ScikitClassifier predictor={str(self.predictor)} vectorizer={str(self.vectorizer)}>"
