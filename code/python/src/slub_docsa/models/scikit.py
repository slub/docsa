"""Use scikit-learn methods and algorithms to provide model implementations."""

from typing import Collection, Iterable, Sequence, Any, cast

import numpy as np

from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

from slub_docsa.common.document import Document
from slub_docsa.common.model import Model
from slub_docsa.evaluation.incidence import subject_targets_from_incidence_matrix


class ScikitTfidfClassifier(Model):
    """Model that uses Scikit-Learn TfidfVectorizer and a Scikit-Learn Predictor that supports multiple labels.

    A list of supported multi-label models can be found here: https://scikit-learn.org/stable/modules/multiclass.html
    """

    def __init__(self, predictor):
        """Initialize meta classifier with scikit-learn predictor class."""
        self.predictor = predictor
        self.vectorizer = None
        self.subject_order = None

    def _build_vectorizer(self, documents: Collection[Document]):
        self.vectorizer = TfidfVectorizer()
        corpus = [d.title for d in documents]
        features = cast(csr_matrix, self.vectorizer.fit_transform(corpus))
        if hasattr(self.predictor, "estimator") and isinstance(self.predictor.estimator, GaussianNB):
            features = features.toarray()
        return features

    def _vectorize(self, documents: Sequence[Document]) -> Any:
        if not self.vectorizer:
            raise RuntimeError("Vectorizer not initialized, execute fit before predict!")

        corpus = [d.title for d in documents]
        features = cast(csr_matrix, self.vectorizer.transform(corpus))
        if hasattr(self.predictor, "estimator") and isinstance(self.predictor.estimator, GaussianNB):
            features = features.toarray()
        return features

    def fit(self, train_documents: Sequence[Document], train_targets: np.ndarray):
        """Fit model with training documents and subjects."""
        features = self._build_vectorizer(train_documents)
        self.predictor.fit(features, train_targets)

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
        probability_list = self.predictor.predict_proba(features)
        if len(probability_list[0].shape) > 1:
            probability_matrix = np.stack(list(map(lambda p: p[:, -1], probability_list)), axis=0)
        else:
            probability_matrix = np.stack(probability_list, axis=1)
        probability_matrix = np.transpose(probability_matrix)
        return probability_matrix

    def __str__(self):
        """Return string describing meta classifier."""
        return f"<ScikitTfidfClassifier predictor={str(self.predictor)} >"
