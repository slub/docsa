"""Use scikit-learn methods and algorithms to provide model implementations."""

from typing import Collection, Iterable, Tuple, Any, cast

import numpy as np

from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

from slub_docsa.common.document import Document
from slub_docsa.common.model import Model
from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_list, subject_list_from_incidence_matrix
from slub_docsa.evaluation.incidence import unique_subject_list


class ScikitTfidfClassifier(Model):
    """Model that uses Scikit-Learn TfidfVectorizer and a Scikit-Learn Predictor that supports multiple labels.

    A list of supported multi-label models can be found here: https://scikit-learn.org/stable/modules/multiclass.html
    """

    def __init__(self, predictor):
        """Initialize meta classifier with scikit-learn predictor class."""
        self.predictor = predictor
        self.vectorizer = None
        self.subject_list = None

    def _build_vectorizer(self, documents: Collection[Document]):
        self.vectorizer = TfidfVectorizer()
        corpus = [d.title for d in documents]
        features = cast(csr_matrix, self.vectorizer.fit_transform(corpus))
        if hasattr(self.predictor, "estimator") and isinstance(self.predictor.estimator, GaussianNB):
            features = features.toarray()
        return features

    def _vectorize(self, documents: Collection[Document]) -> Any:
        if not self.vectorizer:
            raise RuntimeError("Vectorizer not initialized, execute fit before predict!")

        corpus = [d.title for d in documents]
        features = cast(csr_matrix, self.vectorizer.transform(corpus))
        if hasattr(self.predictor, "estimator") and isinstance(self.predictor.estimator, GaussianNB):
            features = features.toarray()
        return features

    def fit(self, train_documents, train_targets):
        """Fit model with training documents and subjects."""
        features = self._build_vectorizer(train_documents)
        self.subject_list = unique_subject_list(train_targets)
        incidence_matrix = subject_incidence_matrix_from_list(train_targets, self.subject_list)
        self.predictor.fit(features, incidence_matrix)

    def predict(self, test_documents: Collection[Document]) -> Iterable[Iterable[str]]:
        """Predict subjects of test documents."""
        if not self.subject_list:
            raise RuntimeError("subject list not initialized, execute fit before predict!")

        features = self._vectorize(test_documents)
        incidence_matrix = self.predictor.predict(features)
        subject_list = subject_list_from_incidence_matrix(incidence_matrix, self.subject_list)
        return subject_list

    def _predict_proba(self, test_documents: Collection[Document]) -> Tuple[np.ndarray, Iterable[str]]:
        if not self.subject_list:
            raise RuntimeError("subject list not initialized, execute fit before predict!")

        features = self._vectorize(test_documents)
        probability_matrix = np.array(self.predictor.predict_proba(features))
        if len(probability_matrix.shape) == 3:
            probability_matrix = np.transpose(probability_matrix[:, :, 1])
        return probability_matrix, self.subject_list

    def __str__(self):
        """Return string describing meta classifier."""
        return f"<ScikitTfidfClassifier predictor={str(self.predictor)} >"
