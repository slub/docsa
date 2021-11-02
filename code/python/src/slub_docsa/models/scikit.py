"""Use scikit-learn methods and algorithms to provide model implementations."""

from typing import Iterable, Sequence, Any

import numpy as np

from scipy.sparse.csr import csr_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

from slub_docsa.common.document import Document
from slub_docsa.common.model import Model
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer, RandomVectorizer
from slub_docsa.evaluation.incidence import subject_targets_from_incidence_matrix


class ScikitClassifier(Model):
    """Model that uses a Scikit-Learn Predictor that supports multiple labels.

    A list of supported multi-label models can be found here: https://scikit-learn.org/stable/modules/multiclass.html
    """

    def __init__(self, predictor, vectorizer: AbstractVectorizer):
        """Initialize meta classifier with scikit-learn predictor class."""
        self.predictor = predictor
        self.vectorizer = vectorizer
        self.subject_order = None

    def _vectorize(self, documents: Sequence[Document]) -> Any:
        if not self.vectorizer:
            raise RuntimeError("Vectorizer not initialized, execute fit before predict!")

        corpus = [document_as_concatenated_string(d) for d in documents]
        features = self.vectorizer.transform(corpus)

        # NaiveBayes classifier requires features as full-size numpy matrix
        if isinstance(features, csr_matrix) and hasattr(self.predictor, "estimator") \
                and isinstance(self.predictor.estimator, GaussianNB):
            features = features.toarray()

        return features

    def fit(self, train_documents: Sequence[Document], train_targets: np.ndarray):
        """Fit model with training documents and subjects."""
        corpus = [document_as_concatenated_string(d) for d in train_documents]
        self.vectorizer.fit(corpus)
        features = self._vectorize(train_documents)
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
        return f"<ScikitClassifier predictor={str(self.predictor)} vectorizer={str(self.vectorizer)}>"


class ScikitTfidiRandomClassifier(ScikitClassifier):
    """Predict fully random probabilities for each class."""

    def __init__(self):
        """Initialize random classifier."""
        super().__init__(DummyClassifier(strategy="uniform"), vectorizer=RandomVectorizer())

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Predict random probabilities between 0 and 1 for each class individually."""
        probabilities = super().predict_proba(test_documents)
        return np.random.random(size=probabilities.shape)
