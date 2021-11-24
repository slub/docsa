"""Use scikit-learn methods and algorithms to provide model implementations."""

from typing import Iterable, Optional, Sequence, Any

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import ClassificationModel
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer
from slub_docsa.evaluation.incidence import subject_targets_from_incidence_matrix


class ScikitClassifier(ClassificationModel):
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
