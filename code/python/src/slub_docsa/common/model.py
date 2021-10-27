"""Base class describing a classification model that can be used for training and prediction."""

from typing import Sequence

import numpy as np

from slub_docsa.common.document import Document


class Model:
    """Represents a model similar to a scikit-learn estimator and predictor interface.

    However, the input of both fit and predict_proba methods are a collection of `Document` instances,
    instead of raw vectorized feaures.
    """

    def fit(self, train_documents: Sequence[Document], train_targets: np.ndarray):
        """Train a model to fit the training data.

        Parameters
        ----------
        train_documents: Sequence[Document]
            The sequence of documents that is used for training a model.
        train_targets: numpy.ndarray
            The incidence matrix describing which document of `train_documents` belongs to which subjects. The matrix
            has to have a shape of (n_docs, n_subjects).

        Returns
        -------
        Model
            self
        """
        raise NotImplementedError()

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Return predicted subject probabilities as a matrix.

        Parameters
        ----------
        test_documents: Sequence[Document]
            The test sequence of documents that are supposed to be evaluated.

        Returns
        -------
        numpy.ndarray
            The matrix of subject probabilities with a shape of (n_docs, n_subjects). The column order has to match
            the order that was provided as `train_targets` to the `fit` method.
        """
        raise NotImplementedError()
