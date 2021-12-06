"""Base classes describing both classification and clustering models that can be used for training and prediction."""

from typing import Optional, Sequence

import numpy as np

from slub_docsa.common.document import Document


class ClassificationModel:
    """Represents a classification model similar to a scikit-learn estimator and predictor interface.

    However, the input of both fit and predict_proba methods are a collection of `Document` instances,
    instead of raw vectorized feaures.
    """

    def fit(
        self,
        train_documents: Sequence[Document],
        train_targets: np.ndarray,
        validation_documents: Optional[Sequence[Document]] = None,
        validation_targets: Optional[np.ndarray] = None,
    ):
        """Train a model to fit the training data.

        Parameters
        ----------
        train_documents: Sequence[Document]
            The sequence of documents that is used for training a model.
        train_targets: numpy.ndarray
            The incidence matrix describing which document of `train_documents` belongs to which subjects. The matrix
            has to have a shape of (n_docs, n_subjects).
        validation_documents: Optional[Sequence[Document]]
            A sequence of documents that can be used to validate the trained model during training, e.g., for each
            epoch when training an artificial neural network
        validation_targets: Optional[numpy.ndarray]
            The incidence matrix for `validation_documents`

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


class PersistableClassificationModel(ClassificationModel):
    """Extends a classification model for save/load methods such that model can be persisted."""

    def save(self, persist_dir: str):
        """Save the fitted model state to disk at some directory.

        Parameters
        ----------
        persist_dir: str
            the path to a directory that can be used to save the model state
        """
        raise NotImplementedError()

    def load(self, persist_dir: str):
        """Load a persisted model state from some directory.

        Parameters
        ----------
        persist_dir: str
            the path to the directory from which the persisted model is loaded
        """
        raise NotImplementedError()


class ClusteringModel:
    """Represents a clustering model similar to the scikit-learn fit and predict clustering model.

    However, the input of both fit and predict methods are a collection of `Document` instances,
    instead of raw vectorized feaures.
    """

    def fit(self, documents: Sequence[Document]):
        """Train a clustering model in case the clustering algorithm is based on some kind of model."""
        raise NotImplementedError()

    def predict(self, documents: Sequence[Document]) -> np.ndarray:
        """Predict cluster assignments as a membership matrix of shape (len(documents), len(clusters)).

        The membership matrix may resemble an exact clustering, meaning each document is assigned to exactly one
        cluster. Alternatively, it may contain membership degrees, meaning each document can be part of multiple
        clusters to a certain degree, such that their degrees sum up to 1. Otherwise, the membership matrix may
        have arbitrary membership degrees, e.g., to represent a hierarchical clustering.

        In all cases, the maximum value of membership matrix elements is 1.
        """
        raise NotImplementedError()
