"""Various dummy models."""

from typing import Optional, Sequence

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import ClassificationModel


class OracleModel(ClassificationModel):
    """Models that knows the true subjects and perfectly predicts subjects.

    In order to gain access to the target subjects, which is usually not possible for a model implementation,
    the method `set_test_targets` needs to be called with true subject targets prior to predicting via `predict_proba`.
    Evaluation methods in `slub_docsa.evaluation.pipeline` have been designed to detect when a subclass of the
    `OracleModel` is evaluated and will call `set_test_targets` accordingly.
    """

    def __init__(self):
        """Initialize model."""
        self.test_targets = None

    def set_test_targets(self, test_targets: np.ndarray):
        """Provide test targets that will be returned when `OracleModel.predict_proba()` is called."""
        self.test_targets = test_targets

    def fit(
        self,
        train_documents: Sequence[Document],
        train_targets: np.ndarray,
        validation_documents: Optional[Sequence[Document]] = None,
        validation_targets: Optional[np.ndarray] = None,
    ):
        """Do not fit an oracle."""
        # do not learn anything
        return self

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Return test targets as provided by `OracleModel.set_test_targets()`."""
        if self.test_targets is None:
            raise RuntimeError("test targets are missing")
        return self.test_targets

    def __str__(self):
        """Return string describing model."""
        return "<OracleModel>"


class NihilisticModel(ClassificationModel):
    """Model that always predicts 0 probabilitiy."""

    def __init__(self):
        """Initialize model."""
        self.n_subjects = None

    def fit(
        self,
        train_documents: Sequence[Document],
        train_targets: np.ndarray,
        validation_documents: Optional[Sequence[Document]] = None,
        validation_targets: Optional[np.ndarray] = None,
    ):
        """Do not learn anything."""
        self.n_subjects = train_targets.shape[1]
        return self

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Return test targets as provided by `OracleModel.set_test_targets()`."""
        if self.n_subjects is None:
            raise ValueError("number of subjects not known, did you call fit?")
        return np.zeros((len(test_documents), self.n_subjects))

    def __str__(self):
        """Return string describing model."""
        return "<NihilisticModel>"


class RandomModel(ClassificationModel):
    """Predict fully random probabilities for each class."""

    def __init__(self):
        """Initialize random classifier."""
        self.n_subjects = None

    def fit(
        self,
        train_documents: Sequence[Document],
        train_targets: np.ndarray,
        validation_documents: Optional[Sequence[Document]] = None,
        validation_targets: Optional[np.ndarray] = None,
    ):
        """Do not learn anything."""
        self.n_subjects = train_targets.shape[1]
        return self

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Predict random probabilities between 0 and 1 for each class individually."""
        return np.random.random(size=(len(test_documents), self.n_subjects))
