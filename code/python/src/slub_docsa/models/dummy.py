"""Various dummy models."""

from typing import Sequence

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import Model


class OracleModel(Model):
    """Models that knows the true subjects and perfectly predicts subjects."""

    def __init__(self):
        """Initialize model."""
        self.test_targets = None

    def set_test_targets(self, test_targets: np.ndarray):
        """Provide test targets that will be returned when `OracleModel.predict_proba()` is called."""
        self.test_targets = test_targets

    def fit(self, train_documents: Sequence[Document], train_targets: np.ndarray):
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


class NihilisticModel(Model):
    """Model that always predicts 0 probabilitiy."""

    def __init__(self):
        """Initialize model."""
        self.n_subjects = None

    def fit(self, train_documents: Sequence[Document], train_targets: np.ndarray):
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
