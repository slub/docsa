"""Model that knows the true subjects an does not make any mistake."""

from typing import Sequence

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import Model


class OracleModel(Model):
    """Perflectly predicts subjects."""

    def __init__(self):
        """Initialize model."""
        self.test_targets = None

    def set_test_targets(self, test_targets: np.ndarray):
        """Provide test targets that will be returned when predict_proba is called."""
        self.test_targets = test_targets

    def fit(self, train_documents: Sequence[Document], train_targets: np.ndarray):
        """Do not fit an oracle."""
        # do not learn anything
        return self

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Return test targets as provided by ``set_test_targets."""
        if self.test_targets is None:
            raise RuntimeError("test targets are missing")
        return self.test_targets
