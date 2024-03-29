"""Scikit based clustering models."""

import logging

from typing import Any, Sequence

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import ClusteringModel
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer
from slub_docsa.evaluation.clustering.membership import crips_cluster_assignments_to_membership_matrix

logger = logging.getLogger(__name__)


class ScikitClusteringModel(ClusteringModel):
    """Assigns random clusters to documents."""

    def __init__(self, model: Any, vectorizer: AbstractVectorizer):
        """Initialize clustering model.

        Parameters
        ----------
        scikit_model: Any
            the scikit clustering model
        vectorizer: AbstractVectorizer
            the vectorizer used to transform documents to feature vectors
        """
        self.model = model
        self.vectorizer = vectorizer

    def fit(self, documents: Sequence[Document]):
        """Fit vectorizer and ."""
        if hasattr(self.model, "predict"):
            # if algorithm has a separate predict function, it makes sense to learn a model first
            self.vectorizer.fit(document_as_concatenated_string(d) for d in documents)
            features = np.array(list(self.vectorizer.transform(document_as_concatenated_string(d) for d in documents)))
            logger.debug("run fit on scikit clustering with features %s", features.shape)
            self.model.fit(features)
            logger.debug("done fitting scikit clustering")
        else:
            logger.debug("fitting is skipped, since algorithm is %s", str(self.model))

    def predict(self, documents: Sequence[Document]):
        """Predict cluster membership matrix by randomly assigning documents."""
        features = np.array(list(self.vectorizer.transform(document_as_concatenated_string(d) for d in documents)))

        # assume crisp clustering
        if hasattr(self.model, "predict"):
            assignments = self.model.predict(features)
        else:
            assignments = self.model.fit_predict(features)

        return crips_cluster_assignments_to_membership_matrix(assignments)

    def __str__(self):
        """Return string describing model."""
        return f"<ScikitClusteringModel algorithm={self.model} vectorizer={self.vectorizer}>"
