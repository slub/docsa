"""Simple dummy clustering algorithms."""

from typing import Sequence

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import ClusteringModel


class RandomClustering(ClusteringModel):
    """Assigns random clusters to documents."""

    def __init__(self, n_clusters: int = 5):
        """Initialize clustering model.

        Parameters
        ----------
        n_clusters: int = 5
            the number of random clusters that are being assigned
        """
        self.n_clusters = n_clusters

    def fit(self, documents: Sequence[Document]):
        """Is not doing anything."""

    def predict(self, documents: Sequence[Document]):
        """Predict cluster membership matrix by randomly assigning documents."""
        # initialize membership matrix
        membership = np.zeros((len(documents), self.n_clusters))

        # assign documents randomly to clusters
        assignments = np.random.default_rng().integers(min=0, max=self.n_clusters, size=len(documents))

        # set membership in matrix
        membership[list(range((len(documents)))), assignments] = 1

        return membership
