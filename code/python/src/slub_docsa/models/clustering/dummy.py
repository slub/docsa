"""Simple dummy clustering algorithms."""

from typing import Sequence

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import ClusteringModel
from slub_docsa.evaluation.incidence import crips_cluster_assignments_to_membership_matrix


class RandomClusteringModel(ClusteringModel):
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
        # assign documents randomly to clusters
        assignments = np.random.default_rng().integers(low=0, high=self.n_clusters, size=len(documents))
        return crips_cluster_assignments_to_membership_matrix(assignments)

    def __str__(self):
        """Return string describing model."""
        return "<RandomClusteringModel>"
