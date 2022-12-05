"""Base class describing a scoring function for evaluation.

Note: Type aliases `MultiClassScoreFunctionType`, `BinaryClassScoreFunctionType` and `IncidenceDecisionFunctionType`
are not correctly described in API documentation due to [issue in pdoc](https://github.com/pdoc3/pdoc/issues/229).
"""

# pylint: disable=too-few-public-methods

from typing import Optional, Sequence

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectTargets


class MultiClassProbabilitiesScore:
    """Score function comparing true and predicted subject probabilities for multiple classes for each document."""

    def __call__(self, true_probabilities: np.ndarray, predicted_probabilities: np.ndarray) -> float:
        """Return score comparing multi-class subject probability matrices.

        Parameters
        ----------
        true_probabilities: np.ndarray
            the matrix containing the true subject probabilities in shape (documents, subjects).
        predicted_probabilities: np.ndarray
            the matrix containing predicted subject probabilities in shape (documents, subjects).

        Returns
        -------
        float
            the score
        """
        raise NotImplementedError()


class BatchedMultiClassProbabilitiesScore:
    """Batched score function comparing true and predicted subject probabilities for multiple classes."""

    def add_batch(self, true_probabilities: np.ndarray, predicted_probabilities: np.ndarray):
        """Add multi-class subject probability matrices for a batch of documents to be processed for scoring.

        Parameters
        ----------
        true_probabilities: np.ndarray
            the matrix containing the true subject probabilities in shape (document_batch, subjects).
        predicted_probabilities: np.ndarray
            the matrix containing predicted subject probabilities in shape (document_batch, subjects).
        """
        raise NotImplementedError()

    def __call__(self) -> float:
        """Return current score comparing multi-class subject probabilities that were added in batches."""
        raise NotImplementedError()


class MultiClassIncidenceScore:
    """Score function comparing true and predicted subject incidences for multiple classes for each document."""

    def __call__(self, true_incidences: np.ndarray, predicted_incidences: np.ndarray) -> float:
        """Return score comparing multi-class subject incidence matrices.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject incidences in shape (documents, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject incidences in shape (documents, subjects).

        Returns
        -------
        float
            the score
        """
        raise NotImplementedError()


class BatchedMultiClassIncidenceScore:
    """Batched score function comparing true and predicted subject incidences for multiple classes."""

    def add_batch(self, true_incidences: np.ndarray, predicted_incidences: np.ndarray):
        """Add multi-class subject incidence matrices for a batch of documents to be processed for scoring.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject incidences in shape (document_batch, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject incidences in shape (document_batch, subjects).
        """
        raise NotImplementedError()

    def __call__(self) -> float:
        """Return current score comparing multi-class subject incidences that were added in batches."""
        raise NotImplementedError()


class PerClassProbabilitiesScore:
    """Score function comparing true and predicted subject probabilities for a single class."""

    def __call__(self, true_probabilities: np.ndarray, predicted_probabilities: np.ndarray) -> Sequence[float]:
        """Return score comparing true subject incidence and and predicted subject probabilities for a single class.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject incidences in shape (documents, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject incidences in shape (documents, subjects).

        Returns
        -------
        float
            the list of scores for each subject
        """
        raise NotImplementedError()


class BatchedPerClassProbabilitiesScore:
    """Batched score function comparing true and predicted subject probabilites for a single class."""

    def add_batch(self, true_probabilities: np.ndarray, predicted_probabilities: np.ndarray):
        """Add multi-class subject probabilities matrices for a batch of documents to be processed for scoring.

        Parameters
        ----------
        true_probabilities: np.ndarray
            the matrix containing the true subject probabilities in shape (document_batch, subjects).
        predicted_probabilities: np.ndarray
            the matrix containing predicted subject probabilities in shape (document_batch, subjects).
        """
        raise NotImplementedError()

    def __call__(self) -> Sequence[float]:
        """Return the current list of scores for each subject."""
        raise NotImplementedError()


class BatchedPerClassIncidenceScore:
    """Batched score function comparing true and predicted subject incidences for a single class."""

    def add_batch(self, true_incidences: np.ndarray, predicted_incidences: np.ndarray):
        """Add multi-class subject incidence matrices for a batch of documents to be processed for scoring.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject incidences in shape (document_batch, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject incidences in shape (document_batch, subjects).
        """
        raise NotImplementedError()

    def __call__(self) -> Sequence[float]:
        """Return the current list of scores for each subject."""
        raise NotImplementedError()


class IncidenceDecisionFunction:
    """Convert a subject probabilities matrix to an incidence matrix by applying some decision."""

    def __call__(self, probabilities: np.ndarray) -> np.ndarray:
        """Return incidence matrix for a given probability matrix.

        Parameters
        ----------
        probabilities : np.ndarray
            the probabilites matrix of shape (documents, subjects)

        Returns
        -------
        np.ndarray
            the incidence matrix of shape (documents, subjects)
        """
        raise NotImplementedError()


class ClusteringScore:
    """Function that evaluates a clustering of documents."""

    def __call__(
        self,
        documents: Sequence[Document],
        membership: np.ndarray,
        targets: Optional[SubjectTargets]
    ) -> float:
        """Return a score evaluating a clustering of documents.

        Parameters
        ----------
        documents : Sequence[Document]
            the list of documents
        membership : np.ndarray
            the membership matrix of shape (len(documents), len(clusters)); the matrix may resemble an exact
            clustering, meaning each document is assigned to exactly one cluster; alternatively, it may contain
            membership degrees, meaning each document can be part of multiple clusters to a certain degree
        targets : Optional[SubjectTargets]
            the true subject assignments for each document

        Returns
        -------
        float
            the clustering score
        """
        raise NotImplementedError()
