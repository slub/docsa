"""Methods to work with incidence matrices."""

import logging
from typing import Iterator, List, Sequence

import numpy as np
from slub_docsa.common.score import IncidenceDecisionFunction

from slub_docsa.common.subject import SubjectHierarchy, SubjectTargets, SubjectUriList
from slub_docsa.data.preprocess.subject import subject_ancestors_list

logger = logging.getLogger(__name__)


def unique_subject_order(targets: Sequence[SubjectUriList]) -> Sequence[str]:
    """Return the list of unique subjects found in `targets`.

    Parameters
    ----------
    targets: Sequence[Iterable[str]]
        An ordered list of lists of subjects. Each list entry corresponds to the list of subjects a document is
        classified as.

    Returns
    -------
    Sequence[str]
        A list of unique subjects found in `targets` without duplicates and in arbitrary but fixed order.

    Examples
    --------
    >>> targets = [
    ...     ["subject1"],
    ...     ["subject1", "subject2"]
    ...     ["subject1", "subject3"]
    ... ]
    >>> unique_subject_order(targets)
    ["subject3", "subject1", "subject2"]
    """
    subject_set = {uri for uri_list in targets for uri in uri_list}
    return list(subject_set)


def subject_incidence_matrix_from_targets(
    targets: Sequence[SubjectUriList],
    subject_order: Sequence[str]
) -> np.ndarray:
    """Return an incidence matrix for the list of subject annotations in `targets`.

    Parameters
    ----------
    targets: Sequence[Iterable[str]]
        an ordered list of subjects lists, each representing the subjects that associated with a document
    subject_order: Sequence[str]
        an ordered list of subjects without duplicates, e.g., generated via `unique_subject_order`

    Returns
    -------
    np.ndarray
        An incidence matrix of shape `(len(targets), len(subject_order))` representing which document is classified
        by which subject. The column order corresponds to the order of subjects found in `subject_order`.
        The order of row corresponds to the order of `targets`.
    """
    incidence_matrix = np.zeros((len(targets), len(subject_order)), dtype=np.uint8)
    subject_order_map = {s_uri: i for i, s_uri in enumerate(subject_order)}
    for i, uri_list in enumerate(targets):
        for uri in uri_list:
            if uri in subject_order_map:
                incidence_matrix[i, subject_order_map[uri]] = 1
            else:
                logger.warning("subject '%s' not given in subject_order list (maybe only in test data)", uri)
    return incidence_matrix


def subject_targets_from_incidence_matrix(
    incidence_matrix: np.ndarray,
    subject_order: Sequence[str]
) -> SubjectTargets:
    """Return subject lists for an incidence matrix given a subject ordering.

    Parameters
    ----------
    incidence_matrix: np.ndarray
        An incidence matrix representing which document is classified by which document.
    subject_order: Sequence[str]
        An order of subjects that describes the column ordering of `incidence_matrix`.

    Returns
    -------
    Sequence[Iterable[str]]
        An order list of lists of subjects. Each list entry corresponds to a row of `incidence_matrix`,
        which contains the list of subjects that equals to 1 for the corresponding element in `incidence_matrix`.
    """
    incidence_array = np.array(incidence_matrix)

    if incidence_array.shape[1] != len(subject_order):
        raise ValueError(
            f"indicence matrix has {incidence_array.shape[1]} columns "
            + f"but subject order has {len(subject_order)} entries"
        )

    return list(map(
        lambda incidence_vector: list(map(lambda i: subject_order[i], np.where(incidence_vector == 1)[0])),
        incidence_array
    ))


def subject_idx_from_incidence_matrix(
    incidence_matrix: Sequence[Sequence[int]],
):
    """Return subject indexes for an incidence matrix.

    Parameters
    ----------
    incidence_matrix: Sequence[Sequence[int]]
        An incidence matrix representing which document is classified by which document.

    Returns
    -------
    Sequence[Iterable[int]]
        An order list of lists of subjects indexes. Each list entry corresponds to a row of `incidence_matrix`,
        which contains the list of indexes (columns) that equals to 1 for the corresponding element in
        `incidence_matrix`.
    """
    return list(map(
        lambda incidence_vector: list(np.where(np.array(incidence_vector) == 1)[0]),
        incidence_matrix
    ))


class ThresholdIncidenceDecision(IncidenceDecisionFunction):
    """Incidence decision based on a simple threshold."""

    def __init__(self, threshold: float = 0.5):
        """Select subjects based on a threshold over probabilities.

        Parameters
        ----------
        threshold: float
            the minimum probability required for a subject to be chosen

        Returns
        -------
        Callback[[np.ndarray], np.ndarray]

        """
        self.threshold = threshold

    def __call__(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform a numpy array of subject probabilities to an incidence matrix by applying a thresold.

        Parameters
        ----------
        probabilities : np.ndarray
            the matrix containg probability scores between 0 and 1

        Returns
        -------
        np.ndarray
            the incidence matrix containing incidences (either 0 or 1)
        """
        # slower: return np.array(np.where(probabilities >= threshold, 1, 0))
        return (probabilities >= self.threshold).astype(np.uint8)

    def __str__(self):
        """Return a string representation of this incidence decision function, which is used for caching."""
        return f"<ThresholdIncidenceDecision threshold={self.threshold}>"


class TopkIncidenceDecision(IncidenceDecisionFunction):
    """Incidence decision function that selects the top k subjects with highest probability (including zero)."""

    def __init__(self, k: int = 3):
        """Select exactly k subjects with highest probabilities.

        Chooses random subjects if less than k subjects have positive probability.

        Parameters
        ----------
        k: int
            the maximum number of subjects that are selected and returned

        Returns
        -------
        Callback[[np.ndarray], np.ndarray]
            A functions that transforms a numpy array of subject probabilities to an incidence matrix of the
            same shape representing which subjects are chosen for which documents.
        """
        self.k = k

    def __call__(self, probabilities: np.ndarray) -> np.ndarray:
        """Return top-k incidence matrix for a probability matrix.

        Parameters
        ----------
        probabilities : np.ndarray
            the matrix containg probability scores between 0 and 1

        Returns
        -------
        np.ndarray
            the incidence matrix containing incidences (either 0 or 1)
        """
        incidence = np.zeros(probabilities.shape, dtype=np.uint8)
        x_indices = np.repeat(np.arange(0, incidence.shape[0]), self.k)
        y_indices = np.argpartition(probabilities, -self.k)[:, -self.k:].flatten()
        incidence[x_indices, y_indices] = 1
        return incidence

    def __str__(self):
        """Return a string representation of this incidence decision function, which is used for caching."""
        return f"<TopKIncidenceDecision k={self.k}>"


class PositiveTopkIncidenceDecision(IncidenceDecisionFunction):
    """Incidence decision function that selects the top k subjects with highest positive (non-zero) probability."""

    def __init__(self, k: int = 3):
        """Select at most k subjects with highest positive probabilities.

        Does not choose subjects with zero probability, and thus, may even not choose any subjects at all.
        Matches decision function used by Annif.

        Parameters
        ----------
        k: int
            the maximum number of subjects that are selected and returned

        Returns
        -------
        Callback[[np.ndarray], np.ndarray]
            A functions that transforms a numpy array of subject probabilities to an incidence matrix of the
            same shape representing which subjects are chosen for which documents.
        """
        self.k = k

    def __call__(self, probabilities: np.ndarray) -> np.ndarray:
        """Return positive top-k incidence matrix for a probability matrix.

        Parameters
        ----------
        probabilities : np.ndarray
            the matrix containg probability scores between 0 and 1

        Returns
        -------
        np.ndarray
            the incidence matrix containing incidences (either 0 or 1)
        """
        incidence = np.zeros(probabilities.shape)
        x_indices = np.repeat(np.arange(0, incidence.shape[0]), self.k)
        y_indices = np.argpartition(probabilities, -self.k)[:, -self.k:].flatten()
        incidence[x_indices, y_indices] = 1
        incidence[probabilities <= 0.0] = 0
        return incidence

    def __str__(self):
        """Return a string representation of this incidence decision function, which is used for caching."""
        return f"<PositiveTopkIncidenceDecision k={self.k}>"


def extend_incidence_list_to_ancestors(
    subject_hierarchy: SubjectHierarchy,
    subject_order: Sequence[str],
    incidence_list: Sequence[int],
) -> Sequence[int]:
    """Return an extended incidence list that marks all ancestors subjects.

    An incidence list refers to a row of the document vs. subject incidence matrix.

    Parameters
    ----------
    subject_hierarchy: SubjectHierarchy
        the subject hierarchy that is used to infer ancestor subjects
    subject_order: Sequence[str]
        the subject order that is used to infer which subject is references by which position in the incidence list
    incidence_list: Sequence[int]
        the incidence list that is extended with incidences for all ancestor subjects

    Returns
    -------
    Sequence[int]
        a new incidence list that is extended (meaning there are additional 1 entries) for all ancestors of the
        subjects previously marked in the incidence list
    """
    extended_incidence: List[int] = list(incidence_list)
    for i, value in enumerate(incidence_list):
        if value == 1:
            # iterate over ancestors and set extended incidence to 1
            subject_uri = subject_order[i]
            ancestors = subject_ancestors_list(subject_uri, subject_hierarchy)
            for ancestor in ancestors:
                if ancestor in subject_order:
                    ancestor_id = subject_order.index(ancestor)
                    extended_incidence[ancestor_id] = 1
    return extended_incidence


class LazySubjectIncidenceTargets(Sequence[Sequence[int]]):
    """Wrapper that calculates subject incidences on-the-fly when retrieving them."""

    def __init__(self, subject_targets: SubjectTargets, subject_order: Sequence[str]):
        """Initialize wrapper with subject targets and unique subject order.

        Parameters
        ----------
        subject_targets : SubjectTargets
            the subject targets that are wrapped
        subject_order : Sequence[str]
            the subject order that is used to calculate the individual incidence arrays
        """
        self.subject_targets = subject_targets
        self.number_of_samples = len(subject_targets)
        self.number_of_subjects = len(subject_order)
        self.subject_order_map = {s_uri: i for i, s_uri in enumerate(subject_order)}

    def __getitem__(self, key: int) -> Sequence[int]:
        """Retrieve an incidence array for a specific index of the subject targets list.

        Subject incidence are returned as numpy array.

        Parameters
        ----------
        key : int
            the index of the requested subject target incidence

        Returns
        -------
        Sequence[int]
            the subject target incidence array
        """
        incidence_array = np.zeros(self.number_of_subjects, dtype=np.uint8)
        for uri in self.subject_targets[key]:
            if uri in self.subject_order_map:
                incidence_array[self.subject_order_map[uri]] = 1
            else:
                logger.warning("subject '%s' not given in subject_order list (maybe only in test data)", uri)
        return incidence_array

    def __len__(self):
        """Return the number of samples in subject targets."""
        return self.number_of_samples

    def __iter__(self) -> Iterator[Sequence[int]]:
        """Iterate over all subject incidences."""
        for i in range(self.number_of_samples):
            yield self[i]
