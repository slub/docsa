"""Methods to work with incidence matrices."""

import logging
from typing import List, Sequence, cast

import numpy as np
from slub_docsa.common.score import IncidenceDecisionFunctionType

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
    incidence_matrix = np.zeros((len(targets), len(subject_order)), dtype=np.int32)
    for i, uri_list in enumerate(targets):
        for uri in uri_list:
            if uri in subject_order:
                incidence_matrix[i, subject_order.index(uri)] = 1
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

    return list(map(lambda l: list(map(lambda i: subject_order[i], np.where(l == 1)[0])), incidence_array))


def subject_idx_from_incidence_matrix(
    incidence_matrix: np.ndarray,
):
    """Return subject indexes for an incidence matrix.

    Parameters
    ----------
    incidence_matrix: np.ndarray
        An incidence matrix representing which document is classified by which document.

    Returns
    -------
    Sequence[Iterable[int]]
        An order list of lists of subjects indexes. Each list entry corresponds to a row of `incidence_matrix`,
        which contains the list of indexes (columns) that equals to 1 for the corresponding element in
        `incidence_matrix`.
    """
    incidence_array = np.array(incidence_matrix)
    return list(map(lambda l: list(np.where(l == 1)[0]), incidence_array))


def threshold_incidence_decision(threshold: float = 0.5) -> IncidenceDecisionFunctionType:
    """Select subjects based on a threshold over probabilities.

    Parameters
    ----------
    threshold: float
        the minimum probability required for a subject to be chosen

    Returns
    -------
    Callback[[np.ndarray], np.ndarray]
        A functions that transforms a numpy array of subject probabilities to an incidence matrix of the
        same shape representing which subjects are chosen for which documents.
    """
    def _decision(probabilities: np.ndarray) -> np.ndarray:
        return np.array(np.where(probabilities >= threshold, 1, 0))

    return _decision


def top_k_incidence_decision(k: int = 3) -> IncidenceDecisionFunctionType:
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

    def _decision(probabilities: np.ndarray) -> np.ndarray:
        incidence = np.zeros(probabilities.shape)
        x_indices = np.repeat(np.arange(0, incidence.shape[0]), k)
        y_indices = np.argpartition(probabilities, -k)[:, -k:].flatten()
        incidence[x_indices, y_indices] = 1
        return incidence

    return _decision


def positive_top_k_incidence_decision(k: int = 3) -> IncidenceDecisionFunctionType:
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

    def _decision(probabilities: np.ndarray) -> np.ndarray:
        incidence = np.zeros(probabilities.shape)
        x_indices = np.repeat(np.arange(0, incidence.shape[0]), k)
        y_indices = np.argpartition(probabilities, -k)[:, -k:].flatten()
        incidence[x_indices, y_indices] = 1
        incidence[probabilities <= 0.0] = 0
        return incidence

    return _decision


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


def is_crisp_cluster_membership(membership: np.ndarray) -> bool:
    """Check wheter a clustering membership matrix has only crisp assignments.

    Meaning, each document is assigned to exactly one cluster with a membership degree of 1.

    Parameters
    ----------
    membership: numpy.ndarray
        the membership matrix to check

    Returns
    -------
    bool
        True if the membership matrix represents a crisp cluster assignment
    """
    if np.max(membership) > 1.0:
        # there is a value larger than one, which is not allowed
        raise ValueError("membership matrix has value larger than 1")

    if np.min(membership) < 0.0:
        # there is avlue smaller than zero, which is not allowed
        raise ValueError("membership matrix has value smaller than 0")

    if not cast(np.ndarray, ((membership == 0) | (membership == 1))).all():
        # there are values that are not zeros or ones
        return False

    if not cast(np.ndarray, (np.sum(membership, axis=1) == 1)).all():
        # there are rows that do not sum up to 1
        return False

    return True


def crips_cluster_assignments_to_membership_matrix(
    assignments: Sequence[int],
) -> np.ndarray:
    """Convert crisp cluster assignemnts to a cluster membership matrix.

    Parameters
    ----------
    assignments: Sequence[int]
        cluster assignments as list of cluster indices

    Returns
    -------
    numpy.ndarray
        membership matrix as simple binary matrix representing cluster assignemnts
    """
    n_documents, n_clusters = len(assignments), np.max(assignments) + 1

    # initialize membership matrix
    membership = np.zeros((n_documents, n_clusters))

    # set membership in matrix
    membership[list(range(n_documents)), assignments] = 1

    return membership


def membership_matrix_to_crisp_cluster_assignments(
    membership: np.ndarray,
) -> Sequence[int]:
    """Covert membership matrix to a crisp cluster assignment list.

    Parameters
    ----------
    membership: numpy.ndarray
        the membership matrix representing a crisp clustering

    Returns
    -------
    Sequence[int]
        a list of cluster assignments (column indexes from membership matrix where values equal 1)
    """
    if not is_crisp_cluster_membership(membership):
        raise ValueError("can not determine crisp cluster assignemnts for non-crisp membership")
    return cast(np.ndarray, np.where(membership == 1)[1]).tolist()
