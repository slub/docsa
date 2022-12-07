"""Methods to tranform cluster membership matrices."""

from typing import Sequence, cast
import numpy as np


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
