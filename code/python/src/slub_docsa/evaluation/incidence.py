"""Methods to work with incidence matrices."""

import logging
from typing import Sequence

import numpy as np
from slub_docsa.common.score import IncidenceDecisionFunctionType

from slub_docsa.common.subject import SubjectTargets, SubjectUriList

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
    incidence_matrix = np.zeros((len(targets), len(subject_order)))
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
        raise ValueError("indicence matrix has %d columns but and subject order has %d entries" % (
            incidence_array.shape[1],
            len(subject_order)
        ))

    return list(map(lambda l: list(map(lambda i: subject_order[i], np.where(l == 1)[0])), incidence_array))


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
