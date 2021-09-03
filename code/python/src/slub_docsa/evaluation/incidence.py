"""Methods to work with incidence matrices."""

import logging
from typing import Sequence

import numpy as np
from slub_docsa.common.score import IncidenceDecisionFunctionType

from slub_docsa.common.subject import SubjectUriList

logger = logging.getLogger(__name__)


def unique_subject_list(targets: Sequence[SubjectUriList]) -> Sequence[str]:
    """Return the list of unique subjects found in `targets`."""
    subject_set = {uri for uri_list in targets for uri in uri_list}
    return list(subject_set)


def subject_incidence_matrix_from_list(
    targets: Sequence[SubjectUriList],
    subject_order: Sequence[str]
) -> np.ndarray:
    """Return an incidence matrix for the list of subject annotations in `targets`."""
    incidence_matrix = np.zeros((len(targets), len(subject_order)))
    for i, uri_list in enumerate(targets):
        for uri in uri_list:
            if uri in subject_order:
                incidence_matrix[i, subject_order.index(uri)] = 1
            else:
                logger.warning("subject '%s' not given in subject_order list (maybe only in test data)", uri)
    return incidence_matrix


def subject_list_from_incidence_matrix(
    incidence_matrix: np.ndarray,
    subject_order: Sequence[str]
) -> Sequence[SubjectUriList]:
    """Return subject list for incidence matrix given subject ordering."""
    incidence_array = np.array(incidence_matrix)

    if incidence_array.shape[1] != len(subject_order):
        raise ValueError("indicence matrix has %d columns but and subject order has %d entries" % (
            incidence_array.shape[1],
            len(subject_order)
        ))

    return list(map(lambda l: list(map(lambda i: subject_order[i], np.where(l == 1)[0])), incidence_array))


def threshold_incidence_decision(threshold: float = 0.5) -> IncidenceDecisionFunctionType:
    """Select subjects based on a threshold over probabilities."""
    def _decision(probabilities: np.ndarray) -> np.ndarray:
        return np.array(np.where(probabilities >= threshold, 1, 0))

    return _decision


def top_k_incidence_decision(k: int = 3) -> IncidenceDecisionFunctionType:
    """Select k subjects with highest probabilities."""

    def _decision(probabilities: np.ndarray) -> np.ndarray:
        incidence = np.zeros(probabilities.shape)
        x_indices = np.repeat(np.arange(0, incidence.shape[0]), k)
        y_indices = np.argpartition(probabilities, -k)[:, -k:].flatten()
        incidence[x_indices, y_indices] = 1
        return incidence

    return _decision
