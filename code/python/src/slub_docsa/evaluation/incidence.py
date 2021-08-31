"""Methods to work with incidence matrices."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def unique_subject_list(targets):
    """Return the list of unique subjects found in `targets`."""
    subject_set = {uri for uri_list in targets for uri in uri_list}
    return list(subject_set)


def subject_incidence_matrix_from_list(targets, subject_order):
    """Return an incidence matrix for the list of subject annotations in `targets`."""
    incidence_matrix = np.zeros((len(targets), len(subject_order)))
    for i, uri_list in enumerate(targets):
        for uri in uri_list:
            if uri in subject_order:
                incidence_matrix[i, subject_order.index(uri)] = 1
            else:
                logger.warning("subject '%s' not given in subject_order list (maybe only in test data)", uri)
    return incidence_matrix


def subject_list_from_incidence_matrix(incidence_matrix, subject_order):
    """Return subject list for incidence matrix given subject ordering."""
    incidence_array = np.array(incidence_matrix)

    if incidence_array.shape[1] != len(subject_order):
        raise ValueError("indicence matrix has %d columns but and subject order has %d entries" % (
            incidence_array.shape[1],
            len(subject_order)
        ))

    return list(map(lambda l: list(map(lambda i: subject_order[i], np.where(l == 1)[0])), incidence_array))
