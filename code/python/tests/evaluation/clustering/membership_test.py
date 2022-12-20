"""Test clustering membership methods."""

import numpy as np

from slub_docsa.evaluation.clustering.membership import crips_cluster_assignments_to_membership_matrix
from slub_docsa.evaluation.clustering.membership import is_crisp_cluster_membership
from slub_docsa.evaluation.clustering.membership import membership_matrix_to_crisp_cluster_assignments


def test_is_crisp_cluster_membership():
    """Check whether crisp cluster membership matrices are detected correctly."""
    # check simple crisp membership matrix
    assert is_crisp_cluster_membership(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))

    # checl simple non-crisp membership matrix
    assert not is_crisp_cluster_membership(np.array([[0, 0.5, 0.5], [0.33, 0.33, 0.33], [0, 1, 0]]))


def test_crips_cluster_assignments_conversion_with_membership_matrix():
    """Test conversion between crisp cluster assignments and membership matrices."""
    simple_membership_matrix = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]])
    simple_cluster_assignments = [2, 2, 1, 0, 0]

    assert np.array_equal(
        crips_cluster_assignments_to_membership_matrix(simple_cluster_assignments),
        simple_membership_matrix
    )

    assert np.array_equal(
        simple_cluster_assignments,
        membership_matrix_to_crisp_cluster_assignments(simple_membership_matrix),
    )

    complex_cluster_assignments = [1, 2, 3, 0, 2, 1, 1, 2, 3, 4, 6, 4, 3, 2]

    assert np.array_equal(
        complex_cluster_assignments,
        membership_matrix_to_crisp_cluster_assignments(
            crips_cluster_assignments_to_membership_matrix(complex_cluster_assignments)
        ),
    )
