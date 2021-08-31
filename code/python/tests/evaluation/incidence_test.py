"""Tests for incidence matrix processing"""

import numpy as np
import pytest

from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_list, subject_list_from_incidence_matrix

example_subject_order = [
    "uri://subject1",
    "uri://subject2",
    "uri://subject3"
]

example_subject_list = [
    ["uri://subject2"],
    ["uri://subject1", "uri://subject2", "uri://subject3"],
    ["uri://subject2", "uri://subject3"],
    [],
]

example_incidence_matrix = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 1],
    [0, 0, 0]
]


def test_subject_incidence_matrix_from_list_example():
    """Test correct subject index matrix for example."""
    np.testing.assert_array_equal(
        subject_incidence_matrix_from_list(example_subject_list, example_subject_order),
        example_incidence_matrix
    )


def test_subject_list_from_incidence_matrix_example():
    """Test correct subject list from incidence matrix for example."""
    assert subject_list_from_incidence_matrix(example_incidence_matrix, example_subject_order) == example_subject_list


def test_subject_list_from_incidence_matrix_compatible_shape():
    """Test incidence matrix and subject ordering have compatible size."""
    with pytest.raises(ValueError):
        subject_list_from_incidence_matrix(np.zeros((10, 4)), ["a", "b", "c"])
