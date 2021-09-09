"""Tests for incidence matrix processing."""

import numpy as np
import pytest

from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets
from slub_docsa.evaluation.incidence import subject_targets_from_incidence_matrix
from slub_docsa.evaluation.incidence import threshold_incidence_decision, top_k_incidence_decision

example_subject_order = [
    "uri://subject1",
    "uri://subject2",
    "uri://subject3"
]

example_subject_targets = [
    ["uri://subject2"],
    ["uri://subject1", "uri://subject2", "uri://subject3"],
    ["uri://subject2", "uri://subject3"],
    [],
]

example_incidence_matrix = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 1],
    [0, 0, 0]
])


def test_subject_incidence_matrix_from_list_example():
    """Test correct subject index matrix for example."""
    np.testing.assert_array_equal(
        subject_incidence_matrix_from_targets(example_subject_targets, example_subject_order),
        example_incidence_matrix
    )


def test_subject_list_from_incidence_matrix_example():
    """Test correct subject list from incidence matrix for example."""
    subject_targets = subject_targets_from_incidence_matrix(example_incidence_matrix, example_subject_order)
    assert subject_targets == example_subject_targets


def test_subject_list_from_incidence_matrix_compatible_shape():
    """Test incidence matrix and subject ordering have compatible size."""
    with pytest.raises(ValueError):
        subject_targets_from_incidence_matrix(np.zeros((10, 4)), ["a", "b", "c"])


def test_threshold_incidence_decision():
    """Test threshold based incidence decision function."""
    probabilities = np.array([
        [0.0, 0.5, 1.0],
        [0.2, 0.6, 0.8],
        [1.0, 1.0, 0.2]
    ])

    incidence_00 = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    incidence_05 = np.array([
        [0, 1, 1],
        [0, 1, 1],
        [1, 1, 0]
    ])

    incidence_10 = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0]
    ])

    assert np.array_equal(incidence_00, threshold_incidence_decision(0.0)(probabilities))
    assert np.array_equal(incidence_05, threshold_incidence_decision(0.5)(probabilities))
    assert np.array_equal(incidence_10, threshold_incidence_decision(1.0)(probabilities))


def test_top_k_incidence_decision():
    """Test top k based incidence decision function."""
    probabilities = np.array([
        [0.8, 0.5, 0.7],
        [0.2, 0.1, 0.0],
        [0.3, 0.2, 0.9],
    ])

    incidence_top1 = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])

    incidence_top2 = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
    ])

    assert np.array_equal(incidence_top1, top_k_incidence_decision(1)(probabilities))
    assert np.array_equal(incidence_top2, top_k_incidence_decision(2)(probabilities))
