"""Test subject preprocess methods."""

import pytest

from slub_docsa.common.subject import SubjectHierarchy, SubjectNode
from slub_docsa.data.preprocess.subject import subject_ancestors_list, subject_label_breadcrumb
from slub_docsa.data.preprocess.subject import prune_subject_uri_to_level, prune_subject_uris_to_level
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level

example_subject_hierarchy: SubjectHierarchy[SubjectNode] = {
    "uri://subject1": SubjectNode(uri="uri://subject1", label="subject1", parent_uri=None),
    "uri://subject2": SubjectNode(uri="uri://subject2", label="subject2", parent_uri=None),
    "uri://subject3": SubjectNode(uri="uri://subject3", label="subject3", parent_uri="uri://subject1"),
    "uri://subject4": SubjectNode(uri="uri://subject4", label="subject4", parent_uri="uri://subject1"),
    "uri://subject5": SubjectNode(uri="uri://subject5", label="subject5", parent_uri="uri://subject3"),
    "uri://subject6": SubjectNode(uri="uri://subject6", label="subject6", parent_uri="uri://subject3"),
}


def test_subject_ancestors_list():
    """Verify ancestors list is extracted correctly from hierarchy."""
    root_node = example_subject_hierarchy["uri://subject1"]
    subject_node = example_subject_hierarchy["uri://subject3"]
    ancestor_list = [root_node, subject_node]
    assert subject_ancestors_list(subject_node, example_subject_hierarchy) == ancestor_list


def test_subject_label_breadcrumb():
    """Check label breadcrumb is correct."""
    subject_node = example_subject_hierarchy["uri://subject5"]
    assert subject_label_breadcrumb(subject_node, example_subject_hierarchy) == "subject1 | subject3 | subject5"


def test_prune_subject_uri_to_level():
    """Check pruning of subject uri to level."""
    assert prune_subject_uri_to_level(1, "uri://subject5", example_subject_hierarchy) == "uri://subject1"
    assert prune_subject_uri_to_level(2, "uri://subject5", example_subject_hierarchy) == "uri://subject3"
    assert prune_subject_uri_to_level(3, "uri://subject5", example_subject_hierarchy) == "uri://subject5"
    assert prune_subject_uri_to_level(4, "uri://subject5", example_subject_hierarchy) == "uri://subject5"
    assert prune_subject_uri_to_level(3, "uri://subject1", example_subject_hierarchy) == "uri://subject1"

    with pytest.raises(ValueError):
        prune_subject_uri_to_level(-1, "uri://subject1", example_subject_hierarchy)
    with pytest.raises(ValueError):
        prune_subject_uri_to_level(1, "uri://does-not-exist", example_subject_hierarchy)


def test_prune_subject_uris_to_level():
    """Check pruning of list of subject uris to level."""
    subject_list = ["uri://subject5", "uri://subject6"]
    assert prune_subject_uris_to_level(1, subject_list, example_subject_hierarchy) == ["uri://subject1"]
    assert prune_subject_uris_to_level(2, subject_list, example_subject_hierarchy) == ["uri://subject3"]
    assert set(prune_subject_uris_to_level(3, subject_list, example_subject_hierarchy)) == set(subject_list)


def test_prune_subject_targets_to_level():
    """Check pruning of subject targets to level."""
    subject_targets = [
        ["uri://subject1"],
        ["uri://subject2", "uri://subject6"],
        ["uri://subject3", "uri://subject4"],
    ]
    true_targets_1 = [
        ["uri://subject1"],
        ["uri://subject1", "uri://subject2"],
        ["uri://subject1"],
    ]
    true_targets_2 = [
        ["uri://subject1"],
        ["uri://subject2", "uri://subject3"],
        ["uri://subject3", "uri://subject4"],
    ]
    pruned_targets_1 = prune_subject_targets_to_level(1, subject_targets, example_subject_hierarchy)
    pruned_targets_2 = prune_subject_targets_to_level(2, subject_targets, example_subject_hierarchy)

    for pruned_targets, true_targets in zip([pruned_targets_1, pruned_targets_2], [true_targets_1, true_targets_2]):
        for pruned_subject_list, true_subject_list in zip(pruned_targets, true_targets):
            assert set(pruned_subject_list) == set(true_subject_list)
