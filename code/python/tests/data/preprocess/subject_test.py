"""Test subject preprocess methods."""

import pytest

from slub_docsa.common.subject import SubjectTuple
from slub_docsa.data.preprocess.subject import build_subject_hierarchy_from_subject_tuples, subject_ancestors_list
from slub_docsa.data.preprocess.subject import prune_subject_uri_to_level, prune_subject_uris_to_level
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level, subject_label_breadcrumb_as_string


def get_example_subject_hierarchy():
    """Generate example subject hierarchy."""
    return build_subject_hierarchy_from_subject_tuples([
        SubjectTuple("uri://subject1", {"en": "subject1"}, None, None),
        SubjectTuple("uri://subject2", {"en": "subject2"}, None, None),
        SubjectTuple("uri://subject3", {"en": "subject3"}, "uri://subject1", None),
        SubjectTuple("uri://subject4", {"en": "subject4"}, "uri://subject1", None),
        SubjectTuple("uri://subject5", {"en": "subject5"}, "uri://subject3", None),
        SubjectTuple("uri://subject6", {"en": "subject6"}, "uri://subject3", None),
    ])


def test_subject_ancestors_list():
    """Verify ancestors list is extracted correctly from hierarchy."""
    parent_uri = "uri://subject1"
    subject_uri = "uri://subject3"
    ancestor_list = [parent_uri, subject_uri]
    assert subject_ancestors_list(subject_uri, get_example_subject_hierarchy()) == ancestor_list


def test_subject_label_breadcrumb():
    """Check label breadcrumb is correct."""
    subject_uri = "uri://subject5"
    subject_hierarchy = get_example_subject_hierarchy()
    breadcrumb = subject_label_breadcrumb_as_string(subject_uri, "en", subject_hierarchy)
    assert breadcrumb == ["subject1", "subject3", "subject5"]


def test_prune_subject_uri_to_level():
    """Check pruning of subject uri to level."""
    subject_hierarchy = get_example_subject_hierarchy()
    assert prune_subject_uri_to_level(1, "uri://subject5", subject_hierarchy) == "uri://subject1"
    assert prune_subject_uri_to_level(2, "uri://subject5", subject_hierarchy) == "uri://subject3"
    assert prune_subject_uri_to_level(3, "uri://subject5", subject_hierarchy) == "uri://subject5"
    assert prune_subject_uri_to_level(4, "uri://subject5", subject_hierarchy) == "uri://subject5"
    assert prune_subject_uri_to_level(3, "uri://subject1", subject_hierarchy) == "uri://subject1"

    with pytest.raises(ValueError):
        prune_subject_uri_to_level(-1, "uri://subject1", subject_hierarchy)
    with pytest.raises(ValueError):
        prune_subject_uri_to_level(1, "uri://does-not-exist", subject_hierarchy)


def test_prune_subject_uris_to_level():
    """Check pruning of list of subject uris to level."""
    subject_hierarchy = get_example_subject_hierarchy()
    subject_list = ["uri://subject5", "uri://subject6"]
    assert prune_subject_uris_to_level(1, subject_list, subject_hierarchy) == ["uri://subject1"]
    assert prune_subject_uris_to_level(2, subject_list, subject_hierarchy) == ["uri://subject3"]
    assert set(prune_subject_uris_to_level(3, subject_list, subject_hierarchy)) == set(subject_list)


def test_prune_subject_targets_to_level():
    """Check pruning of subject targets to level."""
    subject_hierarchy = get_example_subject_hierarchy()
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
    pruned_targets_1 = prune_subject_targets_to_level(1, subject_targets, subject_hierarchy)
    pruned_targets_2 = prune_subject_targets_to_level(2, subject_targets, subject_hierarchy)

    for pruned_targets, true_targets in zip([pruned_targets_1, pruned_targets_2], [true_targets_1, true_targets_2]):
        for pruned_subject_list, true_subject_list in zip(pruned_targets, true_targets):
            assert set(pruned_subject_list) == set(true_subject_list)
