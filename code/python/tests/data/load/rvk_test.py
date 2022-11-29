"""Tests RVK data source."""

from slub_docsa.data.load.subjects.rvk import read_rvk_subjects


def test_rvk_first_level_classes():
    """Check that there are 34 first level classes in RVK."""
    assert len(list(read_rvk_subjects(depth=1))) == 34
