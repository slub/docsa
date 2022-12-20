"""Common Tests."""

from slub_docsa.common.paths import get_data_dir


def test_trivial():
    """Test a trivial assertion."""
    # pylint: disable=comparison-with-itself
    assert True


def test_data_dir_not_empty():
    """Test that data dir is not empty."""
    assert get_data_dir()
