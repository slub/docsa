"""Common Tests."""

from slub_docsa.common.paths import DATA_DIR


def test_trivial():
    """Test a trivial assertion."""
    # pylint: disable=comparison-with-itself
    assert True is True


def test_data_dir_not_empty():
    """Test that data dir is not empty."""
    assert DATA_DIR
