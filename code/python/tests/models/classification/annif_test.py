"""Annif model tests."""

from slub_docsa.models.classification.natlibfi_annif import AnnifModel

from .common import check_model_persistence_equal_predictions, check_model_predicts_non_zero_probabilities


def test_annif_omikuji_model():
    """Create Annif omikuji model and test model returns some scores."""
    check_model_predicts_non_zero_probabilities(AnnifModel("omikuji", "en"))


def test_annif_tfidf_model():
    """Create Annif omikuji model and test model returns some scores."""
    check_model_predicts_non_zero_probabilities(AnnifModel("tfidf", "en"))


def test_annif_omikuji_persistence():
    """Check that predictions for Annif Omikuji model are the same after persistence."""
    check_model_persistence_equal_predictions(lambda: AnnifModel("omikuji", "en"))


def test_annif_tfidf_persistence():
    """Check that predictions for Annif Omikuji model are the same after persistence."""
    check_model_persistence_equal_predictions(lambda: AnnifModel("tfidf", "en"))
