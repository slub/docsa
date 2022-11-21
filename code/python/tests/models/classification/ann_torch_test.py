"""Tests for ANN models based on torch."""

from slub_docsa.data.preprocess.vectorizer import TfidfVectorizer
from slub_docsa.models.classification.ann_torch import TorchSingleLayerDenseTanhModel

from .common import check_model_persistence_equal_predictions, check_model_predicts_non_zero_probabilities


def test_simple_ann_torch_model():
    """Test fitting and predicting with a basic torch model."""
    check_model_predicts_non_zero_probabilities(
        TorchSingleLayerDenseTanhModel(vectorizer=TfidfVectorizer(max_features=100))
    )


def test_simple_ann_torch_persistence():
    """Check that predictions for Annif Omikuji model are the same after persistence."""
    check_model_persistence_equal_predictions(
        lambda: TorchSingleLayerDenseTanhModel(vectorizer=TfidfVectorizer(max_features=100))
    )
