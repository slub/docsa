"""Tests for ANN models based on torch."""

from slub_docsa.data.preprocess.vectorizer import TfidfVectorizer, WordpieceVectorizer
from slub_docsa.models.classification.ann.bert import TorchBertModel
from slub_docsa.models.classification.ann.dense import TorchSingleLayerDenseTanhModel

from .common import check_model_persistence_equal_predictions, check_model_predicts_non_zero_probabilities


def test_simple_ann_torch_model():
    """Test fitting and predicting with a basic torch model."""
    check_model_predicts_non_zero_probabilities(
        TorchSingleLayerDenseTanhModel(vectorizer=TfidfVectorizer(max_features=100), max_epochs=5)
    )


def test_simple_ann_torch_persistence():
    """Check that predictions for Annif Omikuji model are the same after persistence."""
    check_model_persistence_equal_predictions(
        lambda: TorchSingleLayerDenseTanhModel(vectorizer=TfidfVectorizer(max_features=100), max_epochs=5)
    )


def test_bert_ann_torch_model():
    """Test fitting and predicting with a basic torch model."""
    check_model_predicts_non_zero_probabilities(
        TorchBertModel(
            vectorizer=WordpieceVectorizer("en", 1000, 32, use_wikipedia_texts=False),
            max_epochs=5,
            dataloader_workers=0
        )
    )


def test_bert_ann_torch_persistence():
    """Check that predictions for Annif Omikuji model are the same after persistence."""
    check_model_persistence_equal_predictions(
        lambda: TorchBertModel(
            vectorizer=WordpieceVectorizer("en", 1000, 32, use_wikipedia_texts=False),
            max_epochs=5,
            dataloader_workers=0
        )
    )
