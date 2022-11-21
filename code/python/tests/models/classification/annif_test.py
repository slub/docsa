"""Annif model tests."""

import tempfile

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.data.artificial.simple import get_static_mini_dataset
from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets, unique_subject_order
from slub_docsa.models.classification.natlibfi_annif import AnnifModel


def _annif_fit_with_mini_dataset(model: AnnifModel):
    dataset = get_static_mini_dataset()
    my_subject_order = unique_subject_order(dataset.subjects)
    incidence_matrix = subject_incidence_matrix_from_targets(dataset.subjects, my_subject_order)
    model.fit(dataset.documents, incidence_matrix)
    return model


def _get_validation_documents():
    return [Document(uri="test", title="boring document title")]


def test_annif_omikuji_model():
    """Create Annif omikuji model and test model returns some scores."""
    model = _annif_fit_with_mini_dataset(AnnifModel("omikuji", "en"))
    probabilties = model.predict_proba(_get_validation_documents())
    assert np.min(probabilties) > 0


def test_annif_tfidf_model():
    """Create Annif omikuji model and test model returns some scores."""
    model = _annif_fit_with_mini_dataset(AnnifModel("tfidf", "en"))
    probabilties = model.predict_proba(_get_validation_documents())
    assert np.min(probabilties) > 0


def test_annif_omikuji_persistence():
    """Check that predictions for Annif Omikuji model are the same after persistence."""
    model = _annif_fit_with_mini_dataset(AnnifModel("omikuji", "en"))
    before_probabilities = model.predict_proba(_get_validation_documents())

    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)

        model = AnnifModel("omikuji", "en")
        model.load(directory)
        after_probabilities = model.predict_proba(_get_validation_documents())

        assert np.array_equal(before_probabilities, after_probabilities)
