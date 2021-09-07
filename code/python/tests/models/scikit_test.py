"""Testing Scikit Models."""

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_list, unique_subject_list
from slub_docsa.evaluation.data import get_static_mini_dataset
from slub_docsa.models.scikit import ScikitTfidfClassifier


def test_scikit_tfidf_model_nearest_neighbor_overfitting():
    """Test that nearest neighbor model overfits for training data."""
    dataset = get_static_mini_dataset()
    model = ScikitTfidfClassifier(predictor=KNeighborsClassifier(n_neighbors=1))

    subject_order = unique_subject_list(dataset.subjects)
    incidence_matrix = subject_incidence_matrix_from_list(dataset.subjects, subject_order)

    model.fit(dataset.documents, incidence_matrix)
    predicted_probabilities = model.predict_proba(dataset.documents)

    assert np.array_equal(incidence_matrix, predicted_probabilities)
