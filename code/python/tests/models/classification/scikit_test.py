"""Testing Scikit Models."""

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from slub_docsa.data.preprocess.vectorizer import TfidfVectorizer

from slub_docsa.evaluation.classification.incidence import subject_incidence_matrix_from_targets, unique_subject_order
from slub_docsa.data.artificial.simple import get_static_mini_dataset
from slub_docsa.models.classification.scikit import ScikitClassifier

from .common import check_model_predicts_non_zero_probabilities, check_model_persistence_equal_predictions


def _create_rforest_model():
    return ScikitClassifier(
        predictor=RandomForestClassifier(n_estimators=10, max_leaf_nodes=100),
        vectorizer=TfidfVectorizer()
    )


def test_scikit_tfidf_model_nearest_neighbor_overfitting():
    """Test that nearest neighbor model overfits for training data."""
    dataset = get_static_mini_dataset()
    model = ScikitClassifier(predictor=KNeighborsClassifier(n_neighbors=1), vectorizer=TfidfVectorizer())

    subject_order = unique_subject_order(dataset.subjects)
    incidence_matrix = subject_incidence_matrix_from_targets(dataset.subjects, subject_order)

    model.fit(dataset.documents, incidence_matrix)
    predicted_probabilities = model.predict_proba(dataset.documents)

    assert np.array_equal(incidence_matrix, predicted_probabilities)


def test_scikit_rforest_model():
    """Check that random forest model return some scores."""
    check_model_predicts_non_zero_probabilities(_create_rforest_model())


def test_scikit_rforest_model_persistence():
    """Check that random forest model reports same scores after being loaded."""
    check_model_persistence_equal_predictions(_create_rforest_model)
