"""Common methods for testing models."""

import tempfile

from typing import Callable

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.model import ClassificationModel, PersistableClassificationModel
from slub_docsa.data.artificial.simple import get_static_mini_dataset
from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets, unique_subject_order


def fit_model_with_mini_dataset(model: ClassificationModel):
    """Load the mini dataset and fit the provided model with it."""
    dataset = get_static_mini_dataset()
    my_subject_order = unique_subject_order(dataset.subjects)
    incidence_matrix = subject_incidence_matrix_from_targets(dataset.subjects, my_subject_order)
    model.fit(dataset.documents, incidence_matrix)
    return model


def mini_dataset_validation_documents():
    """Return a single document as validation document."""
    return [Document(uri="test", title="boring document title")]


def check_model_predicts_non_zero_probabilities(model):
    """Check model returns some scores after fitting to mini dataset."""
    model = fit_model_with_mini_dataset(model)
    probabilties = model.predict_proba(mini_dataset_validation_documents())
    assert np.max(probabilties) > 0  # nosec B101


def check_model_persistence_equal_predictions(model_generator: Callable[[], PersistableClassificationModel]):
    """Check that predictions for a perstiable model are the same after loading it."""
    model = fit_model_with_mini_dataset(model_generator())
    before_probabilities = model.predict_proba(mini_dataset_validation_documents())

    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)

        model = model_generator()
        model.load(directory)
        after_probabilities = model.predict_proba(mini_dataset_validation_documents())

        assert np.allclose(before_probabilities, after_probabilities, )  # nosec B101
