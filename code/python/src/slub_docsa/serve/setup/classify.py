"""Setup classification models for REST service."""

from typing import Callable, Mapping, NamedTuple, Sequence

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from slub_docsa.common.document import Document
from slub_docsa.common.model import PersistableClassificationModel
from slub_docsa.data.preprocess.vectorizer import TfidfVectorizer
from slub_docsa.models.classification.scikit import ScikitClassifier
from slub_docsa.serve.store.models import load_published_classification_model


class ClassificationResult(NamedTuple):
    """A classification result consisting of a score and the predicted subject."""

    score: float
    """A certainty score for the prediction."""

    subject_uri: str
    """The URI of the predicted subject."""


def get_classification_model_type_map():
    """Return a map of classification model types and their generator functions."""
    return {
        "tfidf_10k_knn_k=1": lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=TfidfVectorizer(max_features=10000),
        )
    }


def generate_classification_model_load_and_classify_function(
    model_directories: Mapping[str, str],
    model_types: Mapping[str, Callable[[], PersistableClassificationModel]]
):
    """Generate function that can be used to load any persisted models and classify documents."""
    # remember current classification model by id
    current = None

    def load_and_classify(model_id: str, documents: Sequence[Document], limit: int = 10, threshold: float = 0.0):
        nonlocal current

        # if the requested model id is different to the currently loaded model, load the requested model
        if current is None or model_id != current.info.model_id:
            current = load_published_classification_model(model_directories[model_id], model_types)

        # do actual classification
        probabilities = current.model.predict_proba(documents)
        limit = min(probabilities.shape[1] - 1, limit)
        topk_indexes = np.argpartition(probabilities, limit, axis=1)[:, -limit:]
        topk_probabilities = np.take_along_axis(probabilities, topk_indexes, axis=1)
        topk_sort_indexes = np.argsort(topk_probabilities, axis=1)
        sorted_topk_indexed = np.take_along_axis(topk_indexes, topk_sort_indexes, axis=1)
        sorted_topk_probabilities = np.take_along_axis(topk_probabilities, topk_sort_indexes, axis=1)

        # compile and return results
        return [
            [
                ClassificationResult(score=p, subject_uri=current.subject_order[i])
                for i, p in zip(indexes, probs) if p > threshold
            ] for indexes, probs in zip(sorted_topk_indexed, sorted_topk_probabilities)
        ]

    return load_and_classify
