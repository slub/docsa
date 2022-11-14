"""Setup classic classification models."""

from sklearn.neighbors import KNeighborsClassifier

from slub_docsa.data.preprocess.vectorizer import TfidfVectorizer
from slub_docsa.models.classification.scikit import ScikitClassifier


def get_classic_classification_models_map():
    """Return a map of classification model types and their generator functions."""
    return {
        "tfidf_10k_knn_k=1": lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=TfidfVectorizer(max_features=10000),
        )
    }
