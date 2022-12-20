"""Default dummy models that can be used for experimentation."""

# pylint: disable=unnecessary-lambda

from sklearn.dummy import DummyClassifier

from slub_docsa.data.preprocess.vectorizer import RandomVectorizer
from slub_docsa.models.classification.dummy import NihilisticModel, OracleModel, RandomModel
from slub_docsa.models.classification.scikit import ScikitClassifier

from slub_docsa.serve.common import ModelTypeMapping


def default_dummy_model_types() -> ModelTypeMapping:
    """Return a list of common dummy models."""
    models = {
        "random": lambda subject_hierarchy, subject_order: RandomModel(),
        "nihilistic": lambda subject_hierarchy, subject_order: NihilisticModel(),
        "stratified": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=DummyClassifier(strategy="stratified"),
            vectorizer=RandomVectorizer(),
        ),
        "oracle": lambda subject_hierarchy, subject_order: OracleModel(),
    }

    return models
