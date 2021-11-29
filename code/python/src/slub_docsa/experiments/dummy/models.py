"""Default dummy models that can be used for experimentation."""

# pylint: disable=unnecessary-lambda

from sklearn.dummy import DummyClassifier

from slub_docsa.data.preprocess.vectorizer import RandomVectorizer
from slub_docsa.models.classification.dummy import NihilisticModel, OracleModel, RandomModel
from slub_docsa.models.classification.scikit import ScikitClassifier

from slub_docsa.experiments.common.models import NamedModelTupleList


def default_dummy_named_model_list() -> NamedModelTupleList:
    """Return a list of common dummy models."""
    models = [
        ("random", lambda: RandomModel()),
        ("nihilistic", lambda: NihilisticModel()),
        ("stratified", lambda: ScikitClassifier(
            predictor=DummyClassifier(strategy="stratified"),
            vectorizer=RandomVectorizer(),
        )),
        ("oracle", lambda: OracleModel()),
    ]

    return models
