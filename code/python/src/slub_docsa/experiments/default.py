"""Provides common defaults for experimentation."""

import os
from typing import List, Tuple, cast

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from slub_docsa.common.paths import ANNIF_DIR
from slub_docsa.common.score import ScoreFunctionType
from slub_docsa.evaluation.incidence import threshold_incidence_decision, top_k_incidence_decision
from slub_docsa.evaluation.score import scikit_incidence_metric
from slub_docsa.models.natlibfi_annif import AnnifModel
from slub_docsa.models.scikit import ScikitTfidfClassifier
from slub_docsa.common.model import Model


ANNIF_PROJECT_DATA_DIR = os.path.join(ANNIF_DIR, "testproject")


def default_named_models() -> Tuple[List[str], List[Model]]:
    """Return a list of default models to use for evaluating model performance."""
    models = [
        (
            "random",
            ScikitTfidfClassifier(predictor=DummyClassifier(strategy="uniform")),
        ),
        (
            "stratified",
            ScikitTfidfClassifier(predictor=DummyClassifier(strategy="stratified")),
        ),
        (
            "knn k=1",
            ScikitTfidfClassifier(predictor=KNeighborsClassifier(n_neighbors=1))
        ),
        (
            "knn k=3",
            ScikitTfidfClassifier(predictor=KNeighborsClassifier(n_neighbors=3)),
        ),
        (
            "dtree",
            ScikitTfidfClassifier(predictor=DecisionTreeClassifier()),
        ),
        (
            "rforest",
            ScikitTfidfClassifier(predictor=RandomForestClassifier()),
        ),
        (
            "mlp",
            ScikitTfidfClassifier(predictor=MLPClassifier()),
        ),
        (
            "log_reg",
            ScikitTfidfClassifier(predictor=MultiOutputClassifier(estimator=LogisticRegression())),
        ),
        (
            "nbayes",
            ScikitTfidfClassifier(predictor=MultiOutputClassifier(estimator=GaussianNB())),
        ),
        (
            "svc",
            ScikitTfidfClassifier(predictor=MultiOutputClassifier(
                estimator=CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)
            ))
        ),
        (
            "annif tfidf",
            AnnifModel(model_type="tfidf")
        ),
        (
            "annif svc",
            AnnifModel(model_type="svc")
        ),
        (
            "annif fasttext",
            AnnifModel(model_type="fasttext")
        )
    ]

    model_names, model_classes = list(zip(*models))
    model_names = cast(List[str], model_names)
    model_classes = cast(List[Model], model_classes)

    return model_names, model_classes


def default_named_scores() -> Tuple[List[str], List[ScoreFunctionType]]:
    """Return a list of default score functions for evaluation."""
    scores = [
        ("t=0.5 accuracy", scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            accuracy_score
        )),
        ("top3 accuracy", scikit_incidence_metric(
            top_k_incidence_decision(3),
            accuracy_score
        )),
        ("t=0.5 f1_score micro", scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            f1_score,
            average="micro",
            zero_division=0
        )),
        ("top3 f1_score micro", scikit_incidence_metric(
            top_k_incidence_decision(3),
            f1_score,
            average="micro",
            zero_division=0
        )),
        ("t=0.5 precision micro", scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            precision_score,
            average="micro",
            zero_division=0
        )),
        ("top3 precision micro", scikit_incidence_metric(
            top_k_incidence_decision(3),
            precision_score,
            average="micro",
            zero_division=0
        )),
        ("t=0.5 recall micro", scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            recall_score,
            average="micro",
            zero_division=0
        )),
        ("top3 recall micro", scikit_incidence_metric(
            top_k_incidence_decision(3),
            recall_score,
            average="micro",
            zero_division=0
        )),
    ]

    score_names, score_functions = list(zip(*scores))
    score_names = cast(List[str], score_names)
    score_functions = cast(List[ScoreFunctionType], score_functions)

    return score_names, score_functions
