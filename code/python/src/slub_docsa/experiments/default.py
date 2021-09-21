"""Provides common defaults for experimentation."""

# pylint: disable=too-many-arguments

import os
import logging

from typing import Iterable, List, Sequence, Tuple, cast

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.metrics import mean_squared_error

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
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.evaluation.incidence import threshold_incidence_decision, positive_top_k_incidence_decision
from slub_docsa.evaluation.incidence import unique_subject_order
from slub_docsa.evaluation.plotting import score_matrix_box_plot
from slub_docsa.evaluation.score import scikit_incidence_metric
from slub_docsa.models.natlibfi_annif import AnnifModel
from slub_docsa.models.oracle import OracleModel
from slub_docsa.models.scikit import ScikitTfidfClassifier, ScikitTfidiRandomClassifier
from slub_docsa.common.model import Model
from slub_docsa.common.dataset import Dataset
from slub_docsa.evaluation.pipeline import evaluate_dataset

logger = logging.getLogger(__name__)

ANNIF_PROJECT_DATA_DIR = os.path.join(ANNIF_DIR, "testproject")


def default_named_models(
    language: str,
    subject_order: Sequence[str] = None,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType] = None,
    model_name_subset: Iterable[str] = None
) -> Tuple[List[str], List[Model]]:
    """Return a list of default models to use for evaluating model performance."""
    models = [
        ("random", ScikitTfidiRandomClassifier()),
        ("stratified", ScikitTfidfClassifier(predictor=DummyClassifier(strategy="stratified"))),
        ("oracle", OracleModel()),
        ("knn k=1", ScikitTfidfClassifier(predictor=KNeighborsClassifier(n_neighbors=1))),
        ("knn k=3", ScikitTfidfClassifier(predictor=KNeighborsClassifier(n_neighbors=3))),
        ("dtree", ScikitTfidfClassifier(predictor=DecisionTreeClassifier(max_depth=7))),
        ("rforest", ScikitTfidfClassifier(predictor=RandomForestClassifier(n_jobs=-1, max_depth=6))),
        ("mlp", ScikitTfidfClassifier(predictor=MLPClassifier(max_iter=10))),
        ("log_reg", ScikitTfidfClassifier(predictor=MultiOutputClassifier(estimator=LogisticRegression()))),
        ("nbayes", ScikitTfidfClassifier(predictor=MultiOutputClassifier(estimator=GaussianNB()))),
        ("svc", ScikitTfidfClassifier(predictor=MultiOutputClassifier(
            estimator=CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)
        ))),
        ("annif tfidf", AnnifModel(model_type="tfidf", language=language)),
        ("annif svc", AnnifModel(model_type="svc", language=language)),
        ("annif fasttext", AnnifModel(model_type="fasttext", language=language)),
        ("annif omikuji", AnnifModel(model_type="omikuji", language=language)),
        ("annif vw_multi", AnnifModel(model_type="vw_multi", language=language)),
        ("annif mllm", AnnifModel(
            model_type="mllm", language=language, subject_order=subject_order, subject_hierarchy=subject_hierarchy
        )),
        ("annif yake", AnnifModel(
            model_type="yake", language=language, subject_order=subject_order, subject_hierarchy=subject_hierarchy
        )),
        ("annif stwfsa", AnnifModel(
            model_type="stwfsa", language=language, subject_order=subject_order, subject_hierarchy=subject_hierarchy
        )),
    ]

    if model_name_subset is not None:
        models = list(filter(lambda i: i[0] in model_name_subset, models))

    model_names, model_classes = list(zip(*models))
    model_names = cast(List[str], model_names)
    model_classes = cast(List[Model], model_classes)

    return model_names, model_classes


def default_named_scores(
    score_name_subset: Iterable[str] = None
) -> Tuple[List[str], List[Tuple[float, float]], List[ScoreFunctionType]]:
    """Return a list of default score functions for evaluation."""
    scores = [
        ("t=0.5 accuracy", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            accuracy_score
        )),
        ("top3 accuracy", [0, 1], scikit_incidence_metric(
            positive_top_k_incidence_decision(3),
            accuracy_score
        )),
        ("t=0.5 f1_score micro", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            f1_score,
            average="micro",
            zero_division=0
        )),
        ("top3 f1_score micro", [0, 1], scikit_incidence_metric(
            positive_top_k_incidence_decision(3),
            f1_score,
            average="micro",
            zero_division=0
        )),
        ("t=0.5 precision micro", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            precision_score,
            average="micro",
            zero_division=0
        )),
        ("top3 precision micro", [0, 1], scikit_incidence_metric(
            positive_top_k_incidence_decision(3),
            precision_score,
            average="micro",
            zero_division=0
        )),
        ("t=0.5 recall micro", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            recall_score,
            average="micro",
            zero_division=0
        )),
        ("top3 recall micro", [0, 1], scikit_incidence_metric(
            positive_top_k_incidence_decision(3),
            recall_score,
            average="micro",
            zero_division=0
        )),
        ("roc auc micro", [0, 1], lambda t, p: roc_auc_score(t, p, average="micro")),
        ("log loss", [0, None], log_loss),
        ("mean squared error", [0, None], mean_squared_error)
    ]

    if score_name_subset is not None:
        scores = list(filter(lambda i: i[0] in score_name_subset, scores))

    score_names, score_ranges, score_functions = list(zip(*scores))
    score_names = cast(List[str], score_names)
    score_ranges = cast(List[Tuple[float, float]], score_ranges)
    score_functions = cast(List[ScoreFunctionType], score_functions)

    return score_names, score_ranges, score_functions


def do_default_box_plot_evaluation(
    dataset: Dataset,
    language: str,
    box_plot_filepath: str,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType] = None,
    model_name_subset: Iterable[str] = None,
    score_name_subset: Iterable[str] = None,
):
    """Do 10-fold cross validation for default models and scores and save box plot."""
    # define subject ordering
    subject_order = unique_subject_order(dataset.subjects)

    logger.debug("subject order: %s", subject_order)

    # setup models and scores
    model_names, model_classes = default_named_models(
        language,
        model_name_subset=model_name_subset,
        subject_order=subject_order,
        subject_hierarchy=subject_hierarchy

    )
    score_names, score_ranges, score_functions = default_named_scores(score_name_subset)

    # do evaluate
    score_matrix = evaluate_dataset(
        n_splits=10,
        dataset=dataset,
        subject_order=subject_order,
        models=model_classes,
        score_functions=score_functions,
        random_state=0
    )

    # generate figure
    score_matrix_box_plot(
        score_matrix,
        model_names,
        score_names,
        score_ranges,
        columns=2
    ).write_html(
        box_plot_filepath,
        include_plotlyjs="cdn",
        # default_height=f"{len(score_names) * 500}px"
    )
