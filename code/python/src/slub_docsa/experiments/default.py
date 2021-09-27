"""Provides common defaults for experimentation."""

# pylint: disable=too-many-arguments, unnecessary-lambda

import os
import logging

from typing import Callable, Iterable, List, NamedTuple, Sequence, Tuple, cast

import numpy as np

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
from slub_docsa.common.score import MultiClassScoreFunctionType, BinaryClassScoreFunctionType
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.evaluation.incidence import threshold_incidence_decision, positive_top_k_incidence_decision
from slub_docsa.evaluation.incidence import unique_subject_order
from slub_docsa.evaluation.plotting import per_subject_precision_recall_vs_samples_plot
from slub_docsa.evaluation.plotting import precision_recall_plot, score_matrix_box_plot
from slub_docsa.evaluation.plotting import per_subject_score_histograms_plot
from slub_docsa.evaluation.score import scikit_incidence_metric
from slub_docsa.models.natlibfi_annif import AnnifModel
from slub_docsa.models.oracle import OracleModel
from slub_docsa.models.scikit import ScikitTfidfClassifier, ScikitTfidiRandomClassifier
from slub_docsa.common.model import Model
from slub_docsa.common.dataset import Dataset
from slub_docsa.evaluation.pipeline import score_models_for_dataset

logger = logging.getLogger(__name__)

ANNIF_PROJECT_DATA_DIR = os.path.join(ANNIF_DIR, "testproject")


class DefaultModelLists(NamedTuple):
    """Stores names and classes for default models."""

    names: List[str]
    classes: List[Model]


class DefaultScoreLists(NamedTuple):
    """Stores names, ranges and functions of default scores (both multi-class and binary)."""

    names: List[str]
    ranges: List[Tuple[float, float]]
    functions: List[Callable]


class DefaultEvaluationResult(NamedTuple):
    """Stores evaluation result matrices as well as model and score info."""

    overall_score_matrix: np.ndarray
    per_class_score_matrix: np.ndarray
    model_lists: DefaultModelLists
    overall_score_lists: DefaultScoreLists
    per_class_score_lists: DefaultScoreLists


def default_named_models(
    language: str,
    subject_order: Sequence[str] = None,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType] = None,
    model_name_subset: Iterable[str] = None
) -> DefaultModelLists:
    """Return a list of default models to use for evaluating model performance."""
    models = [
        ("random", lambda: ScikitTfidiRandomClassifier()),
        ("stratified", lambda: ScikitTfidfClassifier(predictor=DummyClassifier(strategy="stratified"))),
        ("oracle", lambda: OracleModel()),
        ("knn k=1", lambda: ScikitTfidfClassifier(predictor=KNeighborsClassifier(n_neighbors=1))),
        ("knn k=3", lambda: ScikitTfidfClassifier(predictor=KNeighborsClassifier(n_neighbors=3))),
        ("dtree", lambda: ScikitTfidfClassifier(predictor=DecisionTreeClassifier(max_leaf_nodes=1000))),
        ("rforest", lambda: ScikitTfidfClassifier(predictor=RandomForestClassifier(n_jobs=-1, max_leaf_nodes=1000))),
        ("mlp", lambda: ScikitTfidfClassifier(predictor=MLPClassifier(max_iter=10))),
        ("log_reg", lambda: ScikitTfidfClassifier(predictor=MultiOutputClassifier(estimator=LogisticRegression()))),
        ("nbayes", lambda: ScikitTfidfClassifier(predictor=MultiOutputClassifier(estimator=GaussianNB()))),
        ("svc", lambda: ScikitTfidfClassifier(predictor=MultiOutputClassifier(
            estimator=CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)
        ))),
        ("annif tfidf", lambda: AnnifModel(model_type="tfidf", language=language)),
        ("annif svc", lambda: AnnifModel(model_type="svc", language=language)),
        ("annif fasttext", lambda: AnnifModel(model_type="fasttext", language=language)),
        ("annif omikuji", lambda: AnnifModel(model_type="omikuji", language=language)),
        ("annif vw_multi", lambda: AnnifModel(model_type="vw_multi", language=language)),
        ("annif mllm", lambda: AnnifModel(
            model_type="mllm", language=language, subject_order=subject_order, subject_hierarchy=subject_hierarchy
        )),
        ("annif yake", lambda: AnnifModel(
            model_type="yake", language=language, subject_order=subject_order, subject_hierarchy=subject_hierarchy
        )),
        ("annif stwfsa", lambda: AnnifModel(
            model_type="stwfsa", language=language, subject_order=subject_order, subject_hierarchy=subject_hierarchy
        )),
    ]

    if model_name_subset is not None:
        models = list(filter(lambda i: i[0] in model_name_subset, models))

    model_names, model_classes = list(zip(*models))
    model_names = cast(List[str], model_names)
    model_classes = cast(List[Model], [cls() for cls in model_classes])

    return DefaultModelLists(model_names, model_classes)


def default_named_multiclass_scores(
    score_name_subset: Iterable[str] = None
) -> DefaultScoreLists:
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
    score_functions = cast(List[MultiClassScoreFunctionType], score_functions)

    return DefaultScoreLists(score_names, score_ranges, score_functions)


def default_named_binary_scores(
    score_name_subset: Iterable[str] = None
) -> DefaultScoreLists:
    """Return a list of default per-subject score functions for evaluation."""
    scores = [
        ("t=0.5 accuracy", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            accuracy_score
        )),
        ("t=0.5 f1_score", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            f1_score,
            zero_division=0
        )),
        ("t=0.5 precision", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            precision_score,
            zero_division=0
        )),
        ("t=0.5 recall", [0, 1], scikit_incidence_metric(
            threshold_incidence_decision(0.5),
            recall_score,
            zero_division=0
        )),
        ("mean squared error", [0, None], mean_squared_error),
        ("# training samples", [0, None], lambda t, _: len(np.where(t > 0)[0]))
    ]

    if score_name_subset is not None:
        scores = list(filter(lambda i: i[0] in score_name_subset, scores))

    score_names, score_ranges, score_functions = list(zip(*scores))
    score_names = cast(List[str], score_names)
    score_ranges = cast(List[Tuple[float, float]], score_ranges)
    score_functions = cast(List[BinaryClassScoreFunctionType], score_functions)

    return DefaultScoreLists(score_names, score_ranges, score_functions)


def do_default_score_matrix_evaluation(
    dataset: Dataset,
    language: str,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType] = None,
    model_name_subset: Iterable[str] = None,
    overall_score_name_subset: Iterable[str] = None,
    per_class_score_name_subset: Iterable[str] = None,
    random_state: float = None,
) -> DefaultEvaluationResult:
    """Do 10-fold cross validation for default models and scores and save box plot."""
    # define subject ordering
    subject_order = unique_subject_order(dataset.subjects)

    # setup models and scores
    model_lists = default_named_models(
        language,
        model_name_subset=model_name_subset,
        subject_order=subject_order,
        subject_hierarchy=subject_hierarchy
    )
    overall_score_lists = default_named_multiclass_scores(overall_score_name_subset)
    per_class_score_lists = default_named_binary_scores(per_class_score_name_subset)

    # do evaluate
    overall_score_matrix, per_class_score_matrix = score_models_for_dataset(
        n_splits=10,
        dataset=dataset,
        subject_order=subject_order,
        models=model_lists.classes,
        overall_score_functions=overall_score_lists.functions,
        per_class_score_functions=per_class_score_lists.functions,
        random_state=random_state,
    )

    return DefaultEvaluationResult(
        overall_score_matrix,
        per_class_score_matrix,
        model_lists,
        overall_score_lists,
        per_class_score_lists
    )


def write_default_plots(
    evaluation_result: DefaultEvaluationResult,
    plot_directory: str,
    filename_suffix: str,
):
    """Write all default plots to a file."""
    write_precision_recall_plot(
        evaluation_result,
        os.path.join(plot_directory, f"precision_recall_plot_{filename_suffix}.html"),
    )

    write_per_subject_precision_recall_vs_samples_plot(
        evaluation_result,
        os.path.join(plot_directory, f"per_subject_precision_recall_vs_samples_plot_{filename_suffix}.html"),
    )

    write_score_matrix_box_plot(
        evaluation_result,
        os.path.join(plot_directory, f"score_plot_{filename_suffix}.html"),
    )

    write_per_subject_score_histograms_plot(
        evaluation_result,
        os.path.join(plot_directory, f"per_subject_score_{filename_suffix}.html"),
    )


def write_score_matrix_box_plot(
    evaluation_result: DefaultEvaluationResult,
    plot_filepath: str
):
    """Generate the score matrix box plot from evaluation results and write it as html file."""
    # generate figure
    score_matrix_box_plot(
        evaluation_result.overall_score_matrix,
        evaluation_result.model_lists.names,
        evaluation_result.overall_score_lists.names,
        evaluation_result.overall_score_lists.ranges,
        columns=2
    ).write_html(
        plot_filepath,
        include_plotlyjs="cdn",
        # default_height=f"{len(score_names) * 500}px"
    )


def write_precision_recall_plot(
    evaluation_result: DefaultEvaluationResult,
    plot_filepath: str
):
    """Generate the precision recall plot from evaluation results and write it as html file."""
    score_names = evaluation_result.overall_score_lists.names
    if "top3 precision micro" not in score_names or "top3 recall micro" not in score_names:
        raise ValueError("score matrix needs to contain top3 precision/recall micro scores")

    precision_idx = score_names.index("top3 precision micro")
    recall_idx = score_names.index("top3 recall micro")

    precision_recall_plot(
        evaluation_result.overall_score_matrix[:, [precision_idx, recall_idx], :],
        evaluation_result.model_lists.names,
    ).write_html(
        plot_filepath,
        include_plotlyjs="cdn",
    )


def write_per_subject_score_histograms_plot(
    evaluation_result: DefaultEvaluationResult,
    plot_filepath: str
):
    """Generate the subject score histograms plot from evaluation results and write it as html file."""
    per_subject_score_histograms_plot(
        evaluation_result.per_class_score_matrix,
        evaluation_result.model_lists.names,
        evaluation_result.per_class_score_lists.names,
        evaluation_result.per_class_score_lists.ranges,
    ).write_html(
        plot_filepath,
        include_plotlyjs="cdn",
    )


def write_per_subject_precision_recall_vs_samples_plot(
    evaluation_result: DefaultEvaluationResult,
    plot_filepath: str
):
    """Generate the per subject precision vs samples plot from evaluation results and write it as html file."""
    score_names = evaluation_result.per_class_score_lists.names
    if "t=0.5 precision" not in score_names or "t=0.5 recall" not in score_names \
            or "# training samples" not in score_names:
        raise ValueError("score matrix needs to contain t=0.5 precision, recall and # training examples scores")

    precision_idx = score_names.index("t=0.5 precision")
    recall_idx = score_names.index("t=0.5 recall")
    samples_idx = score_names.index("# training samples")

    per_subject_precision_recall_vs_samples_plot(
        evaluation_result.per_class_score_matrix[:, [samples_idx, precision_idx, recall_idx], :, :],
        evaluation_result.model_lists.names,
    ).write_html(
        plot_filepath,
        include_plotlyjs="cdn",
    )
