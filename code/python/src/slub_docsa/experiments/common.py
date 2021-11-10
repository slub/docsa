"""Provides common defaults for experimentation."""

# pylint: disable=too-many-arguments, unnecessary-lambda, too-many-locals

import os
import logging

from typing import Any, Callable, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple, cast

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

from slub_docsa.common.paths import ANNIF_DIR, CACHE_DIR
from slub_docsa.common.score import MultiClassScoreFunctionType, BinaryClassScoreFunctionType
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.data.preprocess.vectorizer import HuggingfaceBertVectorizer, RandomVectorizer, TfidfStemmingVectorizer
from slub_docsa.data.preprocess.vectorizer import PersistedCachedVectorizer
from slub_docsa.data.store.predictions import persisted_fit_model_and_predict
from slub_docsa.evaluation.incidence import threshold_incidence_decision, positive_top_k_incidence_decision
from slub_docsa.evaluation.incidence import unique_subject_order
from slub_docsa.evaluation.plotting import score_matrices_box_plot
from slub_docsa.evaluation.plotting import per_subject_precision_recall_vs_samples_plot
from slub_docsa.evaluation.plotting import precision_recall_plot, score_matrix_box_plot
from slub_docsa.evaluation.plotting import per_subject_score_histograms_plot
from slub_docsa.evaluation.score import cesa_bianchi_h_loss, scikit_incidence_metric
from slub_docsa.evaluation.score import scikit_metric_for_best_threshold_based_on_f1score
from slub_docsa.evaluation.split import DatasetSplitFunction, scikit_kfold_splitter
from slub_docsa.evaluation.split import skmultilearn_iterative_stratification_splitter
from slub_docsa.models.dummy import NihilisticModel, OracleModel
from slub_docsa.models.ann_torch import TorchSingleLayerDenseModel
from slub_docsa.models.natlibfi_annif import AnnifModel
from slub_docsa.models.scikit import ScikitClassifier, ScikitTfidiRandomClassifier
from slub_docsa.common.model import Model
from slub_docsa.common.dataset import Dataset
from slub_docsa.evaluation.pipeline import score_models_for_dataset

logger = logging.getLogger(__name__)

ANNIF_PROJECT_DATA_DIR = os.path.join(ANNIF_DIR, "testproject")
PREDICTIONS_CACHE = os.path.join(CACHE_DIR, "predictions")
VECTORIZATION_CACHE = os.path.join(CACHE_DIR, "vectorizer")


class DefaultModelLists(NamedTuple):
    """Stores names and classes for default models."""

    names: List[str]
    classes: List[Model]


class DefaultScoreLists(NamedTuple):
    """Stores names, ranges and functions of default scores (both multi-class and binary)."""

    names: List[str]
    ranges: List[Tuple[float, float]]
    functions: List[Callable]


class DefaultScoreMatrixDatasetResult(NamedTuple):
    """Stores evaluation result matrices as well as model and score info."""

    dataset_name: str
    model_names: List[str]
    overall_score_matrix: np.ndarray
    per_class_score_matrix: np.ndarray
    overall_score_lists: DefaultScoreLists
    per_class_score_lists: DefaultScoreLists


DefaultScoreMatrixResult = Sequence[DefaultScoreMatrixDatasetResult]


def get_qucosa_dbmdz_bert_vectorizer(subtext_samples: int = 1):
    """Load persisted dbmdz bert vectorizer."""
    os.makedirs(VECTORIZATION_CACHE, exist_ok=True)
    filename = f"dbmdz_bert_qucosa_subtext_samples={subtext_samples}.sqlite"
    dbmdz_bert_cache_fp = os.path.join(VECTORIZATION_CACHE, filename)
    return PersistedCachedVectorizer(dbmdz_bert_cache_fp, HuggingfaceBertVectorizer(subtext_samples=subtext_samples))


def default_named_models(
    lang_code: str,
    subject_order: Sequence[str] = None,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType] = None,
    model_name_subset: Iterable[str] = None
) -> DefaultModelLists:
    """Return a list of default models to use for evaluating model performance."""
    dbmdz_bert_vectorizer_sts_1 = get_qucosa_dbmdz_bert_vectorizer(1)
    dbmdz_bert_vectorizer_sts_8 = get_qucosa_dbmdz_bert_vectorizer(8)

    models = [
        ("random", lambda: ScikitTfidiRandomClassifier()),
        ("nihilistic", lambda: NihilisticModel()),
        ("stratified", lambda: ScikitClassifier(
            predictor=DummyClassifier(strategy="stratified"),
            vectorizer=RandomVectorizer(),
        )),
        ("oracle", lambda: OracleModel()),
        ("tfidf 10k knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=10000),
        )),
        ("tfidf 40k knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=40000),
        )),
        ("dbmdz bert sts1 knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=dbmdz_bert_vectorizer_sts_1,
        )),
        ("dbmdz bert sts8 knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=dbmdz_bert_vectorizer_sts_8,
        )),
        ("random vectorizer knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=RandomVectorizer(),
        )),
        ("tfidf 10k knn k=3", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=3),
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=10000),
        )),
        ("tfidf 10k dtree", lambda: ScikitClassifier(
            predictor=DecisionTreeClassifier(max_leaf_nodes=1000),
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=10000),
        )),
        ("tfidf 10k rforest", lambda: ScikitClassifier(
            predictor=RandomForestClassifier(n_jobs=-1, max_leaf_nodes=1000),
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=10000),
        )),
        ("dbmdz bert sts1 rforest", lambda: ScikitClassifier(
            predictor=RandomForestClassifier(n_jobs=-1, max_leaf_nodes=1000),
            vectorizer=dbmdz_bert_vectorizer_sts_1,
        )),
        ("tfidf 10k scikit mlp", lambda: ScikitClassifier(
            predictor=MLPClassifier(max_iter=10),
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=10000),
        )),
        ("tfidf 10k torch ann", lambda: TorchSingleLayerDenseModel(
            epochs=50,
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=10000),
        )),
        ("tfidf 40k torch ann", lambda: TorchSingleLayerDenseModel(
            epochs=50,
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=40000),
        )),
        ("dbmdz bert sts1 scikit mlp", lambda: ScikitClassifier(
            predictor=MLPClassifier(max_iter=10),
            vectorizer=dbmdz_bert_vectorizer_sts_1,
        )),
        ("dbmdz bert sts1 torch ann", lambda: TorchSingleLayerDenseModel(
            epochs=50,
            vectorizer=dbmdz_bert_vectorizer_sts_1,
        )),
        ("dbmdz bert sts8 torch ann", lambda: TorchSingleLayerDenseModel(
            epochs=50,
            vectorizer=dbmdz_bert_vectorizer_sts_8,
        )),
        ("tfidf 10k log_reg", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(estimator=LogisticRegression()),
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=10000),
        )),
        ("tfidf 10k nbayes", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(estimator=GaussianNB()),
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=10000),
        )),
        ("tfidf 10k svc", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(
                estimator=CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)
            ),
            vectorizer=TfidfStemmingVectorizer(lang_code, max_features=10000),
        )),
        ("annif tfidf", lambda: AnnifModel(model_type="tfidf", lang_code=lang_code)),
        ("annif svc", lambda: AnnifModel(model_type="svc", lang_code=lang_code)),
        ("annif fasttext", lambda: AnnifModel(model_type="fasttext", lang_code=lang_code)),
        ("annif omikuji", lambda: AnnifModel(model_type="omikuji", lang_code=lang_code)),
        ("annif vw_multi", lambda: AnnifModel(model_type="vw_multi", lang_code=lang_code)),
        ("annif mllm", lambda: AnnifModel(
            model_type="mllm", lang_code=lang_code, subject_order=subject_order, subject_hierarchy=subject_hierarchy
        )),
        ("annif yake", lambda: AnnifModel(
            model_type="yake", lang_code=lang_code, subject_order=subject_order, subject_hierarchy=subject_hierarchy
        )),
        ("annif stwfsa", lambda: AnnifModel(
            model_type="stwfsa", lang_code=lang_code, subject_order=subject_order, subject_hierarchy=subject_hierarchy
        )),
    ]

    if model_name_subset is not None:
        models = list(filter(lambda i: i[0] in model_name_subset, models))

    model_names, model_classes = list(zip(*models))
    model_names = cast(List[str], model_names)
    model_classes = cast(List[Model], [cls() for cls in model_classes])

    return DefaultModelLists(model_names, model_classes)


def default_named_multiclass_scores(
    subject_order: Sequence[str] = None,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType] = None,
    score_name_subset: Iterable[str] = None
) -> DefaultScoreLists:
    """Return a list of default score functions for evaluation."""
    scores = [
        # ("t=0.5 accuracy", [0, 1], scikit_incidence_metric(
        #     threshold_incidence_decision(0.5),
        #     accuracy_score
        # )),
        # ("top3 accuracy", [0, 1], scikit_incidence_metric(
        #     positive_top_k_incidence_decision(3),
        #     accuracy_score
        # )),
        ("t=best f1_score micro", [0, 1], scikit_metric_for_best_threshold_based_on_f1score(
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
        ("t=best precision micro", [0, 1], scikit_metric_for_best_threshold_based_on_f1score(
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
        ("t=best recall micro", [0, 1], scikit_metric_for_best_threshold_based_on_f1score(
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
        ("t=best h_loss", [0, 1], scikit_metric_for_best_threshold_based_on_f1score(
            cesa_bianchi_h_loss(subject_hierarchy, subject_order, log_factor=1000),
        )),
        ("top3 h_loss", [0, 1], scikit_incidence_metric(
            positive_top_k_incidence_decision(3),
            cesa_bianchi_h_loss(subject_hierarchy, subject_order, log_factor=1000),
        )),
        ("roc auc micro", [0, 1], lambda t, p: roc_auc_score(t, p, average="micro")),
        ("log loss", [0, None], log_loss),
        # ("mean squared error", [0, None], mean_squared_error)
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
        ("# test samples", [0, None], lambda t, _: len(np.where(t > 0)[0]))
    ]

    if score_name_subset is not None:
        scores = list(filter(lambda i: i[0] in score_name_subset, scores))

    score_names, score_ranges, score_functions = list(zip(*scores))
    score_names = cast(List[str], score_names)
    score_ranges = cast(List[Tuple[float, float]], score_ranges)
    score_functions = cast(List[BinaryClassScoreFunctionType], score_functions)

    return DefaultScoreLists(score_names, score_ranges, score_functions)


def get_split_function_by_name(name: str, n_splits: int, random_state: float = None) -> DatasetSplitFunction:
    """Return split function that matches simplified name."""
    # set up split function
    if name == "random":
        return scikit_kfold_splitter(n_splits, random_state)
    if name == "stratified":
        return skmultilearn_iterative_stratification_splitter(n_splits, random_state)
    raise RuntimeError("split function name not correct")


def do_default_score_matrix_evaluation(
    named_datasets: Iterator[Tuple[str, Dataset, Optional[SubjectHierarchyType[SubjectNodeType]]]],
    split_function: DatasetSplitFunction,
    lang_code: str,
    model_name_subset: Iterable[str] = None,
    overall_score_name_subset: Iterable[str] = None,
    per_class_score_name_subset: Iterable[str] = None,
    load_cached_predictions: bool = False,
    stop_after_evaluating_split: int = None,
) -> DefaultScoreMatrixResult:
    """Do 10-fold cross validation for default models and scores and save box plot."""
    n_splits = 10
    results: DefaultScoreMatrixResult = []

    for dataset_name, dataset, subject_hierarchy in named_datasets:
        # load predictions cache
        os.makedirs(PREDICTIONS_CACHE, exist_ok=True)
        fit_model_and_predict = persisted_fit_model_and_predict(
            os.path.join(PREDICTIONS_CACHE, dataset_name + ".dbm"),
            load_cached_predictions,
        )

        # define subject ordering
        subject_order = list(sorted(unique_subject_order(dataset.subjects)))

        # setup models and scores
        model_lists = default_named_models(
            lang_code,
            model_name_subset=model_name_subset,
            subject_order=subject_order,
            subject_hierarchy=subject_hierarchy
        )
        overall_score_lists = default_named_multiclass_scores(
            subject_order,
            subject_hierarchy,
            overall_score_name_subset
        )
        per_class_score_lists = default_named_binary_scores(per_class_score_name_subset)

        # do evaluate
        overall_score_matrix, per_class_score_matrix = score_models_for_dataset(
            n_splits=n_splits,
            dataset=dataset,
            subject_order=subject_order,
            models=model_lists.classes,
            split_function=split_function,
            overall_score_functions=overall_score_lists.functions,
            per_class_score_functions=per_class_score_lists.functions,
            fit_model_and_predict=fit_model_and_predict,
            stop_after_evaluating_split=stop_after_evaluating_split,
        )

        results.append(DefaultScoreMatrixDatasetResult(
            dataset_name,
            model_lists.names,
            overall_score_matrix,
            per_class_score_matrix,
            overall_score_lists,
            per_class_score_lists
        ))

    return results


def write_default_plots(
    evaluation_result: DefaultScoreMatrixResult,
    plot_directory: str,
    filename_suffix: str,
):
    """Write all default plots to a file."""
    write_multiple_score_matrix_box_plot(
        evaluation_result,
        os.path.join(plot_directory, f"overall_score_plot_{filename_suffix}"),
    )

    for dataset_result in evaluation_result:
        prefix = f"{dataset_result.dataset_name}"

        write_precision_recall_plot(
            dataset_result,
            os.path.join(plot_directory, f"{prefix}_precision_recall_plot_{filename_suffix}"),
        )

        write_per_subject_precision_recall_vs_samples_plot(
            dataset_result,
            os.path.join(plot_directory, f"{prefix}_per_subject_precision_recall_vs_samples_plot_{filename_suffix}"),
        )

        write_score_matrix_box_plot(
            dataset_result,
            os.path.join(plot_directory, f"{prefix}_score_plot_{filename_suffix}"),
        )

        write_per_subject_score_histograms_plot(
            dataset_result,
            os.path.join(plot_directory, f"{prefix}_per_subject_score_{filename_suffix}"),
        )


def write_multiple_figure_formats(
    figure: Any,
    filepath: str,
):
    """Write a plotly figure as a html, pdf and jpg file."""
    figure.write_html(
        f"{filepath}.html",
        include_plotlyjs="cdn",
    )
    figure.write_image(
        f"{filepath}.pdf",
        width=1600, height=900
    )
    figure.write_image(
        f"{filepath}.jpg",
        width=1600, height=900
    )


def write_multiple_score_matrix_box_plot(
    evaluation_result: DefaultScoreMatrixResult,
    plot_filepath: str
):
    """Generate the score matrix box plot comparing multiple datasets and write it as html file."""
    score_matrices = [er.overall_score_matrix for er in evaluation_result]
    dataset_names = [er.dataset_name for er in evaluation_result]
    model_names = evaluation_result[0].model_names
    score_names = evaluation_result[0].overall_score_lists.names
    score_ranges = evaluation_result[0].overall_score_lists.ranges

    # generate figure
    figure = score_matrices_box_plot(
        score_matrices,
        dataset_names,
        model_names,
        score_names,
        score_ranges,
        columns=2
    )
    write_multiple_figure_formats(figure, plot_filepath)


def write_score_matrix_box_plot(
    evaluation_result: DefaultScoreMatrixDatasetResult,
    plot_filepath: str
):
    """Generate the score matrix box plot from evaluation results and write it as html file."""
    # generate figure
    figure = score_matrix_box_plot(
        evaluation_result.overall_score_matrix,
        evaluation_result.model_names,
        evaluation_result.overall_score_lists.names,
        evaluation_result.overall_score_lists.ranges,
        columns=2
    )
    write_multiple_figure_formats(figure, plot_filepath)


def write_precision_recall_plot(
    evaluation_result: DefaultScoreMatrixDatasetResult,
    plot_filepath: str
):
    """Generate the precision recall plot from evaluation results and write it as html file."""
    score_names = evaluation_result.overall_score_lists.names
    if "top3 precision micro" not in score_names or "top3 recall micro" not in score_names:
        raise ValueError("score matrix needs to contain top3 precision/recall micro scores")

    precision_idx = score_names.index("top3 precision micro")
    recall_idx = score_names.index("top3 recall micro")

    figure = precision_recall_plot(
        evaluation_result.overall_score_matrix[:, [precision_idx, recall_idx], :],
        evaluation_result.model_names,
    )
    write_multiple_figure_formats(figure, plot_filepath)


def write_per_subject_score_histograms_plot(
    evaluation_result: DefaultScoreMatrixDatasetResult,
    plot_filepath: str
):
    """Generate the subject score histograms plot from evaluation results and write it as html file."""
    figure = per_subject_score_histograms_plot(
        evaluation_result.per_class_score_matrix,
        evaluation_result.model_names,
        evaluation_result.per_class_score_lists.names,
        evaluation_result.per_class_score_lists.ranges,
    )
    write_multiple_figure_formats(figure, plot_filepath)


def write_per_subject_precision_recall_vs_samples_plot(
    evaluation_result: DefaultScoreMatrixDatasetResult,
    plot_filepath: str
):
    """Generate the per subject precision vs samples plot from evaluation results and write it as html file."""
    score_names = evaluation_result.per_class_score_lists.names
    if "t=0.5 precision" not in score_names or "t=0.5 recall" not in score_names \
            or "# test samples" not in score_names:
        raise ValueError("score matrix needs to contain t=0.5 precision, recall and # test examples scores")

    precision_idx = score_names.index("t=0.5 precision")
    recall_idx = score_names.index("t=0.5 recall")
    samples_idx = score_names.index("# test samples")

    figure = per_subject_precision_recall_vs_samples_plot(
        evaluation_result.per_class_score_matrix[:, [samples_idx, precision_idx, recall_idx], :, :],
        evaluation_result.model_names,
    )
    write_multiple_figure_formats(figure, plot_filepath)
