"""Provides various utilily methods for experimentation."""

# pylint: disable=too-many-arguments, unnecessary-lambda, too-many-locals

import os
import logging

from typing import Callable, Iterable, Iterator, Optional, Tuple

from slub_docsa.common.paths import ANNIF_DIR, CACHE_DIR
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.data.store.predictions import persisted_fit_classification_model_and_predict
from slub_docsa.evaluation.incidence import unique_subject_order
from slub_docsa.evaluation.split import DatasetSplitFunction, scikit_kfold_splitter
from slub_docsa.evaluation.split import skmultilearn_iterative_stratification_splitter
from slub_docsa.experiments.common.models import NamedClassificationModels
from slub_docsa.experiments.common.plots import DefaultScoreMatrixDatasetResult, DefaultScoreMatrixResult
from slub_docsa.experiments.common.scores import default_named_binary_classification_scores
from slub_docsa.experiments.common.scores import default_named_multiclass_scores
from slub_docsa.common.dataset import Dataset
from slub_docsa.evaluation.pipeline import score_classification_models_for_dataset

logger = logging.getLogger(__name__)

ANNIF_PROJECT_DATA_DIR = os.path.join(ANNIF_DIR, "testproject")
PREDICTIONS_CACHE = os.path.join(CACHE_DIR, "predictions")


def get_split_function_by_name(name: str, n_splits: int, random_state: float = None) -> DatasetSplitFunction:
    """Return split function that matches simplified name."""
    # set up split function
    if name == "random":
        return scikit_kfold_splitter(n_splits, random_state)
    if name == "stratified":
        return skmultilearn_iterative_stratification_splitter(n_splits, random_state)
    raise RuntimeError("split function name not correct")


def do_default_score_matrix_classification_evaluation(
    named_datasets: Iterator[Tuple[str, Dataset, Optional[SubjectHierarchyType[SubjectNodeType]]]],
    split_function: DatasetSplitFunction,
    named_models_generator: Callable[
        [Iterable[str], Optional[SubjectHierarchyType[SubjectNodeType]]],
        NamedClassificationModels
    ],
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
        fit_and_predict = persisted_fit_classification_model_and_predict(
            os.path.join(PREDICTIONS_CACHE, dataset_name + ".dbm"),
            load_cached_predictions,
        )

        # define subject ordering
        subject_order = list(sorted(unique_subject_order(dataset.subjects)))

        # setup models and scores
        model_lists = named_models_generator(subject_order, subject_hierarchy)
        overall_score_lists = default_named_multiclass_scores(
            subject_order,
            subject_hierarchy,
            overall_score_name_subset
        )
        per_class_score_lists = default_named_binary_classification_scores(per_class_score_name_subset)

        # do evaluate
        overall_score_matrix, per_class_score_matrix = score_classification_models_for_dataset(
            n_splits=n_splits,
            dataset=dataset,
            subject_order=subject_order,
            models=model_lists.classes,
            split_function=split_function,
            overall_score_functions=overall_score_lists.functions,
            per_class_score_functions=per_class_score_lists.functions,
            fit_and_predict=fit_and_predict,
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
