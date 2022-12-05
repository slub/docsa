"""Provides various utilily methods for experimentation."""

# pylint: disable=too-many-arguments, unnecessary-lambda, too-many-locals

import os
import logging

from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

import numpy as np

from slub_docsa.common.paths import get_cache_dir
from slub_docsa.common.dataset import Dataset
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.store.predictions import persisted_training_and_evaluation
from slub_docsa.evaluation.classification.incidence import unique_subject_order
from slub_docsa.evaluation.classification.split import DatasetSplitFunction, scikit_kfold_splitter
from slub_docsa.evaluation.classification.split import skmultilearn_iterative_stratification_splitter
from slub_docsa.evaluation.classification.pipeline import score_classification_models_for_dataset_with_splits
from slub_docsa.evaluation.clustering.pipeline import score_clustering_models_for_documents
from slub_docsa.experiments.common.models import NamedClassificationModels, NamedClusteringModels
from slub_docsa.experiments.common.plots import DefaultScoreMatrixDatasetResult, DefaultScoreMatrixResult
from slub_docsa.experiments.common.scores import NamedScoreLists, default_named_per_class_score_list
from slub_docsa.experiments.common.scores import default_named_score_list, initialize_named_score_tuple_list

logger = logging.getLogger(__name__)


def get_split_function_by_name(name: str, n_splits: int, random_state: Optional[float] = None) -> DatasetSplitFunction:
    """Return split function that matches simplified name."""
    # set up split function
    if name == "random":
        return scikit_kfold_splitter(n_splits, random_state)
    if name == "stratified":
        return skmultilearn_iterative_stratification_splitter(n_splits, random_state)
    raise RuntimeError("split function name not correct")


def do_default_score_matrix_classification_evaluation(
    named_datasets: Iterator[Tuple[str, Dataset, Optional[SubjectHierarchy]]],
    split_function: DatasetSplitFunction,
    named_models_generator: Callable[
        [Iterable[str], Optional[SubjectHierarchy]],
        NamedClassificationModels
    ],
    n_splits: int = 10,
    score_name_subset: Optional[Iterable[str]] = None,
    per_class_score_name_subset: Optional[Iterable[str]] = None,
    load_cached_scores: bool = False,
    stop_after_evaluating_split: Optional[int] = None,
) -> DefaultScoreMatrixResult:
    """Do 10-fold cross validation for default models and scores and save box plot."""
    results: DefaultScoreMatrixResult = []
    persisted_scores_cache_dir = os.path.join(get_cache_dir(), "scores")

    for dataset_name, dataset, subject_hierarchy_generator in named_datasets:
        # load scores from cache
        os.makedirs(persisted_scores_cache_dir, exist_ok=True)
        train_and_evaluate = persisted_training_and_evaluation(
            os.path.join(persisted_scores_cache_dir, dataset_name + ".sqlite"),
            load_cached_scores,
        )

        # define subject ordering
        subject_order = list(sorted(unique_subject_order(dataset.subjects)))
        subject_hierarchy = subject_hierarchy_generator()

        # setup models and scores
        model_lists = named_models_generator(subject_order, subject_hierarchy)
        score_lists = initialize_named_score_tuple_list(
            default_named_score_list(subject_order, subject_hierarchy),
            score_name_subset
        )
        per_class_score_lists = initialize_named_score_tuple_list(
            default_named_per_class_score_list(),
            per_class_score_name_subset
        )

        # do evaluate
        scores, per_class_scores = score_classification_models_for_dataset_with_splits(
            n_splits=n_splits,
            split_function=split_function,
            subject_order=subject_order,
            dataset=dataset,
            model_generators=model_lists.generators,
            score_generators=score_lists.generators,
            per_class_score_generators=per_class_score_lists.generators,
            train_and_evaluate=train_and_evaluate,
            stop_after_evaluating_split=stop_after_evaluating_split,
        )

        results.append(DefaultScoreMatrixDatasetResult(
            dataset_name,
            model_lists.names,
            np.array(scores),
            np.array(per_class_scores),
            score_lists,
            per_class_score_lists
        ))

    return results


def do_default_score_matrix_clustering_evaluation(
    named_datasets: Iterator[Tuple[str, Dataset, Optional[SubjectHierarchy]]],
    named_models_generator: Callable[[Iterable[str]], NamedClusteringModels],
    named_scores_generator: Callable[[], NamedScoreLists[Any]],
    repeats: int = 10,
    max_documents: Optional[int] = None,
) -> DefaultScoreMatrixResult:
    """Run clustering algorithms on each dataset and calculate multiple scores."""
    results: DefaultScoreMatrixResult = []

    for dataset_name, dataset, _ in named_datasets:

        # define subject ordering
        subject_order = list(sorted(unique_subject_order(dataset.subjects)))

        model_lists = named_models_generator(subject_order)
        score_lists = named_scores_generator()

        score_matrix = score_clustering_models_for_documents(
            dataset.documents,
            dataset.subjects,
            model_lists.classes,
            score_lists.functions,
            repeats,
            max_documents,
        )

        results.append(DefaultScoreMatrixDatasetResult(
            dataset_name,
            model_lists.names,
            overall_score_matrix=score_matrix,
            per_class_score_matrix=None,
            overall_score_lists=score_lists,
            per_class_score_lists=None,
        ))

    return results
