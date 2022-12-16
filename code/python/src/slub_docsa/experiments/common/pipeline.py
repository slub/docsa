"""Provides various utilily methods for experimentation."""

# pylint: disable=too-many-arguments, unnecessary-lambda, too-many-locals, too-many-function-args

from functools import partial
import os
import logging

from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

from slub_docsa.common.paths import get_cache_dir
from slub_docsa.common.dataset import Dataset
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.store.predictions import persisted_training_and_evaluation
from slub_docsa.data.store.subject import cached_unique_subject_order
from slub_docsa.evaluation.classification.incidence import unique_subject_order
from slub_docsa.evaluation.classification.split import DatasetSplitFunction, scikit_kfold_splitter
from slub_docsa.evaluation.classification.split import skmultilearn_iterative_stratification_splitter
from slub_docsa.evaluation.clustering.pipeline import score_clustering_models_for_documents
from slub_docsa.experiments.common.datasets import NamedDataset
from slub_docsa.experiments.common.models import NamedClusteringModels
from slub_docsa.experiments.common.plots import DefaultScoreMatrixDatasetResult, DefaultScoreMatrixResult
from slub_docsa.experiments.common.scores import NamedScoreLists, default_named_per_class_score_list
from slub_docsa.experiments.common.scores import default_named_score_list, initialize_named_score_tuple_list
from slub_docsa.serve.common import ModelTypeMapping

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
    named_datasets: Sequence[NamedDataset],
    split_function_name: str,
    split_number: int,
    model_types: ModelTypeMapping,
    score_name_subset: Optional[Iterable[str]] = None,
    per_class_score_name_subset: Optional[Iterable[str]] = None,
    stop_after_evaluating_split: Optional[int] = None,
    load_cached_scores: bool = True,
    publish_model: bool = True,
    evaluate_batch_size: int = 100,
    random_state: int = None,
) -> DefaultScoreMatrixResult:
    """Do 10-fold cross validation for default models and scores and save box plot."""
    results: DefaultScoreMatrixResult = []
    scores_cache_dir = os.path.join(get_cache_dir(), "experiments/scores")
    models_publish_dir = os.path.join(get_cache_dir(), "experiments/models")
    subject_order_cache_dir = os.path.join(get_cache_dir(), "experiments/subject_orders")
    os.makedirs(scores_cache_dir, exist_ok=True)
    os.makedirs(models_publish_dir, exist_ok=True)
    os.makedirs(subject_order_cache_dir, exist_ok=True)

    split_function = get_split_function_by_name(split_function_name, split_number, random_state=random_state)

    for named_dataset in named_datasets:
        # define subject ordering
        subject_order = cached_unique_subject_order(
            named_dataset.name, named_dataset.dataset.subjects, subject_order_cache_dir
        )
        subject_hierarchy = named_dataset.schema_generator()

        # setup models and scores
        model_type_order = list(model_types.keys())
        score_lists = initialize_named_score_tuple_list(
            default_named_score_list(),  # subject_order, subject_hierarchy),
            score_name_subset
        )
        per_class_score_lists = initialize_named_score_tuple_list(
            default_named_per_class_score_list(),
            per_class_score_name_subset
        )

        dataset_scores = []
        dataset_per_class_scores = []
        for split_idx, (train_dataset, test_dataset) in enumerate(split_function(named_dataset.dataset)):
            model_scores = []
            model_per_class_scores = []

            split_id = f"{split_function_name}_{str(split_idx)}"

            for model_type in model_type_order:
                # load scores from cache
                train_and_evaluate = persisted_training_and_evaluation(
                    scores_cache_dir,
                    models_publish_dir,
                    named_dataset.schema_id,
                    named_dataset.name,
                    model_type,
                    named_dataset.languages,
                    split_id,
                    score_lists.names,
                    per_class_score_lists.names,
                    evaluate_batch_size,
                    publish_model,
                    load_cached_scores,
                )

                scores, per_class_scores = train_and_evaluate(
                    partial(model_types[model_type], subject_hierarchy, subject_order),
                    subject_order,
                    train_dataset,
                    test_dataset,
                    score_lists.generators,
                    per_class_score_lists.generators,
                    None
                )
                model_scores.append(scores)
                model_per_class_scores.append(per_class_scores)

            dataset_scores.append(model_scores)
            dataset_per_class_scores.append(model_per_class_scores)

            if split_idx >= stop_after_evaluating_split:
                break

        results.append(DefaultScoreMatrixDatasetResult(
            named_dataset.name,
            model_type_order,
            np.array(dataset_scores),
            np.array(dataset_per_class_scores),
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

    for named_dataset in named_datasets:

        # define subject ordering
        subject_order = list(sorted(unique_subject_order(named_dataset.dataset.subjects)))

        model_lists = named_models_generator(subject_order)
        score_lists = named_scores_generator()

        score_matrix = score_clustering_models_for_documents(
            named_dataset.dataset.documents,
            named_dataset.dataset.subjects,
            model_lists.classes,
            score_lists.generators,
            repeats,
            max_documents,
        )

        results.append(DefaultScoreMatrixDatasetResult(
            named_dataset.name,
            model_lists.names,
            score_matrix=score_matrix,
            per_class_score_matrix=None,
            score_lists=score_lists,
            per_class_score_lists=None,
        ))

    return results
