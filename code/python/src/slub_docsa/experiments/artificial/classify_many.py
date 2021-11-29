"""Experiments based on artifical data that was randomly generated."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.experiments.annif.models import default_annif_named_model_list
from slub_docsa.experiments.artificial.datasets import default_named_artificial_datasets
from slub_docsa.experiments.common.models import initialize_classification_models_from_tuple_list
from slub_docsa.experiments.common.pipeline import do_default_score_matrix_classification_evaluation
from slub_docsa.experiments.common.pipeline import get_split_function_by_name
from slub_docsa.experiments.common.plots import write_default_classification_plots
from slub_docsa.experiments.dummy.models import default_dummy_named_model_list

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    random_state = 123
    load_cached_predictions = False
    dataset_subset = [
        "random_no_correlations",
        "random_easy_to_predict",
        "random_hierarchical"
    ]
    split_function_name = "random"  # either: random, stratified
    n_token = 1000
    n_docs = 1000
    n_subjects = 10
    n_splits = 10
    min_samples = 10
    model_name_subset = [
        "oracle",
        "nihilistic",
        # "random",
        "tfidf knn k=1",
        "dbmdz bert knn k=1",
        "random vectorizer knn k=1",
        # "knn k=3",
        # "mlp",
        # "rforest",
        # "annif tfidf",
        # "annif omikuji",
        # "annif vw_multi",
        # "annif fasttext",
        # "annif mllm",
        # "annif yake",
        # "annif stwfsa"
    ]

    filename_suffix = f"split={split_function_name}"

    named_datasets = default_named_artificial_datasets(n_token, n_docs, n_subjects, min_samples)

    def _model_list_generator(subject_order, subject_hierarchy):
        model_list = default_dummy_named_model_list() \
            + default_annif_named_model_list("en", subject_order, subject_hierarchy)
        return initialize_classification_models_from_tuple_list(model_list, model_name_subset)

    evaluation_result = do_default_score_matrix_classification_evaluation(
        named_datasets=named_datasets,
        split_function=get_split_function_by_name(split_function_name, n_splits, random_state),
        named_models_generator=_model_list_generator,
        load_cached_predictions=load_cached_predictions,
    )

    write_default_classification_plots(evaluation_result, os.path.join(FIGURES_DIR, "artificial"), filename_suffix)
