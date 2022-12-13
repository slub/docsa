"""Experiments based on artifical data that was randomly generated."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import get_figures_dir
from slub_docsa.experiments.artificial.datasets import default_named_artificial_datasets
from slub_docsa.experiments.common.models import filter_model_type_mapping
from slub_docsa.experiments.common.pipeline import do_default_score_matrix_classification_evaluation
from slub_docsa.experiments.common.plots import write_default_classification_plots
from slub_docsa.experiments.dummy.models import default_dummy_model_types
from slub_docsa.serve.models.classification.common import get_all_classification_model_types

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    random_state = 123
    load_cached_scores = False
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
    model_subset = [
        "oracle",
        "nihilistic",
        "random",
        "tfidf_10k_knn_k=1",
        "tfidf_10k_knn_k=3",
        "tfidf_10k_dtree",
        "tfidf_10k_rforest",
        "tfidf_10k_log_reg",
        "tfidf_10k_nbayes",
        "tfidf_10k_svc",
        "annif_tfidf",
        # "annif omikuji",
        # "annif vw_multi",
        "annif_fasttext",
        # "annif mllm",
        # "annif yake",
        # "annif stwfsa"
    ]

    filename_suffix = f"split={split_function_name}"

    named_datasets = default_named_artificial_datasets(n_token, n_docs, n_subjects, min_samples)

    model_types = default_dummy_model_types()
    model_types.update(get_all_classification_model_types())
    model_types = filter_model_type_mapping(model_types, model_subset)

    evaluation_result = do_default_score_matrix_classification_evaluation(
        named_datasets=named_datasets,
        model_types=model_types,
        split_function_name=split_function_name,
        split_number=n_splits,
        load_cached_scores=load_cached_scores,
        random_state=random_state,
    )

    write_default_classification_plots(
        evaluation_result,
        os.path.join(get_figures_dir(), "artificial"),
        filename_suffix
    )
