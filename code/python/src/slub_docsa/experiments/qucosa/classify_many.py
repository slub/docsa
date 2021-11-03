"""Experiments based on artifical data that was randomly generated."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.data.load.rvk import get_rvk_subject_store

from slub_docsa.experiments.common import do_default_score_matrix_evaluation, get_split_function_by_name
from slub_docsa.experiments.common import write_default_plots
from slub_docsa.experiments.qucosa.datasets import default_named_qucosa_datasets

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("slub_docsa.data.load.qucosa").setLevel(logging.DEBUG)

    random_state = 123
    load_cached_predictions = True
    stop_after_evaluating_first_split = True
    dataset_subset = [
        "qucosa_de_titles_langid_rvk",
        "qucosa_de_abstracts_langid_rvk",
        "qucosa_de_fulltexts_langid_rvk",
    ]
    split_function_name = "random"  # either: random, stratified
    n_splits = 10
    model_subset = [
        # ### "random", ####
        "oracle",
        "nihilistic",
        "tfidf knn k=1",
        "dbmdz bert knn k=1",
        "tfidf rforest",
        "dbmdz bert rforest",
        "tfidf scikit mlp",
        "tfidf torch ann",
        "dbmdz bert scikit mlp",
        "dbmdz bert torch ann",
        "annif tfidf",
        "annif svc",
        "annif omikuji",
        "annif vw_multi",
        "annif mllm",
        "annif fasttext"
        "annif yake",
        # ### "annif stwfsa" ###
    ]
    filename_suffix = f"split={split_function_name}"

    rvk_hierarchy = get_rvk_subject_store()

    evaluation_result = do_default_score_matrix_evaluation(
        named_datasets=default_named_qucosa_datasets(dataset_subset),
        split_function=get_split_function_by_name(split_function_name, n_splits, random_state),
        language="german",
        model_name_subset=model_subset,
        load_cached_predictions=load_cached_predictions,
        stop_after_evaluating_first_split=stop_after_evaluating_first_split,
    )

    write_default_plots(evaluation_result, os.path.join(FIGURES_DIR, "qucosa"), filename_suffix)