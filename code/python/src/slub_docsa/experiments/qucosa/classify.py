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
    dataset_subset = None  # abstracts, titles, fulltexts
    split_function_name = "random"  # either: random, stratified
    language_code = "de"  # either: de, en
    prune_level = None  # 34 subjects at 1, 325 subjects at 2, in total 4857 subjects
    n_splits = 10
    min_samples = 10
    model_subset = [
        # ### "random", ####
        "oracle",
        "nihilistic",
        "knn k=1",
        # "rforest",
        # "mlp",
        "annif tfidf",
        # "annif svc",
        "annif omikuji",
        # "annif vw_multi",
        # "annif mllm",
        # "annif fasttext"
        # "annif yake",
        # ### "annif stwfsa" ###
    ]
    filename_suffix = f"split={split_function_name}"

    rvk_hierarchy = get_rvk_subject_store()

    evaluation_result = do_default_score_matrix_evaluation(
        named_datasets=default_named_qucosa_datasets(dataset_subset),
        split_function=get_split_function_by_name(split_function_name, n_splits, random_state),
        language="german",
        model_name_subset=model_subset
    )

    write_default_plots(evaluation_result, os.path.join(FIGURES_DIR, "qucosa"), filename_suffix)
