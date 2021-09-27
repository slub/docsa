"""Experiments based on artifical data that was randomly generated."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.data.artificial.hierarchical import generate_hierarchical_random_dataset_from_dbpedia
from slub_docsa.data.artificial.simple import generate_random_dataset
from slub_docsa.data.preprocess.dataset import remove_subjects_with_insufficient_samples
from slub_docsa.experiments.default import do_default_score_matrix_evaluation, write_default_plots

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    dataset = None
    subject_hierarchy = None
    dataset_name = "hierarchical"
    n_token = 1000
    n_docs = 5000
    n_subjects = 20
    model_name_subset = [
        "oracle", "random", "knn k=1",
        # "knn k=3", "mlp", "rforest",
        # "annif tfidf",
        # "annif mllm",
        # "annif yake",
        # "annif stwfsa"
    ]

    filename_suffix = f"{dataset_name}_token={n_token}_docs={n_docs}_subj={n_subjects}"

    # setup dataset
    if dataset_name == "random":
        dataset = generate_random_dataset(n_token, n_docs, n_subjects)
    if dataset_name == "hierarchical":
        dataset, subject_hierarchy = generate_hierarchical_random_dataset_from_dbpedia(
            "english", n_token, n_docs, n_subjects
        )

    if dataset is None:
        raise RuntimeError("dataset can not be none, check parameters")

    # remove subjects with less than 10 samples
    dataset = remove_subjects_with_insufficient_samples(dataset, 10)

    logger.info("subject hierarchy is %s", subject_hierarchy)

    if dataset is None:
        raise ValueError("dataset can not be none")

    evaluation_result = do_default_score_matrix_evaluation(
        dataset=dataset,
        language="english",
        subject_hierarchy=subject_hierarchy,
        model_name_subset=model_name_subset
    )

    write_default_plots(evaluation_result, os.path.join(FIGURES_DIR, "artificial"), filename_suffix)
