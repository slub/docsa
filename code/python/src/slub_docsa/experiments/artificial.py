"""Experiments based on artifical data that was randomly generated."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.data.artificial.hierarchical import generate_hierarchical_random_dataset_from_dbpedia
from slub_docsa.data.artificial.simple import generate_random_dataset
from slub_docsa.experiments.default import do_default_score_matrix_evaluation, write_per_subject_score_histograms_plot
from slub_docsa.experiments.default import write_score_matrix_box_plot, write_precision_recall_plot

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    dataset = None
    subject_hierarchy = None
    dataset_name = "hierarchical"
    n_token = 2000
    n_docs = 5000
    n_subjects = 10
    model_name_subset = ("oracle", "random", "knn k=1", "annif tfidf", "mlp")

    filename = f"{dataset_name}_token={n_token}_docs={n_docs}_subj={n_subjects}.html"

    # setup dataset
    if dataset_name == "random":
        dataset = generate_random_dataset(n_token, n_docs, n_subjects)
    if dataset_name == "hierarchical":
        dataset, subject_hierarchy = generate_hierarchical_random_dataset_from_dbpedia("english", n_docs, n_subjects)

    if dataset is None:
        raise ValueError("dataset can not be none")

    evaluation_result = do_default_score_matrix_evaluation(
        dataset=dataset,
        language="english",
        subject_hierarchy=subject_hierarchy,
        model_name_subset=model_name_subset
    )

    write_precision_recall_plot(
        evaluation_result,
        os.path.join(FIGURES_DIR, f"artificial/precision_recall_plot_{filename}"),
    )

    write_score_matrix_box_plot(
        evaluation_result,
        os.path.join(FIGURES_DIR, f"artificial/score_plot_{filename}"),
    )

    write_per_subject_score_histograms_plot(
        evaluation_result,
        os.path.join(FIGURES_DIR, f"artificial/per_subject_score_{filename}"),
    )
