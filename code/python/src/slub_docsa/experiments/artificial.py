"""Experiments based on artifical data that was randomly generated."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.data.artificial.hierarchical import generate_hierarchical_random_dataset
from slub_docsa.data.artificial.simple import generate_random_dataset
from slub_docsa.experiments.default import do_default_box_plot_evaluation

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset = None
    dataset_name = "hierarchical"
    n_token = 2000
    n_docs = 10000
    n_subjects = 10

    filename = f"box_plot_{dataset_name}_token={n_token}_docs={n_docs}_subj={n_subjects}.html"

    # setup dataset
    if dataset_name == "random":
        dataset = generate_random_dataset(n_token, n_docs, n_subjects)
    if dataset_name == "hierarchical":
        dataset, subject_hierarchy = generate_hierarchical_random_dataset(n_token, n_docs, n_subjects)

    if dataset is None:
        raise ValueError("dataset can not be none")

    do_default_box_plot_evaluation(
        dataset=dataset,
        language="english",
        box_plot_filepath=os.path.join(FIGURES_DIR, f"artificial/{filename}"),
        # model_name_subset=("random", "oracle", "knn k=1")
    )
