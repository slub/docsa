"""Experiments based on artifical data that was randomly generated."""

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.data.artificial.hierarchical import generate_hierarchical_random_dataset
from slub_docsa.experiments.default import do_default_box_plot_evaluation

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # setup dataset
    # uncorrelated_dataset = generate_random_dataset(2000, 5000, 10)
    dataset, subject_hierarchy = generate_hierarchical_random_dataset(2000, 1000, 3)

    do_default_box_plot_evaluation(
        dataset,
        os.path.join(FIGURES_DIR, "artificial_hierarchy_box_plot.html"),
        model_name_subset=("random", "oracle", "knn k=1")
    )
