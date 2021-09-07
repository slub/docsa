"""Experiments based on artifical data that was randomly generated."""

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.data.load.qucosa import read_qucosa_simple_rvk_training_dataset
from slub_docsa.experiments.default import do_default_box_plot_evaluation

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # setup dataset
    dataset = read_qucosa_simple_rvk_training_dataset()

    do_default_box_plot_evaluation(
        dataset,
        os.path.join(FIGURES_DIR, "qucosa_box_plot.html"),
    )
