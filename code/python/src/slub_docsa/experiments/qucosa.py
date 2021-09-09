"""Experiments based on artifical data that was randomly generated."""

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.data.load.qucosa import read_qucosa_simple_rvk_training_dataset
from slub_docsa.data.load.rvk import get_rvk_subject_store
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level
from slub_docsa.experiments.default import do_default_box_plot_evaluation
from slub_docsa.evaluation.incidence import unique_subject_order

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    PRUNE_LEVEL = 1

    # setup dataset
    rvk_hierarchy = get_rvk_subject_store()
    dataset = read_qucosa_simple_rvk_training_dataset()

    dataset.subjects = prune_subject_targets_to_level(1, dataset.subjects, rvk_hierarchy)

    unique_subjects = unique_subject_order(dataset.subjects)
    logger.info("qucosa has %d unique subjects at prune level %d", len(unique_subjects), PRUNE_LEVEL)

    do_default_box_plot_evaluation(
        dataset,
        os.path.join(FIGURES_DIR, "qucosa_box_plot_pl=%d.html" % (PRUNE_LEVEL)),
        model_name_subset=["random", "stratified", "knn k=1", "rforest", "mlp", "annif tfidf", "annif fasttext"]
    )
