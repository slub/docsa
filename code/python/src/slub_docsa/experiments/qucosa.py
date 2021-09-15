"""Experiments based on artifical data that was randomly generated."""

# pylint: disable=invalid-name

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

    prune_level = None  # 34 subjects at 1, 325 subjects at 2, in total 4857 subjects
    model_subset = [
        "random", "oracle",
        "knn k=1", "rforest", "mlp",
        "annif tfidf", "annif svc", "annif fasttext", "annif omikuji", "annif vw_multi",
        # "annif mllm",
    ]
    filename = f"box_plot_prune_level={prune_level}.html"

    # setup dataset
    rvk_hierarchy = get_rvk_subject_store()
    dataset = read_qucosa_simple_rvk_training_dataset()

    if prune_level is not None:
        if prune_level < 1:
            raise ValueError("prune level must be at least 1")
        dataset.subjects = prune_subject_targets_to_level(prune_level, dataset.subjects, rvk_hierarchy)

    unique_subjects = unique_subject_order(dataset.subjects)
    logger.info("qucosa has %d unique subjects at prune level %s", len(unique_subjects), str(prune_level))

    do_default_box_plot_evaluation(
        dataset=dataset,
        language="german",
        box_plot_filepath=os.path.join(FIGURES_DIR, f"qucosa/{filename}"),
        model_name_subset=model_subset,
    )
