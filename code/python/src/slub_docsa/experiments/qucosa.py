"""Experiments based on artifical data that was randomly generated."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.data.load.qucosa import read_qucosa_abstracts_rvk_training_dataset
from slub_docsa.data.load.qucosa import read_qucosa_titles_rvk_training_dataset
from slub_docsa.data.load.rvk import get_rvk_subject_store
from slub_docsa.data.preprocess.dataset import remove_subjects_with_insufficient_samples
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level, prune_subject_targets_to_minimum_samples
from slub_docsa.experiments.default import do_default_score_matrix_evaluation, write_default_plots
from slub_docsa.evaluation.incidence import unique_subject_order

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    prune_level = None  # 34 subjects at 1, 325 subjects at 2, in total 4857 subjects
    min_samples = 10
    model_subset = [
        "random", "oracle", "knn k=1",
        # "rforest", "mlp",
        # "annif tfidf", "annif svc", "annif fasttext", "annif omikuji", "annif vw_multi",
        "annif mllm",
        # "annif yake", "annif stwfsa"
    ]
    dataset_name = "titles"  # abstracts, titles
    filename_suffix = f"{dataset_name}_prune_level={prune_level}_mim_samples={min_samples}"

    # setup dataset
    rvk_hierarchy = get_rvk_subject_store()

    dataset = None
    if dataset_name == "titles":
        dataset = read_qucosa_titles_rvk_training_dataset()
    elif dataset_name == "abstracts":
        dataset = read_qucosa_abstracts_rvk_training_dataset()

    if dataset is None:
        raise RuntimeError("dataset can not be none, check parameters")

    logger.info("qucosa has %d unique subjects before pruning", len(unique_subject_order(dataset.subjects)))
    logger.info("qucosa has %d samples before pruning", len(dataset.documents))

    if prune_level is not None:
        if prune_level < 1:
            raise ValueError("prune level must be at least 1")
        dataset.subjects = prune_subject_targets_to_level(prune_level, dataset.subjects, rvk_hierarchy)
    else:
        dataset.subjects = prune_subject_targets_to_minimum_samples(min_samples, dataset.subjects, rvk_hierarchy)

    # remove subjects with less than 10 samples
    dataset = remove_subjects_with_insufficient_samples(dataset, min_samples)

    unique_subjects = unique_subject_order(dataset.subjects)
    logger.info("qucosa has %d unique subjects after pruning", len(unique_subjects))
    logger.info("qucosa has %d samples after pruning", len(dataset.documents))

    evaluation_result = do_default_score_matrix_evaluation(
        dataset=dataset,
        language="german",
        subject_hierarchy=rvk_hierarchy,
        model_name_subset=model_subset
    )

    write_default_plots(evaluation_result, os.path.join(FIGURES_DIR, "qucosa"), filename_suffix)
