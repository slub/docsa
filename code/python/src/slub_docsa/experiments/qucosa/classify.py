"""Experiments based on artifical data that was randomly generated."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.data.load.qucosa import read_qucosa_abstracts_rvk_training_dataset
from slub_docsa.data.load.qucosa import read_qucosa_documents_from_directory, read_qucosa_fulltext_rvk_training_dataset
from slub_docsa.data.load.qucosa import read_qucosa_titles_rvk_training_dataset
from slub_docsa.data.load.rvk import get_rvk_subject_store
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.data.preprocess.language import filter_dataset_by_detected_fulltext_language_via_langid
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level, prune_subject_targets_to_minimum_samples
from slub_docsa.experiments.common import do_default_score_matrix_evaluation, get_split_function_by_name
from slub_docsa.experiments.common import write_default_plots
from slub_docsa.evaluation.incidence import unique_subject_order

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("slub_docsa.data.load.qucosa").setLevel(logging.DEBUG)

    dataset_name = "titles"  # abstracts, titles, fulltexts
    split_function_name = "stratified"  # either: random, stratified
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
        # "annif tfidf",
        # "annif svc",
        # "annif omikuji",
        # "annif vw_multi",
        # "annif mllm",
        # "annif fasttext"
        # "annif yake",
        # ### "annif stwfsa" ###
    ]
    filename_suffix = f"{dataset_name}_{language_code}_split={split_function_name}_" + \
        f"prune_level={prune_level}_min_samples={min_samples}"

    # setup dataset
    logger.info("loading rvk subjects")
    rvk_hierarchy = get_rvk_subject_store()

    logger.info("loading qucosa dataset")
    dataset = None
    if dataset_name == "titles":
        dataset = read_qucosa_titles_rvk_training_dataset(read_qucosa_documents_from_directory())
    elif dataset_name == "abstracts":
        dataset = read_qucosa_abstracts_rvk_training_dataset(read_qucosa_documents_from_directory())
    elif dataset_name == "fulltexts":
        dataset = read_qucosa_fulltext_rvk_training_dataset(read_qucosa_documents_from_directory(), language_code)
        dataset = filter_dataset_by_detected_fulltext_language_via_langid(dataset, language_code)

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
    dataset = filter_subjects_with_insufficient_samples(dataset, min_samples)

    unique_subjects = unique_subject_order(dataset.subjects)
    logger.info("qucosa has %d unique subjects after pruning", len(unique_subjects))
    logger.info("qucosa has %d samples after pruning", len(dataset.documents))

    evaluation_result = do_default_score_matrix_evaluation(
        dataset=dataset,
        split_function=get_split_function_by_name(split_function_name, n_splits),
        language="german",
        subject_hierarchy=rvk_hierarchy,
        model_name_subset=model_subset
    )

    write_default_plots(evaluation_result, os.path.join(FIGURES_DIR, "qucosa"), filename_suffix)
