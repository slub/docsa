"""Evaluates and compares multiple models and multiple variants of the Qucosa dataset."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.data.load.rvk import get_rvk_subject_store
from slub_docsa.experiments.annif.models import default_annif_named_model_list
from slub_docsa.experiments.common.models import initialize_classification_models_from_tuple_list

from slub_docsa.experiments.common.pipeline import do_default_score_matrix_classification_evaluation
from slub_docsa.experiments.common.pipeline import get_split_function_by_name
from slub_docsa.experiments.common.plots import write_default_classification_plots
from slub_docsa.experiments.dummy.models import default_dummy_named_model_list
from slub_docsa.experiments.qucosa.datasets import qucosa_named_datasets
from slub_docsa.experiments.qucosa.models import default_qucosa_named_model_list

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("slub_docsa.data.load.qucosa").setLevel(logging.DEBUG)

    random_state = 123
    load_cached_predictions = True
    stop_after_evaluating_split = 0  # 0, 1, 2, 3, None
    dataset_subset = [
        "qucosa_de_titles_langid_rvk",
        "qucosa_de_abstracts_langid_rvk",
        "qucosa_de_fulltexts_langid_rvk",
    ]
    split_function_name = "random"  # either: random, stratified
    n_splits = 10
    model_name_subset = [
        # ### "random", ####
        "oracle",
        "nihilistic",
        "tfidf 2k knn k=1",
        "tfidf 10k knn k=1",
        "tfidf 40k knn k=1",
        "dbmdz bert sts1 knn k=1",
        "dbmdz bert sts8 knn k=1",
        # "tfidf rforest",
        # "dbmdz bert sts1 rforest",
        # "tfidf scikit mlp",
        "tfidf 2k torch ann",
        "tfidf 10k torch ann",
        "tfidf 40k torch ann",
        # "dbmdz bert sts1 scikit mlp",
        "dbmdz bert sts1 torch ann",
        "dbmdz bert sts8 torch ann",
        # "annif tfidf",
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

    def _model_list_generator(subject_order, subject_hierarchy):
        model_list = default_dummy_named_model_list() \
            + default_qucosa_named_model_list() \
            + default_annif_named_model_list("de", subject_order, subject_hierarchy)
        return initialize_classification_models_from_tuple_list(model_list, model_name_subset)

    evaluation_result = do_default_score_matrix_classification_evaluation(
        named_datasets=qucosa_named_datasets(dataset_subset),
        split_function=get_split_function_by_name(split_function_name, n_splits, random_state),
        named_models_generator=_model_list_generator,
        load_cached_predictions=load_cached_predictions,
        stop_after_evaluating_split=stop_after_evaluating_split,
    )

    write_default_classification_plots(evaluation_result, os.path.join(FIGURES_DIR, "qucosa"), filename_suffix)
