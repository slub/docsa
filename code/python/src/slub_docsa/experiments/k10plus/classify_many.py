"""Evaluates and compares multiple models for the k10plus dataset."""

# pylint: disable=invalid-name, too-many-arguments

import logging
import os
from typing import List, Optional, Union
from typing_extensions import Literal

from slub_docsa.common.paths import get_figures_dir
from slub_docsa.experiments.annif.models import default_annif_named_model_list
from slub_docsa.experiments.common.models import initialize_classification_models_from_tuple_list
from slub_docsa.experiments.common.pipeline import do_default_score_matrix_classification_evaluation
from slub_docsa.experiments.common.pipeline import get_split_function_by_name
from slub_docsa.experiments.common.plots import write_default_classification_plots
from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets
from slub_docsa.experiments.dummy.models import default_dummy_named_model_list
from slub_docsa.experiments.qucosa.models import default_qucosa_named_classification_model_list
from slub_docsa.experiments.k10plus.datasets import k10plus_named_datasets_tuple_list

logger = logging.getLogger(__name__)


def k10plus_experiments_classify_many(
    language: str,
    model_subset: List[str],
    dataset_subset: Optional[List[str]],
    n_splits: int = 10,
    # load_cached_predictions: bool = False,
    random_state: Optional[int] = None,
    split_function_name: Union[Literal["random"], Literal["stratified"]] = "random",
    stop_after_evaluating_split: Optional[int] = None,
):
    """Perform k10plus experiments comparing many classification models for many dataset variants."""
    filename_suffix = f"split={split_function_name}"

    def _model_list_generator(subject_order, subject_hierarchy):
        model_list = default_dummy_named_model_list() \
            + default_qucosa_named_classification_model_list() \
            + default_annif_named_model_list(language, subject_order, subject_hierarchy)
        return initialize_classification_models_from_tuple_list(model_list, model_subset)

    named_datasets = filter_and_cache_named_datasets(
        k10plus_named_datasets_tuple_list(schemas=["rvk", "ddc"], languages=[language], variants=["public"]),
        dataset_subset
    )

    evaluation_result = do_default_score_matrix_classification_evaluation(
        named_datasets=named_datasets,
        split_function=get_split_function_by_name(split_function_name, n_splits, random_state),
        named_models_generator=_model_list_generator,
        n_splits=n_splits,
        # load_cached_predictions=load_cached_predictions,
        stop_after_evaluating_split=stop_after_evaluating_split,
    )

    write_default_classification_plots(evaluation_result, os.path.join(get_figures_dir(), "k10plus"), filename_suffix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    k10plus_experiments_classify_many(
        language="de",
        model_subset=[
            # ### "random", ####
            "oracle",
            "nihilistic",
            # "tfidf_2k_knn_k=1",
            # "tfidf_10k_knn_k=1",
            # "tfidf_40k_knn_k=1",
            # "dbmdz_bert_sts1_knn_k=1",
            # "dbmdz_bert_sts8_knn_k=1",
            # "tfidf_rforest",
            # "dbmdz_bert_sts1_rforest",
            # "tfidf_scikit_mlp",
            # "tfidf_2k_torch_ann",
            # "tfidf_10k_torch_ann",
            # "tfidf_40k_torch_ann",
            # "dbmdz_bert_sts1_scikit_mlp",
            # "dbmdz_bert_sts1_torch_ann",
            # "dbmdz_bert_sts8_torch_ann",
            # "annif_tfidf",
            # "annif_svc",
            "annif_omikuji",
            # "annif_vw_multi",
            # "annif_mllm",
            # "annif_fasttext"
            # "annif_yake",
            # ### "annif_stwfsa" ###
        ],
        dataset_subset=None,
        n_splits=10,
        # load_cached_predictions=True,
        random_state=123,
        split_function_name="random",
        stop_after_evaluating_split=0
    )
