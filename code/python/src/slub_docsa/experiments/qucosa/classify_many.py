"""Evaluates and compares multiple models and multiple variants of the Qucosa dataset."""

# pylint: disable=invalid-name, too-many-arguments

import logging
import os
from typing import List, Optional, Union
from typing_extensions import Literal

from slub_docsa.common.paths import get_figures_dir
from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets
from slub_docsa.experiments.common.models import filter_model_type_mapping

from slub_docsa.experiments.common.pipeline import do_default_score_matrix_classification_evaluation
from slub_docsa.experiments.common.plots import write_default_classification_plots
from slub_docsa.experiments.dummy.models import default_dummy_model_types
from slub_docsa.experiments.qucosa.datasets import qucosa_named_sample_generators
from slub_docsa.serve.models.classification.common import get_all_classification_model_types

logger = logging.getLogger(__name__)


def qucosa_experiments_classify_many(
    dataset_subset: List[str],
    model_subset: List[str],
    n_splits: int = 10,
    load_cached_scores: bool = False,
    random_state: Optional[int] = None,
    split_function_name: Union[Literal["random"], Literal["stratified"]] = "random",
    stop_after_evaluating_split: Optional[int] = None,
    check_qucosa_download: bool = False,
):
    """Perform qucosa experiments comparing many classification models for many dataset variants."""
    filename_suffix = f"split={split_function_name}"

    model_types = default_dummy_model_types()
    model_types.update(get_all_classification_model_types())
    model_types = filter_model_type_mapping(model_types, model_subset)

    named_datasets = filter_and_cache_named_datasets(
        qucosa_named_sample_generators(check_qucosa_download), dataset_subset
    )

    evaluation_result = do_default_score_matrix_classification_evaluation(
        named_datasets=named_datasets,
        model_types=model_types,
        split_function_name=split_function_name,
        split_number=n_splits,
        load_cached_scores=load_cached_scores,
        stop_after_evaluating_split=stop_after_evaluating_split,
        random_state=random_state,
    )

    write_default_classification_plots(evaluation_result, os.path.join(get_figures_dir(), "qucosa"), filename_suffix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("slub_docsa.data.load.qucosa").setLevel(logging.DEBUG)

    qucosa_experiments_classify_many(
        dataset_subset=[
            # "qucosa_de_titles_langid_rvk",
            # "qucosa_de_abstracts_langid_rvk",
            # "qucosa_de_fulltexts_langid_rvk",
            # "qucosa_de_titles_langid_ddc",
            # "qucosa_de_abstracts_langid_ddc",
            # "qucosa_de_fulltexts_langid_ddc",
            "qucosa_de_complete_but_only_titles_rvk",
            "qucosa_de_complete_but_only_abstracts_rvk",
            # "qucosa_de_complete_but_only_fulltexts_rvk",
        ],
        model_subset=[
            # ### "random", ####
            "oracle",
            "nihilistic",
            # "tfidf_2k_knn_k=1",
            "tfidf_10k_knn_k=1",
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
            "annif_omikuji_de",
            # "annif_vw_multi",
            # "annif_mllm",
            # "annif_fasttext"
            # "annif_yake",
            # ### "annif_stwfsa" ###
        ],
        n_splits=10,
        load_cached_scores=True,
        random_state=123,
        split_function_name="random",
        stop_after_evaluating_split=0
    )
