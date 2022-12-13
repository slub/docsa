"""Evaluates and compares multiple models for the k10plus dataset."""

# pylint: disable=invalid-name, too-many-arguments

import logging
import os
from typing import List, Optional, Tuple, Union
from typing_extensions import Literal

from slub_docsa.common.paths import get_figures_dir
from slub_docsa.experiments.common.models import filter_model_type_mapping
from slub_docsa.experiments.common.pipeline import do_default_score_matrix_classification_evaluation
from slub_docsa.experiments.common.plots import write_default_classification_plots
from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets
from slub_docsa.experiments.dummy.models import default_dummy_model_types
from slub_docsa.experiments.k10plus.datasets import k10plus_named_sample_generators
from slub_docsa.serve.models.classification.common import get_all_classification_model_types

logger = logging.getLogger(__name__)


def k10plus_experiments_classify_many(
    language: str,
    model_subset: List[str],
    dataset_schemas: List[str],
    dataset_variants: List[Tuple[str, int]],
    n_splits: int = 10,
    load_cached_scores: bool = False,
    random_state: Optional[int] = None,
    split_function_name: Union[Literal["random"], Literal["stratified"]] = "random",
    stop_after_evaluating_split: Optional[int] = None,
):
    """Perform k10plus experiments comparing many classification models for many dataset variants."""
    filename_suffix = f"split={split_function_name}"

    model_types = default_dummy_model_types()
    model_types.update(get_all_classification_model_types())
    model_types = filter_model_type_mapping(model_types, model_subset)

    named_datasets = filter_and_cache_named_datasets(
        k10plus_named_sample_generators(
            schemas=dataset_schemas, languages=[language], variants=dataset_variants
        ),
        None
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

    write_default_classification_plots(evaluation_result, os.path.join(get_figures_dir(), "k10plus"), filename_suffix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("sqlitedict").setLevel(logging.WARNING)

    k10plus_experiments_classify_many(
        language="de",
        model_subset=[
            # ### "random", ####
            "oracle",
            "nihilistic",
            "tfidf_10k_knn_k=1",
            "dbmdz_bert_sts1_knn_k=1",
            # "dbmdz_bert_sts1_rforest",
            # "tfidf_10k_rforest",
            "tfidf_10k_torch_ann",
            "dbmdz_bert_sts1_torch_ann",
            "tiny_bert_torch_ann",
            "annif_tfidf_de",
            "annif_svc_de",
            "annif_omikuji_de",
            "annif_mllm_de",
            "annif_fasttext_de"
            "annif_yake_de",
        ],
        dataset_schemas=["rvk"],
        dataset_variants=[("public", 10000)],  # , ("slub_raw", 10000), ("slub_clean", 10000)],
        n_splits=10,
        load_cached_scores=True,
        random_state=123,
        split_function_name="random",
        stop_after_evaluating_split=0
    )
