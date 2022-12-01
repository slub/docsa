"""Evaluate multiple clustering models for multiple score and datasets."""

# pylint: disable=invalid-name

import logging
import os
from typing import Sequence, Optional

from slub_docsa.common.paths import get_figures_dir
from slub_docsa.experiments.common.models import initialize_clustering_models_from_tuple_list
from slub_docsa.experiments.common.pipeline import do_default_score_matrix_clustering_evaluation
from slub_docsa.experiments.common.plots import write_default_clustering_plots
from slub_docsa.experiments.common.scores import default_named_clustering_score_list, initialize_named_score_tuple_list

from slub_docsa.experiments.qucosa.datasets import qucosa_named_datasets
from slub_docsa.experiments.qucosa.models import default_qucosa_named_clustering_models_tuple_list
from slub_docsa.experiments.common.vectorizer import get_cached_tfidf_stemming_vectorizer

logger = logging.getLogger(__name__)


def qucosa_experiments_cluster_many(
    dataset_subset: Sequence[str],
    model_subset: Sequence[str],
    repeats: int = 10,
    max_documents: Optional[int] = None,
    check_qucosa_download: bool = False,
):
    """Perform qucosa clustering experiment comparing multiple models and dataset variants."""
    def _model_generator(subject_order):
        return initialize_clustering_models_from_tuple_list(
            default_qucosa_named_clustering_models_tuple_list(n_subjects=len(subject_order)),
            model_subset
        )

    def _score_generator():
        vectorizer = get_cached_tfidf_stemming_vectorizer(max_features=10000, fit_only_once=True)
        return initialize_named_score_tuple_list(
            default_named_clustering_score_list(vectorizer)
        )

    evaluation_result = do_default_score_matrix_clustering_evaluation(
        named_datasets=qucosa_named_datasets(dataset_subset, check_qucosa_download),
        named_models_generator=_model_generator,
        named_scores_generator=_score_generator,
        repeats=repeats,
        max_documents=max_documents
    )

    write_default_clustering_plots(
        evaluation_result,
        os.path.join(get_figures_dir(), "qucosa/"),
        f"repeats={repeats}_max_docs={max_documents}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    qucosa_experiments_cluster_many(
        dataset_subset=[
            # "qucosa_de_titles_langid_rvk",
            # "qucosa_de_abstracts_langid_rvk",
            # "qucosa_de_fulltexts_langid_rvk",
            "qucosa_de_titles_langid_ddc",
            "qucosa_de_abstracts_langid_ddc",
            "qucosa_de_fulltexts_langid_ddc",
        ],
        model_subset=[
            "random_c=20",
            "random_c=subjects",
            "tfidf_10k_kMeans_c=20",
            "tfidf_10k_kMeans_c=subjects",
            "tfidf_10k_agg_sl_cosine_c=subjects",
            "tfidf_10k_agg_sl_eucl_c=subjects",
            "tfidf_10k_agg_ward_eucl_c=subjects",
            "bert_kMeans_c=20",
            "bert_kMeans_c=subjects",
            "bert_kMeans_agg_sl_eucl_c=subjects",
            "bert_kMeans_agg_ward_eucl_c=subjects"
        ],
        repeats=5,
        max_documents=5000
    )
