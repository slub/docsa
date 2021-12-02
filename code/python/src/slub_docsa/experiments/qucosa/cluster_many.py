"""Evaluate multiple clustering models for multiple score and datasets."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.experiments.common.models import initialize_clustering_models_from_tuple_list
from slub_docsa.experiments.common.pipeline import do_default_score_matrix_clustering_evaluation
from slub_docsa.experiments.common.plots import write_default_clustering_plots
from slub_docsa.experiments.common.scores import default_named_clustering_score_list, initialize_named_score_tuple_list

from slub_docsa.experiments.qucosa.datasets import qucosa_named_datasets
from slub_docsa.experiments.qucosa.models import default_qucosa_named_clustering_models_tuple_list
from slub_docsa.experiments.qucosa.vectorizer import get_qucosa_tfidf_stemming_vectorizer


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    repeats = 10
    max_documents = 5000

    dataset_subset = [
        "qucosa_de_titles_langid_rvk",
        "qucosa_de_abstracts_langid_rvk",
        "qucosa_de_fulltexts_langid_rvk",
    ]
    model_subset = [
        "random c=20",
        "random c=subjects",
        "tfidf 10k kMeans c=20",
        "tfidf 10k kMeans c=subjects",
        "tfidf 10k agg sl cosine c=subjects",
        "tfidf 10k agg sl eucl c=subjects",
        "tfidf 10k agg ward eucl c=subjects",
        "bert kMeans c=20",
        "bert kMeans c=subjects",
        "bert kMeans agg sl eucl c=subjects",
        "bert kMeans agg ward eucl c=subjects"
    ]

    def _model_generator(subject_order):
        return initialize_clustering_models_from_tuple_list(
            default_qucosa_named_clustering_models_tuple_list(n_subjects=len(subject_order)),
            model_subset
        )

    def _score_generator():
        vectorizer = get_qucosa_tfidf_stemming_vectorizer(max_features=10000, cache_vectors=True, fit_only_once=True)
        return initialize_named_score_tuple_list(
            default_named_clustering_score_list(vectorizer)
        )

    evaluation_result = do_default_score_matrix_clustering_evaluation(
        named_datasets=qucosa_named_datasets(dataset_subset),
        named_models_generator=_model_generator,
        named_scores_generator=_score_generator,
        repeats=repeats,
        max_documents=max_documents
    )

    write_default_clustering_plots(
        evaluation_result,
        os.path.join(FIGURES_DIR, "qucosa/"),
        f"repeats={repeats}_max_docs={max_documents}"
    )
