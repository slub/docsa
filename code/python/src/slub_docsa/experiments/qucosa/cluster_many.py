"""Evaluate multiple clustering models for multiple score and datasets."""

# pylint: disable=invalid-name

import logging
import os

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, rand_score, adjusted_rand_score
from sklearn.metrics import homogeneity_score, completeness_score

from scipy.spatial.distance import cosine

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.evaluation.pipeline import score_clustering_models_for_documents
from slub_docsa.evaluation.plotting import score_matrix_box_plot, write_multiple_figure_formats
from slub_docsa.evaluation.score import clustering_membership_score_function, scikit_clustering_label_score_function
from slub_docsa.evaluation.similarity import indexed_document_distance_generator_from_vectorizer, intra_cluster_distance
from slub_docsa.experiments.qucosa.common import default_named_qucosa_datasets
from slub_docsa.experiments.qucosa.common import get_qucosa_tfidf_stemming_vectorizer
from slub_docsa.models.clustering.dummy import RandomClusteringModel
from slub_docsa.models.clustering.scikit import ScikitClusteringModel

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    repeats = 1
    n_clusters = 20

    _, dataset, _ = list(default_named_qucosa_datasets(["qucosa_de_fulltexts_langid_rvk"]))[0]
    vectorizer = get_qucosa_tfidf_stemming_vectorizer(max_features=10000, cache_vectors=True, fit_only_once=True)

    documents = [dataset.documents[i] for i in range(5000)]
    subject_targets = [dataset.subjects[i] for i in range(5000)]

    models = [
        RandomClusteringModel(n_clusters=n_clusters),
        ScikitClusteringModel(KMeans(n_clusters=n_clusters), vectorizer),
        ScikitClusteringModel(
            AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="complete"),
            vectorizer
        ),
        ScikitClusteringModel(
            AgglomerativeClustering(n_clusters=n_clusters, affinity="euclidean", linkage="ward"),
            vectorizer
        ),
    ]

    scores = [
        scikit_clustering_label_score_function(mutual_info_score),
        scikit_clustering_label_score_function(adjusted_mutual_info_score),
        scikit_clustering_label_score_function(rand_score),
        scikit_clustering_label_score_function(adjusted_rand_score),
        scikit_clustering_label_score_function(homogeneity_score),
        scikit_clustering_label_score_function(completeness_score),
        clustering_membership_score_function(
            indexed_document_distance_generator_from_vectorizer(vectorizer, cosine),
            intra_cluster_distance
        )
    ]

    score_matrix = score_clustering_models_for_documents(
        documents,
        subject_targets,
        models,
        scores,
        repeats,
    )

    print(score_matrix)

    figure = score_matrix_box_plot(
        score_matrix,
        ["random", "kmeans", "agg cosine cl", "agg eucl ward"],
        ["mutual info", "adj mutual info", "rand", "adj rand", "homogeneity", "completeness", "intra cluster distance"],
        [(0.0, None), (0.0, None), (None, None), (None, None), (None, None), (None, None), (None, None)],
        columns=2
    )
    write_multiple_figure_formats(figure, os.path.join(FIGURES_DIR, "qucosa_cluster_score_plot"))
