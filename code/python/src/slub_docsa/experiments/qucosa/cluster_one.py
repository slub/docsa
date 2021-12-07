"""Clustering experiments for qucosa data."""

# pylint: disable=invalid-name

import logging
import os

import numpy as np

from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from slub_docsa.common.paths import get_figures_dir
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level, subject_label_breadcrumb
from slub_docsa.evaluation.incidence import membership_matrix_to_crisp_cluster_assignments, unique_subject_order
from slub_docsa.evaluation.plotting import cluster_distribution_by_subject_plot, subject_distribution_by_cluster_plot
from slub_docsa.evaluation.plotting import write_multiple_figure_formats

from slub_docsa.evaluation.score import clustering_membership_score_function
from slub_docsa.evaluation.similarity import indexed_document_distance_generator_from_vectorizer
from slub_docsa.evaluation.similarity import intra_cluster_distance
from slub_docsa.experiments.qucosa.vectorizer import get_qucosa_tfidf_stemming_vectorizer
from slub_docsa.experiments.qucosa.datasets import qucosa_named_datasets
from slub_docsa.models.clustering.scikit import ScikitClusteringModel

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    max_documents = 5000
    n_clusters = 50

    _, dataset, subject_hierarchy = list(qucosa_named_datasets(["qucosa_de_fulltexts_langid_ddc"]))[0]
    vectorizer = get_qucosa_tfidf_stemming_vectorizer(max_features=10000, cache_vectors=True, fit_only_once=True)

    sampled_idx = np.random.choice(range(len(dataset.documents)), size=max_documents, replace=False)
    sampled_documents = [dataset.documents[i] for i in sampled_idx]
    sampled_subject_targets = [dataset.subjects[i] for i in sampled_idx]
    sampled_subject_targets = prune_subject_targets_to_level(2, sampled_subject_targets, subject_hierarchy)
    sampled_unique_subject_order = unique_subject_order(sampled_subject_targets)

    indexed_distance_generator = indexed_document_distance_generator_from_vectorizer(vectorizer, cosine)
    cluster_score = clustering_membership_score_function(indexed_distance_generator, intra_cluster_distance)

    model = ScikitClusteringModel(KMeans(n_clusters=n_clusters), vectorizer)

    model.fit(sampled_documents)
    membership = model.predict(sampled_documents)

    score = cluster_score(sampled_documents, membership, sampled_subject_targets)
    logger.info("clustering score is: %f", score)

    cluster_assignments = membership_matrix_to_crisp_cluster_assignments(membership)

    document_labels = [d.title for d in sampled_documents]
    subject_labels = {
        s: subject_label_breadcrumb(subject_hierarchy[s], subject_hierarchy) for s in sampled_unique_subject_order
    }

    fig = subject_distribution_by_cluster_plot(
        cluster_assignments,
        document_labels,
        sampled_subject_targets,
        subject_labels,
    )

    write_multiple_figure_formats(fig, os.path.join(get_figures_dir(), "qucosa/subject_distribution_by_cluster"))

    fig = cluster_distribution_by_subject_plot(
        cluster_assignments,
        document_labels,
        sampled_subject_targets,
        subject_labels,
    )

    write_multiple_figure_formats(fig, os.path.join(get_figures_dir(), "qucosa/cluster_distribution_by_subject"))
