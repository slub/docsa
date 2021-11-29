"""Clustering experiments for qucosa data."""

# pylint: disable=invalid-name

import logging

from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans

from slub_docsa.evaluation.score import clustering_membership_score_function
from slub_docsa.evaluation.similarity import indexed_document_distance_generator_from_vectorizer
from slub_docsa.evaluation.similarity import intra_cluster_distance
from slub_docsa.experiments.qucosa.vectorizer import get_qucosa_tfidf_stemming_vectorizer
from slub_docsa.experiments.qucosa.datasets import qucosa_named_datasets
from slub_docsa.models.clustering.scikit import ScikitClusteringModel

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    _, dataset, _ = list(qucosa_named_datasets(["qucosa_de_fulltexts_langid_rvk"]))[0]
    vectorizer = get_qucosa_tfidf_stemming_vectorizer(max_features=10000, cache_vectors=True, fit_only_once=True)

    documents = dataset.documents  # [dataset.documents[i] for i in range(1000)]
    subject_targets = dataset.subjects  # [dataset.subjects[i] for i in range(1000)]
    indexed_distance_generator = indexed_document_distance_generator_from_vectorizer(vectorizer, cosine)
    cluster_score = clustering_membership_score_function(indexed_distance_generator, intra_cluster_distance)

    model = ScikitClusteringModel(KMeans(n_clusters=10), vectorizer)

    model.fit(documents)
    membership = model.predict(documents)

    score = cluster_score(documents, membership, subject_targets)
    print(score)
