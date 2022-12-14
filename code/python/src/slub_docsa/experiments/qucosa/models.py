"""Common models that are evaluated when experimenting with qucosa and artifical datasets."""

# pylint: disable=unnecessary-lambda

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

from slub_docsa.experiments.common.vectorizer import get_cached_tfidf_stemming_vectorizer
from slub_docsa.experiments.common.models import NamedClusteringModelTupleList
from slub_docsa.models.clustering.scikit import ScikitClusteringModel
from slub_docsa.models.clustering.dummy import RandomClusteringModel
from slub_docsa.serve.models.classification.dbmdz import _get_cached_dbmdz_bert_vectorizer


def default_qucosa_named_clustering_models_tuple_list(
    n_subjects: int,
) -> NamedClusteringModelTupleList:
    """Return default qucosa clustering models."""
    tfidf_vectorizer_10k = get_cached_tfidf_stemming_vectorizer(max_features=10000)
    dbmdz_bert_vectorizer_sts_1 = _get_cached_dbmdz_bert_vectorizer(subtext_samples=1)

    models = [
        ("random_c=20", lambda: RandomClusteringModel(n_clusters=20)),
        ("random_c=subjects", lambda: RandomClusteringModel(n_clusters=n_subjects)),
        ("tfidf_10k_kMeans_c=20", lambda: ScikitClusteringModel(
            model=MiniBatchKMeans(n_clusters=20),
            vectorizer=tfidf_vectorizer_10k
        )),
        ("tfidf_10k_kMeans_c=subjects", lambda: ScikitClusteringModel(
            model=MiniBatchKMeans(n_clusters=n_subjects),
            vectorizer=tfidf_vectorizer_10k
        )),
        ("tfidf_10k_agg_sl_cosine_c=subjects", lambda: ScikitClusteringModel(
            model=AgglomerativeClustering(n_clusters=n_subjects, affinity="cosine", linkage="single"),
            vectorizer=tfidf_vectorizer_10k
        )),
        ("tfidf_10k_agg_sl_eucl_c=subjects", lambda: ScikitClusteringModel(
            model=AgglomerativeClustering(n_clusters=n_subjects, affinity="euclidean", linkage="single"),
            vectorizer=tfidf_vectorizer_10k
        )),
        ("tfidf_10k_agg_ward_eucl_c=subjects", lambda: ScikitClusteringModel(
            model=AgglomerativeClustering(n_clusters=n_subjects, affinity="euclidean", linkage="ward"),
            vectorizer=tfidf_vectorizer_10k
        )),
        ("bert_kMeans_c=20", lambda: ScikitClusteringModel(
            model=MiniBatchKMeans(n_clusters=20),
            vectorizer=dbmdz_bert_vectorizer_sts_1
        )),
        ("bert_kMeans_c=subjects", lambda: ScikitClusteringModel(
            model=MiniBatchKMeans(n_clusters=n_subjects),
            vectorizer=dbmdz_bert_vectorizer_sts_1
        )),
        ("bert_kMeans_agg_sl_eucl_c=subjects", lambda: ScikitClusteringModel(
            model=AgglomerativeClustering(n_clusters=n_subjects, affinity="euclidean", linkage="single"),
            vectorizer=dbmdz_bert_vectorizer_sts_1
        )),
        ("bert_kMeans_agg_ward_eucl_c=subjects", lambda: ScikitClusteringModel(
            model=AgglomerativeClustering(n_clusters=n_subjects, affinity="euclidean", linkage="ward"),
            vectorizer=dbmdz_bert_vectorizer_sts_1
        )),
    ]

    return models
