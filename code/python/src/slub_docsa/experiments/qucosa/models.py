"""Common models that are evaluated when experimenting with qucosa and artifical datasets."""

# pylint: disable=unnecessary-lambda

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

from slub_docsa.data.preprocess.vectorizer import RandomVectorizer
from slub_docsa.models.classification.ann.dense import TorchSingleLayerDenseReluModel
from slub_docsa.models.classification.ann.pretrained import TorchBertSequenceClassificationHeadModel
from slub_docsa.models.classification.scikit import ScikitClassifier

from slub_docsa.experiments.common.vectorizer import get_cached_tfidf_stemming_vectorizer, get_cached_tfidf_vectorizer
from slub_docsa.experiments.common.vectorizer import get_cached_dbmdz_bert_vectorizer
from slub_docsa.experiments.common.models import NamedClassificationModelTupleList, NamedClusteringModelTupleList
from slub_docsa.models.clustering.scikit import ScikitClusteringModel
from slub_docsa.models.clustering.dummy import RandomClusteringModel


def default_qucosa_named_classification_model_list() -> NamedClassificationModelTupleList:
    """Return a list of default qucosa models to use for evaluating model performance."""
    models = [
        ("tfidf_2k_knn_k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=2000),
        )),
        ("tfidf_10k_knn_k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=10000),
        )),
        ("tfidf_no_stem_10k_knn_k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=get_cached_tfidf_vectorizer(max_features=10000),
        )),
        ("tfidf_40k_knn_k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=40000),
        )),
        ("dbmdz_bert_sts1_knn_k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=get_cached_dbmdz_bert_vectorizer(subtext_samples=1),
        )),
        ("dbmdz_bert_sts8_knn_k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=get_cached_dbmdz_bert_vectorizer(subtext_samples=8),
        )),
        ("random_vectorizer_knn_k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=RandomVectorizer(),
        )),
        ("tfidf_10k_knn_k=3", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=3),
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=10000),
        )),
        ("tfidf_10k_dtree", lambda: ScikitClassifier(
            predictor=DecisionTreeClassifier(max_leaf_nodes=1000),
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=10000),
        )),
        ("tfidf_10k_rforest", lambda: ScikitClassifier(
            predictor=RandomForestClassifier(n_jobs=-1, max_leaf_nodes=1000),
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=10000),
        )),
        ("dbmdz_bert_sts1_rforest", lambda: ScikitClassifier(
            predictor=RandomForestClassifier(n_jobs=-1, max_leaf_nodes=1000),
            vectorizer=get_cached_dbmdz_bert_vectorizer(subtext_samples=1),
        )),
        ("tfidf_10k_scikit_mlp", lambda: ScikitClassifier(
            predictor=MLPClassifier(max_iter=10),
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=10000),
        )),
        ("tfidf_2k_torch_ann", lambda: TorchSingleLayerDenseReluModel(
            epochs=50,
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=2000),
        )),
        ("tfidf_10k_torch_ann", lambda: TorchSingleLayerDenseReluModel(
            epochs=50,
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=10000),
        )),
        ("tfidf_no_stem_10k_torch_ann", lambda: TorchSingleLayerDenseReluModel(
            epochs=50,
            vectorizer=get_cached_tfidf_vectorizer(max_features=10000),
        )),
        ("tfidf_40k_torch_ann", lambda: TorchSingleLayerDenseReluModel(
            epochs=50,
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=40000),
        )),
        ("dbmdz_bert_sts1_scikit_mlp", lambda: ScikitClassifier(
            predictor=MLPClassifier(max_iter=10),
            vectorizer=get_cached_dbmdz_bert_vectorizer(subtext_samples=1),
        )),
        ("dbmdz_bert_sts1_torch_ann", lambda: TorchBertSequenceClassificationHeadModel(
            epochs=50,
            vectorizer=get_cached_dbmdz_bert_vectorizer(subtext_samples=1),
        )),
        ("dbmdz_bert_sts8_torch_ann", lambda: TorchBertSequenceClassificationHeadModel(
            epochs=50,
            vectorizer=get_cached_dbmdz_bert_vectorizer(subtext_samples=8),
        )),
        ("tfidf_10k_log_reg", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(estimator=LogisticRegression()),
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=10000),
        )),
        ("tfidf_10k_nbayes", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(estimator=GaussianNB()),
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=10000),
        )),
        ("tfidf_10k_svc", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(
                estimator=CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)
            ),
            vectorizer=get_cached_tfidf_stemming_vectorizer(max_features=10000),
        )),
    ]

    return models


def default_qucosa_named_clustering_models_tuple_list(
    n_subjects: int,
) -> NamedClusteringModelTupleList:
    """Return default qucosa clustering models."""
    tfidf_vectorizer_10k = get_cached_tfidf_stemming_vectorizer(max_features=10000)
    dbmdz_bert_vectorizer_sts_1 = get_cached_dbmdz_bert_vectorizer(subtext_samples=1)

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
