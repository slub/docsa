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
from slub_docsa.models.classification.ann_torch import TorchSingleLayerDenseReluModel
from slub_docsa.models.classification.ann_torch import TorchBertSequenceClassificationHeadModel
from slub_docsa.models.classification.scikit import ScikitClassifier

from slub_docsa.experiments.qucosa.vectorizer import get_qucosa_tfidf_stemming_vectorizer
from slub_docsa.experiments.qucosa.vectorizer import get_qucosa_dbmdz_bert_vectorizer
from slub_docsa.experiments.common.models import NamedClassificationModelTupleList, NamedClusteringModelTupleList
from slub_docsa.models.clustering.scikit import ScikitClusteringModel
from slub_docsa.models.clustering.dummy import RandomClusteringModel


def default_qucosa_named_model_list() -> NamedClassificationModelTupleList:
    """Return a list of default qucosa models to use for evaluating model performance."""
    tfidf_vectorizer_2k = get_qucosa_tfidf_stemming_vectorizer(max_features=2000)
    tfidf_vectorizer_10k = get_qucosa_tfidf_stemming_vectorizer(max_features=10000)
    tfidf_vectorizer_40k = get_qucosa_tfidf_stemming_vectorizer(max_features=40000)
    dbmdz_bert_vectorizer_sts_1 = get_qucosa_dbmdz_bert_vectorizer(1)
    dbmdz_bert_vectorizer_sts_8 = get_qucosa_dbmdz_bert_vectorizer(8)

    models = [
        ("tfidf 2k knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=tfidf_vectorizer_2k,
        )),
        ("tfidf 10k knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=tfidf_vectorizer_10k,
        )),
        ("tfidf 40k knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=tfidf_vectorizer_40k,
        )),
        ("dbmdz bert sts1 knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=dbmdz_bert_vectorizer_sts_1,
        )),
        ("dbmdz bert sts8 knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=dbmdz_bert_vectorizer_sts_8,
        )),
        ("random vectorizer knn k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=RandomVectorizer(),
        )),
        ("tfidf 10k knn k=3", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=3),
            vectorizer=tfidf_vectorizer_10k,
        )),
        ("tfidf 10k dtree", lambda: ScikitClassifier(
            predictor=DecisionTreeClassifier(max_leaf_nodes=1000),
            vectorizer=tfidf_vectorizer_10k,
        )),
        ("tfidf 10k rforest", lambda: ScikitClassifier(
            predictor=RandomForestClassifier(n_jobs=-1, max_leaf_nodes=1000),
            vectorizer=tfidf_vectorizer_10k,
        )),
        ("dbmdz bert sts1 rforest", lambda: ScikitClassifier(
            predictor=RandomForestClassifier(n_jobs=-1, max_leaf_nodes=1000),
            vectorizer=dbmdz_bert_vectorizer_sts_1,
        )),
        ("tfidf 10k scikit mlp", lambda: ScikitClassifier(
            predictor=MLPClassifier(max_iter=10),
            vectorizer=tfidf_vectorizer_10k,
        )),
        ("tfidf 2k torch ann", lambda: TorchSingleLayerDenseReluModel(
            epochs=50,
            vectorizer=tfidf_vectorizer_2k,
        )),
        ("tfidf 10k torch ann", lambda: TorchSingleLayerDenseReluModel(
            epochs=50,
            vectorizer=tfidf_vectorizer_10k,
        )),
        ("tfidf 40k torch ann", lambda: TorchSingleLayerDenseReluModel(
            epochs=50,
            vectorizer=tfidf_vectorizer_40k,
        )),
        ("dbmdz bert sts1 scikit mlp", lambda: ScikitClassifier(
            predictor=MLPClassifier(max_iter=10),
            vectorizer=dbmdz_bert_vectorizer_sts_1,
        )),
        ("dbmdz bert sts1 torch ann", lambda: TorchBertSequenceClassificationHeadModel(
            epochs=50,
            vectorizer=dbmdz_bert_vectorizer_sts_1,
        )),
        ("dbmdz bert sts8 torch ann", lambda: TorchBertSequenceClassificationHeadModel(
            epochs=50,
            vectorizer=dbmdz_bert_vectorizer_sts_8,
        )),
        ("tfidf 10k log_reg", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(estimator=LogisticRegression()),
            vectorizer=tfidf_vectorizer_10k,
        )),
        ("tfidf 10k nbayes", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(estimator=GaussianNB()),
            vectorizer=tfidf_vectorizer_10k,
        )),
        ("tfidf 10k svc", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(
                estimator=CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)
            ),
            vectorizer=tfidf_vectorizer_10k,
        )),
    ]

    return models


def default_qucosa_named_clustering_models_tuple_list(
    n_subjects: int,
) -> NamedClusteringModelTupleList:
    """Return default qucosa clustering models."""
    tfidf_vectorizer_10k = get_qucosa_tfidf_stemming_vectorizer(max_features=10000)
    dbmdz_bert_vectorizer_sts_1 = get_qucosa_dbmdz_bert_vectorizer(1)

    models = [
        ("random c=20", lambda: RandomClusteringModel(n_clusters=20)),
        ("random c=subjects", lambda: RandomClusteringModel(n_clusters=n_subjects)),
        ("tfidf 10k kMeans c=20", lambda: ScikitClusteringModel(
            model=MiniBatchKMeans(n_clusters=20),
            vectorizer=tfidf_vectorizer_10k
        )),
        ("tfidf 10k kMeans c=subjects", lambda: ScikitClusteringModel(
            model=MiniBatchKMeans(n_clusters=n_subjects),
            vectorizer=tfidf_vectorizer_10k
        )),
        ("tfidf 10k agg sl cosine c=subjects", lambda: ScikitClusteringModel(
            model=AgglomerativeClustering(n_clusters=n_subjects, affinity="cosine", linkage="single"),
            vectorizer=tfidf_vectorizer_10k
        )),
        ("tfidf 10k agg sl eucl c=subjects", lambda: ScikitClusteringModel(
            model=AgglomerativeClustering(n_clusters=n_subjects, affinity="euclidean", linkage="single"),
            vectorizer=tfidf_vectorizer_10k
        )),
        ("tfidf 10k agg ward eucl c=subjects", lambda: ScikitClusteringModel(
            model=AgglomerativeClustering(n_clusters=n_subjects, affinity="euclidean", linkage="ward"),
            vectorizer=tfidf_vectorizer_10k
        )),
        ("bert kMeans c=20", lambda: ScikitClusteringModel(
            model=MiniBatchKMeans(n_clusters=20),
            vectorizer=dbmdz_bert_vectorizer_sts_1
        )),
        ("bert kMeans c=subjects", lambda: ScikitClusteringModel(
            model=MiniBatchKMeans(n_clusters=n_subjects),
            vectorizer=dbmdz_bert_vectorizer_sts_1
        )),
        ("bert kMeans agg sl eucl c=subjects", lambda: ScikitClusteringModel(
            model=AgglomerativeClustering(n_clusters=n_subjects, affinity="euclidean", linkage="single"),
            vectorizer=dbmdz_bert_vectorizer_sts_1
        )),
        ("bert kMeans agg ward eucl c=subjects", lambda: ScikitClusteringModel(
            model=AgglomerativeClustering(n_clusters=n_subjects, affinity="euclidean", linkage="ward"),
            vectorizer=dbmdz_bert_vectorizer_sts_1
        )),
    ]

    return models
