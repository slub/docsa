"""Common models that are evaluated when experimenting with qucosa and artifical datasets."""

# pylint: disable=unnecessary-lambda

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from slub_docsa.data.preprocess.vectorizer import ScikitTfidfVectorizer
from slub_docsa.models.classification.scikit import ScikitClassifier

from slub_docsa.experiments.common.models import NamedClassificationModelTupleList


def default_artificial_named_classification_model_list() -> NamedClassificationModelTupleList:
    """Return a list of default qucosa models to use for evaluating model performance."""
    models = [
        ("tfidf_10k_knn_k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=ScikitTfidfVectorizer(max_features=10000),
        )),
        ("tfidf_10k_knn_k=3", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=3),
            vectorizer=ScikitTfidfVectorizer(max_features=10000),
        )),
        ("tfidf_10k_dtree", lambda: ScikitClassifier(
            predictor=DecisionTreeClassifier(max_leaf_nodes=1000),
            vectorizer=ScikitTfidfVectorizer(max_features=10000),
        )),
        ("tfidf_10k_rforest", lambda: ScikitClassifier(
            predictor=RandomForestClassifier(n_jobs=-1, max_leaf_nodes=1000),
            vectorizer=ScikitTfidfVectorizer(max_features=10000),
        )),
        ("tfidf_10k_log_reg", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(estimator=LogisticRegression()),
            vectorizer=ScikitTfidfVectorizer(max_features=10000),
        )),
        ("tfidf_10k_nbayes", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(estimator=GaussianNB()),
            vectorizer=ScikitTfidfVectorizer(max_features=10000),
        )),
        ("tfidf_10k_svc", lambda: ScikitClassifier(
            predictor=MultiOutputClassifier(
                estimator=CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)
            ),
            vectorizer=ScikitTfidfVectorizer(max_features=10000),
        )),
    ]

    return models
