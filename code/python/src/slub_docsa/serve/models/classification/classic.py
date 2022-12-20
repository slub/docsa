"""Setup classic classification models."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

from slub_docsa.data.preprocess.vectorizer import GensimTfidfVectorizer, ScikitTfidfVectorizer, StemmingVectorizer
from slub_docsa.models.classification.scikit import ScikitClassifier
from slub_docsa.serve.common import ModelTypeMapping


def get_classic_classification_models_map() -> ModelTypeMapping:
    """Return a map of classification model types and their generator functions."""
    def get_tfidf_stemming_vectorizer_de():
        return StemmingVectorizer(vectorizer=GensimTfidfVectorizer(max_features=10000), lang_code="de")

    return {
        "tfidf_10k_knn_k=1": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1, metric="cosine"),
            vectorizer=ScikitTfidfVectorizer(max_features=10000),
        ),
        "tfidf_snowball_de_10k_knn_k=1": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1, metric="cosine"),
            vectorizer=get_tfidf_stemming_vectorizer_de(),
        ),
        "tfidf_snowball_de_10k_knn_k=3": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance"),
            vectorizer=get_tfidf_stemming_vectorizer_de(),
        ),
        "tfidf_snowball_de_10k_dtree": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=DecisionTreeClassifier(max_leaf_nodes=1000),
            vectorizer=get_tfidf_stemming_vectorizer_de(),
        ),
        "tfidf_snowball_de_10k_rforest": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=RandomForestClassifier(n_jobs=-1, n_estimators=50, max_leaf_nodes=1000),
            vectorizer=get_tfidf_stemming_vectorizer_de(),
        ),
        "tfidf_snowball_de_10k_log_reg": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=MultiOutputClassifier(estimator=LogisticRegression()),
            vectorizer=get_tfidf_stemming_vectorizer_de(),
        ),
        "tfidf_snowball_de_10k_nbayes": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=MultiOutputClassifier(estimator=GaussianNB()),
            vectorizer=get_tfidf_stemming_vectorizer_de(),
        ),
        "tfidf_snowball_de_10k_svc": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=MultiOutputClassifier(
                estimator=CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)
            ),
            vectorizer=get_tfidf_stemming_vectorizer_de(),
        ),
    }
