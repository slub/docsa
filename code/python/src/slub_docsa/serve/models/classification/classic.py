"""Setup classic classification models."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from slub_docsa.data.preprocess.vectorizer import TfidfVectorizer, TfidfStemmingVectorizer
from slub_docsa.models.classification.ann_torch import TorchSingleLayerDenseReluModel
from slub_docsa.models.classification.natlibfi_annif import AnnifModel
from slub_docsa.models.classification.scikit import ScikitClassifier


def get_classic_classification_models_map():
    """Return a map of classification model types and their generator functions."""
    return {
        "tfidf_10k_knn_k=1": lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1, metric="cosine"),
            vectorizer=TfidfVectorizer(max_features=10000),
        ),
        "tfidf_snowball_de_10k_knn_k=1": lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1, metric="cosine"),
            vectorizer=TfidfStemmingVectorizer(lang_code="de", max_features=10000),
        ),
        "tfidf_snowball_de_10k_knn_k=3": lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance"),
            vectorizer=TfidfStemmingVectorizer(lang_code="de", max_features=10000),
        ),
        "tfidf_snowball_de_10k_rforest": lambda: ScikitClassifier(
            predictor=RandomForestClassifier(n_jobs=-1, n_estimators=50, max_leaf_nodes=1000),
            vectorizer=TfidfStemmingVectorizer(lang_code="de", max_features=10000),
        ),
        "annif_tfidf_de": lambda: AnnifModel(model_type="tfidf", lang_code="de"),
        "annif_svc_de": lambda: AnnifModel(model_type="svc", lang_code="de"),
        "annif_fasttext_de": lambda: AnnifModel(model_type="fasttext", lang_code="de"),
        "annif_omikuji_de": lambda: AnnifModel(model_type="omikuji", lang_code="de"),
        "tfidf_snowball_de_10k_torch_ann": lambda: TorchSingleLayerDenseReluModel(
            epochs=50,
            vectorizer=TfidfStemmingVectorizer(lang_code="de", max_features=10000),
        )
    }
