"""Setup classification models based on dbmdz vectorization."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from slub_docsa.data.preprocess.vectorizer import CachedVectorizer, HuggingfaceBertVectorizer
from slub_docsa.models.classification.ann.pretrained import TorchBertSequenceClassificationHeadModel
from slub_docsa.models.classification.scikit import ScikitClassifier
from slub_docsa.serve.common import ModelTypeMapping


def get_dbmdz_classification_models_map() -> ModelTypeMapping:
    """Return a map of classification model types and their generator functions."""
    def dbmdz_bert_vectorizer():
        return CachedVectorizer(HuggingfaceBertVectorizer(subtext_samples=1))

    return {
        "dbmdz_bert_sts1_knn_k=1": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=dbmdz_bert_vectorizer(),
        ),
        "dbmdz_bert_sts1_rforest": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=RandomForestClassifier(
                n_jobs=-1, max_leaf_nodes=1000, n_estimators=100, max_samples=0.1
            ),
            vectorizer=dbmdz_bert_vectorizer(),
        ),
        "dbmdz_bert_sts1_torch_ann": lambda subject_hierarchy, subject_order: TorchBertSequenceClassificationHeadModel(
            epochs=50,
            vectorizer=dbmdz_bert_vectorizer(),
        ),
    }
