"""Setup classification models based on dbmdz vectorization."""

import os
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from slub_docsa.common.paths import get_cache_dir, get_figures_dir
from slub_docsa.data.preprocess.vectorizer import HuggingfaceBertVectorizer, PersistedCachedVectorizer
from slub_docsa.models.classification.ann.pretrained import TorchBertSequenceClassificationHeadModel
from slub_docsa.models.classification.scikit import ScikitClassifier
from slub_docsa.serve.common import ModelTypeMapping


def _get_cached_dbmdz_bert_vectorizer(
        cache_dir: Optional[str] = None,
        subtext_samples: int = 1,
        hidden_states: int = 1,
):
    """Load persisted dbmdz bert vectorizer."""
    if cache_dir is None:
        cache_dir = os.path.join(get_cache_dir(), "vectorizer")

    os.makedirs(cache_dir, exist_ok=True)
    filename = f"dbmdz_sts={subtext_samples}_hs={hidden_states}.sqlite"
    dbmdz_bert_cache_fp = os.path.join(cache_dir, filename)
    return PersistedCachedVectorizer(dbmdz_bert_cache_fp, HuggingfaceBertVectorizer(subtext_samples=subtext_samples))


def get_dbmdz_classification_models_map() -> ModelTypeMapping:
    """Return a map of classification model types and their generator functions."""
    return {
        "dbmdz_bert_sts1_knn_k=1": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=_get_cached_dbmdz_bert_vectorizer(),
        ),
        "dbmdz_bert_sts1_rforest": lambda subject_hierarchy, subject_order: ScikitClassifier(
            predictor=RandomForestClassifier(
                n_jobs=-1, max_leaf_nodes=1000, n_estimators=100, max_samples=0.1
            ),
            vectorizer=_get_cached_dbmdz_bert_vectorizer(),
        ),
        "dbmdz_bert_sts1_torch_ann": lambda subject_hierarchy, subject_order: TorchBertSequenceClassificationHeadModel(
            max_training_t05_f1=0.85,
            max_training_time=600,
            batch_size=64,
            positive_class_weight=20.0,
            positive_class_weight_min=5.0,
            positive_class_weight_decay=0.8,
            vectorizer=_get_cached_dbmdz_bert_vectorizer(),
            preload_vectorizations=True,
            dataloader_workers=0,
            plot_training_history_filepath=os.path.join(
                get_figures_dir(), "ann_history/dbmdz_bert_sts1_torch_ann"
            )
        ),
    }
