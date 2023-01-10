"""Setup models based on artificial neural networks."""

import os

from slub_docsa.common.paths import get_figures_dir
from slub_docsa.data.preprocess.vectorizer import GensimTfidfVectorizer, StemmingVectorizer
from slub_docsa.experiments.common.vectorizer import get_static_wikipedia_wordpiece_vectorizer
from slub_docsa.models.classification.ann.bert import TorchBertModel
from slub_docsa.models.classification.ann.dense import TorchSingleLayerDenseReluModel
from slub_docsa.serve.common import ModelTypeMapping


def get_ann_classification_models_map() -> ModelTypeMapping:
    """Return a map of classification model types and their generator functions."""
    models = {}
    for lang_code in ["de", "en"]:
        models.update({
            f"tfidf_snowball_{lang_code}_10k_torch_ann": lambda subject_hierarchy, subject_order, lc=lang_code:
                TorchSingleLayerDenseReluModel(
                    batch_size=64,
                    max_training_t05_f1=0.85,
                    max_training_time=3600,  # 1 hour
                    positive_class_weight=20.0,
                    positive_class_weight_min=5.0,
                    positive_class_weight_decay=0.8,
                    vectorizer=StemmingVectorizer(
                        vectorizer=GensimTfidfVectorizer(max_features=10000),
                        lang_code=lc,
                    ),
                    preload_vectorizations=False,
                    dataloader_workers=8,
                    plot_training_history_filepath=os.path.join(
                        get_figures_dir(), f"ann_history/tfidf_snowball_{lc}_10k_torch_ann"
                    )
                ),
            f"tiny_bert_torch_ann_{lang_code}": lambda subject_hierarchy, subject_order, lc=lang_code:
                TorchBertModel(
                    vectorizer=get_static_wikipedia_wordpiece_vectorizer(
                        lang_code=lc, vocabulary_size=30000, max_length=64, uncased=True, limit=4000000
                    ),
                    batch_size=64,
                    max_epochs=None,
                    max_training_time=14400,  # 4 hours
                    max_training_t05_f1=0.85,
                    learning_rate=0.0001,
                    learning_rate_decay=1.0,
                    positive_class_weight=100.0,
                    positive_class_weight_min=5.0,
                    positive_class_weight_decay=0.9,
                    bert_config={
                        "hidden_size": 384,
                        "num_hidden_layers": 2,
                        "hidden_dropout_prob": 0.1,
                        "intermediate_size": 768,
                        "num_attention_heads": 4,
                        "attention_dropout_prob": 0.1,
                        "classifier_dropout": 0.1,
                    },
                    plot_training_history_filepath=os.path.join(get_figures_dir(), "ann_history/tiny_bert_torch_ann")
                )
        })
    return models
