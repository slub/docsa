"""Setup models based on artificial neural networks."""

import os

from functools import partial

from slub_docsa.common.paths import get_figures_dir
from slub_docsa.data.preprocess.vectorizer import TfidfStemmingVectorizer
from slub_docsa.experiments.common.vectorizer import get_static_wikipedia_wordpiece_vectorizer
from slub_docsa.models.classification.ann.bert import TorchBertModel
from slub_docsa.models.classification.ann.dense import TorchSingleLayerDenseReluModel
from slub_docsa.serve.common import ModelTypeMapping


def get_ann_classification_models_map() -> ModelTypeMapping:
    """Return a map of classification model types and their generator functions."""
    wikipedia_wordpiece_vectorizer = partial(
        get_static_wikipedia_wordpiece_vectorizer, lang_code="de", vocabulary_size=40000, max_length=48, uncased=True
    )

    return {
        "tfidf_snowball_de_10k_torch_ann": lambda subject_hierarchy, subject_order: TorchSingleLayerDenseReluModel(
            max_epochs=32,
            max_training_time=60,
            vectorizer=TfidfStemmingVectorizer(lang_code="de", max_features=10000),
            plot_training_history_filepath=os.path.join(
                get_figures_dir(), "ann_history/tfidf_snowball_de_10k_torch_ann"
            )
        ),
        "tiny_bert_torch_ann_de": lambda subject_hierarchy, subject_order: TorchBertModel(
            vectorizer=wikipedia_wordpiece_vectorizer(),
            batch_size=64,
            max_epochs=None,
            max_training_time=120,
            learning_rate=0.0001,
            learning_rate_decay=1.0,
            positive_class_weight=100.0,
            positive_class_weight_decay=0.95,
            bert_config={
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "hidden_dropout_prob": 0.1,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "attention_dropout_prob": 0.1,
                "classifier_dropout": 0.1,
            },
            plot_training_history_filepath=os.path.join(get_figures_dir(), "ann_history/tiny_bert_torch_ann")
        )
    }
