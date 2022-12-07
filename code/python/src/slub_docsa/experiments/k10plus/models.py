"""Common models that are evaluated when experimenting with qucosa and artifical datasets."""

# pylint: disable=unnecessary-lambda

from functools import partial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from slub_docsa.models.classification.ann.dense import TorchSingleLayerDenseReluModel
from slub_docsa.models.classification.ann.pretrained import TorchBertSequenceClassificationHeadModel
from slub_docsa.models.classification.scikit import ScikitClassifier
from slub_docsa.models.classification.ann.bert import TorchBertModel

from slub_docsa.experiments.common.vectorizer import get_static_wikipedia_wordpiece_vectorizer
from slub_docsa.experiments.common.vectorizer import get_cached_tfidf_stemming_vectorizer
from slub_docsa.experiments.common.vectorizer import get_cached_dbmdz_bert_vectorizer
from slub_docsa.experiments.common.models import NamedClassificationModelTupleList


def default_k10plus_named_classification_model_list(language) -> NamedClassificationModelTupleList:
    """Return a list of default qucosa models to use for evaluating model performance."""
    tfidf_stemming_vectorizer = partial(
        get_cached_tfidf_stemming_vectorizer, lang_code=language, max_features=10000, cache_prefix="k10plus"
    )
    dbmdz_bert_vectorizer = partial(get_cached_dbmdz_bert_vectorizer, subtext_samples=1)
    wikipedia_wordpiece_vectorizer = partial(
        get_static_wikipedia_wordpiece_vectorizer, lang_code="de", vocabulary_size=40000, max_length=48, uncased=True
    )

    models = [
        ("tfidf_10k_knn_k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=tfidf_stemming_vectorizer()
        )),
        ("dbmdz_bert_sts1_knn_k=1", lambda: ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=dbmdz_bert_vectorizer(),
        )),
        ("dbmdz_bert_sts1_rforest", lambda: ScikitClassifier(
            predictor=RandomForestClassifier(
                n_jobs=-1, max_leaf_nodes=1000, n_estimators=100, max_samples=0.1
            ),
            vectorizer=dbmdz_bert_vectorizer(),
        )),
        ("tfidf_10k_rforest", lambda: ScikitClassifier(
            predictor=RandomForestClassifier(
                n_jobs=-1, max_leaf_nodes=1000, n_estimators=100, max_samples=0.1
            ),
            vectorizer=tfidf_stemming_vectorizer(),
        )),
        ("tfidf_10k_torch_ann", lambda: TorchSingleLayerDenseReluModel(
            epochs=50,
            vectorizer=tfidf_stemming_vectorizer(),
        )),
        ("dbmdz_bert_sts1_torch_ann", lambda: TorchBertSequenceClassificationHeadModel(
            epochs=50,
            vectorizer=dbmdz_bert_vectorizer(),
        )),
        ("tiny_bert_torch_ann", lambda: TorchBertModel(
            vectorizer=wikipedia_wordpiece_vectorizer(),
            batch_size=64,
            epochs=64,
            lr=0.0001,
            positive_class_weight=200.0,
            positive_class_weight_decay=0.95,
            bert_config={
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "hidden_dropout_prob": 0.1,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "attention_dropout_prob": 0.1,
                "classifier_dropout": 0.1,
            }
        ))
    ]

    return models


if __name__ == "__main__":

    for model_name, model_generator in default_k10plus_named_classification_model_list(language="de"):
        print(model_name, str(model_generator()))
