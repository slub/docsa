"""Runs a single model for the k10plus dataset."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.paths import get_cache_dir, get_figures_dir
from slub_docsa.data.preprocess.vectorizer import WordpieceVectorizer
from slub_docsa.evaluation.classification.pipeline import score_classification_models_for_dataset_with_splits
from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets
from slub_docsa.experiments.common.scores import default_named_per_class_score_list, default_named_score_list
from slub_docsa.experiments.common.scores import initialize_named_score_tuple_list
from slub_docsa.evaluation.classification.incidence import unique_subject_order
from slub_docsa.evaluation.classification.split import scikit_kfold_splitter, scikit_kfold_train_test_split
from slub_docsa.experiments.k10plus.datasets import k10plus_named_datasets_tuple_list
from slub_docsa.models.classification.ann.bert import TorchBertModel
from slub_docsa.experiments.common.vectorizer import get_cached_tfidf_stemming_vectorizer
# from slub_docsa.models.classification.ann.base import TorchSingleLayerDenseReluModel
# from slub_docsa.models.classification.natlibfi_annif import AnnifModel


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    random_state = 123
    dataset_name = "k10plus_public_de_rvk_ms=50_limit=10000"

    plot_training_history_filepath = os.path.join(get_figures_dir(), "k10plus/classify_one_ann_history")
    stemming_cache_filepath = os.path.join(get_cache_dir(), "stemming/k10plus_cache.sqlite")

    logger.debug("load k10plus dataset from samples")
    _, dataset, subject_hierarchy_generator = next(filter_and_cache_named_datasets(
        k10plus_named_datasets_tuple_list(),
        [dataset_name]
    ))

    logger.debug("calculate unique subject order")
    subject_order = unique_subject_order(dataset.subjects)
    logger.debug("there are %d unique subjects", len(subject_order))

    logger.debug("split dataset into training and test set")
    train_dataset, test_dataset = scikit_kfold_train_test_split(0.9, dataset, random_state=random_state)

    logger.debug("initialize vectorizer")
    # vectorizer = get_qucosa_dbmdz_bert_vectorizer(subtext_samples=1, hidden_states=1)
    vectorizer = get_cached_tfidf_stemming_vectorizer(cache_prefix=dataset_name, max_features=10000)
    vectorizer = WordpieceVectorizer("de", vocabulary_size=10000, max_length=32, uncased=True)

    logger.debug("initialize model")

    def _model_generator():
        # return AnnifModel(model_type="omikuji", lang_code="de")
        # model = TorchSingleLayerDenseTanhModel(
        # model = TorchBertSequenceClassificationHeadModel(
        # return TorchSingleLayerDenseReluModel(
        return TorchBertModel(
            vectorizer=vectorizer,
            batch_size=64,
            epochs=128,
            lr=0.0001,
            plot_training_history_filepath=plot_training_history_filepath
        )

    scores = initialize_named_score_tuple_list(default_named_score_list(
        # subject_order, subject_hierarchy_generator()
    ))
    per_class_scores = initialize_named_score_tuple_list(default_named_per_class_score_list())

    scores, per_class_scores = score_classification_models_for_dataset_with_splits(
        10, scikit_kfold_splitter(10, random_state=123),
        subject_order, dataset, [_model_generator], scores.generators, per_class_scores.generators,
        stop_after_evaluating_split=0,
        use_test_data_as_validation_data=True
    )

    logger.debug("print score")
    print(scores)
