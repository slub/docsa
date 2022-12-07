"""Runs a single model for a specific qucosa dataset variant."""

# pylint: disable=invalid-name

import logging
import os

from sklearn.metrics import f1_score
from slub_docsa.common.paths import get_cache_dir, get_figures_dir
from slub_docsa.evaluation.classification.score.scikit import scikit_metric_for_best_threshold_based_on_f1score

from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets
from slub_docsa.experiments.qucosa.datasets import qucosa_named_datasets_tuple_list
from slub_docsa.experiments.common.vectorizer import get_cached_tfidf_stemming_vectorizer
from slub_docsa.evaluation.classification.incidence import unique_subject_order, subject_incidence_matrix_from_targets
from slub_docsa.evaluation.classification.split import scikit_kfold_train_test_split
from slub_docsa.models.classification.ann.dense import TorchSingleLayerDenseReluModel


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    random_state = 123
    dataset_name = "qucosa_de_fulltexts_langid_ddc"
    check_qucosa_download = False

    plot_training_history_filepath = os.path.join(get_figures_dir(), "qucosa/classify_one_ann_history")
    stemming_cache_filepath = os.path.join(get_cache_dir(), "stemming/global_cache.sqlite")

    _, dataset, _ = next(filter_and_cache_named_datasets(
        qucosa_named_datasets_tuple_list(check_qucosa_download), [dataset_name]
    ))

    subject_order = unique_subject_order(dataset.subjects)

    train_dataset, test_dataset = scikit_kfold_train_test_split(0.9, dataset, random_state=random_state)
    train_incidence = subject_incidence_matrix_from_targets(train_dataset.subjects, subject_order)
    test_incidence = subject_incidence_matrix_from_targets(test_dataset.subjects, subject_order)

    # vectorizer = get_qucosa_dbmdz_bert_vectorizer(subtext_samples=1, hidden_states=1)
    vectorizer = get_cached_tfidf_stemming_vectorizer(max_features=10000)

    # model = AnnifModel(model_type="tfidf", lang_code="de")
    # model = TorchSingleLayerDenseTanhModel(
    # model = TorchBertSequenceClassificationHeadModel(
    model = TorchSingleLayerDenseReluModel(
        vectorizer=vectorizer,
        batch_size=16,
        epochs=50,
        lr=0.0001,
        plot_training_history_filepath=plot_training_history_filepath
    )

    model.fit(train_dataset.documents, train_incidence, test_dataset.documents, test_incidence)

    predicted_probabilities = model.predict_proba(test_dataset.documents)

    score = scikit_metric_for_best_threshold_based_on_f1score(
        f1_score, average="micro", zero_division=0
    )(test_incidence, predicted_probabilities)

    print(score)
