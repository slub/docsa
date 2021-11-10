"""Runs a single model for a specfiic dataset."""

# pylint: disable=invalid-name

import logging

from sklearn.metrics import f1_score
from slub_docsa.evaluation.score import scikit_metric_for_best_threshold_based_on_f1score

from slub_docsa.experiments.qucosa.datasets import default_named_qucosa_datasets
from slub_docsa.evaluation.incidence import unique_subject_order, subject_incidence_matrix_from_targets
from slub_docsa.evaluation.split import scikit_kfold_train_test_split
from slub_docsa.experiments.common import get_qucosa_dbmdz_bert_vectorizer
from slub_docsa.models.ann_torch import TorchBertSequenceClassificationHeadModel

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    random_state = 123
    _, dataset, _ = list(default_named_qucosa_datasets(["qucosa_de_fulltexts_langid_rvk"]))[0]

    subject_order = unique_subject_order(dataset.subjects)

    train_dataset, test_dataset = scikit_kfold_train_test_split(0.9, dataset, random_state=random_state)
    train_incidence = subject_incidence_matrix_from_targets(train_dataset.subjects, subject_order)
    test_incidence = subject_incidence_matrix_from_targets(test_dataset.subjects, subject_order)

    dbmdz_vectorizer = get_qucosa_dbmdz_bert_vectorizer(1)

    model = TorchBertSequenceClassificationHeadModel(vectorizer=dbmdz_vectorizer, batch_size=16, epochs=10, lr=0.0001)
    # model = HuggingfaceSequenceClassificationModel("dbmdz/bert-base-german-uncased")
    model.fit(train_dataset.documents, train_incidence, test_dataset.documents, test_incidence)

    predicted_probabilities = model.predict_proba(test_dataset.documents)

    score = scikit_metric_for_best_threshold_based_on_f1score(
        f1_score, average="micro", zero_division=0
    )(test_incidence, predicted_probabilities)

    print(score)
