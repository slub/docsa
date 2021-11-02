"""Runs a single model for a specfiic dataset."""

# pylint: disable=invalid-name

import logging

from sklearn.metrics import f1_score

from slub_docsa.experiments.qucosa.datasets import default_named_qucosa_datasets
from slub_docsa.evaluation.incidence import unique_subject_order, subject_incidence_matrix_from_targets
from slub_docsa.evaluation.incidence import positive_top_k_incidence_decision
from slub_docsa.evaluation.split import scikit_kfold_train_test_split
from slub_docsa.experiments.common import get_dbmdz_bert_vectorizer
from slub_docsa.models.ann_torch import TorchBertSequenceClassificationHeadModel

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    random_state = 123
    _, dataset, _ = list(default_named_qucosa_datasets(["qucosa_de_fulltexts_langid_rvk"]))[0]

    subject_order = unique_subject_order(dataset.subjects)

    train_dataset, test_dataset = scikit_kfold_train_test_split(0.98, dataset, random_state=random_state)
    train_incidence = subject_incidence_matrix_from_targets(train_dataset.subjects, subject_order)

    dbmdz_vectorizer = get_dbmdz_bert_vectorizer()

    model = TorchBertSequenceClassificationHeadModel(vectorizer=dbmdz_vectorizer, lr=0.001)
    # model = HuggingfaceSequenceClassificationModel("dbmdz/bert-base-german-uncased")
    model.fit(train_dataset.documents, train_incidence)

    predicted_probabilities = model.predict_proba(test_dataset.documents)

    predicted_incidence = positive_top_k_incidence_decision(3)(predicted_probabilities)
    test_incidence = subject_incidence_matrix_from_targets(test_dataset.subjects, subject_order)

    print(f1_score(
        test_incidence,
        predicted_incidence,
        average="micro",
        zero_division=0
    ))
