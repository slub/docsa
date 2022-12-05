"""Test tfidf annif model for qucosa data and compare results with CLI output."""


import logging
import os
from typing import cast

from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.sparse import csr_matrix

from slub_docsa.common.paths import get_annif_dir
from slub_docsa.data.load.subjects.rvk import load_rvk_subject_hierarchy_from_sqlite
from slub_docsa.data.load.tsv import save_dataset_as_annif_tsv, save_subject_labels_as_annif_tsv
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.data.preprocess.skos import subject_hierarchy_to_skos_graph, subject_labels_to_skos_graph
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_minimum_samples
from slub_docsa.evaluation.classification.incidence import subject_incidence_matrix_from_targets
from slub_docsa.evaluation.classification.incidence import PositiveTopkIncidenceDecision
from slub_docsa.evaluation.classification.incidence import unique_subject_order
from slub_docsa.evaluation.classification.score.common import absolute_confusion_from_incidence
from slub_docsa.evaluation.classification.split import scikit_kfold_train_test_split
from slub_docsa.models.classification.natlibfi_annif import AnnifModel
from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets
from slub_docsa.experiments.qucosa.datasets import qucosa_named_datasets_tuple_list

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    check_qucosa_download = False
    MIN_SAMPLES = 10
    LIMIT = 5
    MODEL_TYPE = "tfidf"
    LANG_CODE = "de"

    logger.info("load dataset")
    _, dataset, _ = next(filter_and_cache_named_datasets(
        qucosa_named_datasets_tuple_list(check_qucosa_download), ["qucosa_de_titles_rvk"]
    ))

    logger.info("load rvk subjects")
    rvk_hierarchy = load_rvk_subject_hierarchy_from_sqlite()

    logger.info("do pruning on rvk subjects")
    dataset.subjects = prune_subject_targets_to_minimum_samples(MIN_SAMPLES, dataset.subjects, rvk_hierarchy)
    dataset = filter_subjects_with_insufficient_samples(dataset, MIN_SAMPLES)

    logger.info("calculate relevant subject list from dataset")
    subject_order = unique_subject_order(dataset.subjects)
    rvk_labels = {uri: rvk_hierarchy.subject_labels(uri) for uri in subject_order}

    logger.info("create annif comparison_experiment directory")
    os.makedirs(os.path.join(get_annif_dir(), "comparison_experiment"), exist_ok=True)

    logger.info("save subject list as Annif TSV file")
    save_subject_labels_as_annif_tsv(
        rvk_labels,
        os.path.join(get_annif_dir(), "comparison_experiment/subjects.tsv"),
    )

    logger.info("save flat subject labels as skos turtle file")
    rvk_flat_graph = subject_labels_to_skos_graph(rvk_labels, LANG_CODE)
    with open(os.path.join(get_annif_dir(), "comparison_experiment/subjects.flat.ttl"), "wb") as f:
        f.write(cast(bytes, rvk_flat_graph.serialize(format="turtle")))

    logger.info("save subject hierarchy as skos turtle file")
    rvk_skos_graph = subject_hierarchy_to_skos_graph(
        subject_hierarchy=rvk_hierarchy,
        lang_code=LANG_CODE,
        mandatory_subject_list=subject_order
    )
    with open(os.path.join(get_annif_dir(), "comparison_experiment/subjects.hierarchical.ttl"), "wb") as f:
        f.write(cast(bytes, rvk_skos_graph.serialize(format="turtle")))

    # split data to fixed train and test set
    training_dataset, test_dataset = scikit_kfold_train_test_split(0.8, dataset, random_state=5)

    test_subject_order = unique_subject_order(test_dataset.subjects)
    logger.info(
        "test data only contains %d of in total %d unique subjects",
        len(test_subject_order),
        len(subject_order)
    )

    logger.info("save training data as Annif TSV file")
    save_dataset_as_annif_tsv(
        training_dataset,
        os.path.join(get_annif_dir(), "comparison_experiment/training_data.tsv"),
    )
    logger.info("save test data as Annif TSV file")
    save_dataset_as_annif_tsv(
        test_dataset,
        os.path.join(get_annif_dir(), "comparison_experiment/test_data.tsv"),
    )

    logger.info("fit Annif model with training data")
    model = AnnifModel(
        model_type=MODEL_TYPE,
        lang_code=LANG_CODE,
        subject_hierarchy=rvk_hierarchy,
        subject_order=subject_order
    )
    train_incidence_matrix = subject_incidence_matrix_from_targets(training_dataset.subjects, subject_order)
    model.fit(training_dataset.documents, train_incidence_matrix)

    logger.info("evaluate Annif model with test data")
    probabilties = model.predict_proba(test_dataset.documents)

    logger.info("score results")
    predicted_incidence_matrix = PositiveTopkIncidenceDecision(LIMIT)(probabilties)
    test_incidence_matrix = subject_incidence_matrix_from_targets(test_dataset.subjects, subject_order)

    # predicted_incidence_matrix = csr_matrix(predicted_incidence_matrix)
    predicted_incidence_matrix_binary = predicted_incidence_matrix > 0.0
    test_incidence_matrix_sparse = csr_matrix(test_incidence_matrix)

    # print results similar to Annif eval output
    score_variants = [
        ("doc avg", "samples", "\t\t"),
        ("subj avg", "macro", "\t\t"),
        ("weighted subj avg", "weighted", "\t"),
        ("microavg", "micro", "\t\t")
    ]

    for label, average, tabs in score_variants:
        print(
            f"Precision ({label}):", tabs,
            precision_score(
                test_incidence_matrix_sparse,
                predicted_incidence_matrix_binary,
                average=average,
                zero_division=0  # type: ignore
            )
        )
        print(
            f"Recall ({label}):", tabs,
            recall_score(
                test_incidence_matrix_sparse,
                predicted_incidence_matrix_binary,
                average=average,
                zero_division=0  # type: ignore
            )
        )
        print(
            f"F1 score ({label}):", tabs,
            f1_score(
                test_incidence_matrix_sparse,
                predicted_incidence_matrix_binary,
                average=average,
                zero_division=0  # type: ignore
            )
        )
    print("...")

    confusion = absolute_confusion_from_incidence(test_incidence_matrix, predicted_incidence_matrix)
    print("True positives:\t\t\t", confusion[0])
    print("False positives:\t\t", confusion[2])
    print("False negatives:\t\t", confusion[3])

    print("Documents evaluated:\t\t", len(test_dataset.documents))
    print("")
    print("Run the following commands for comparison:")
    print("rm -rf data/projects/qucosa-de")
    print("rm -rf data/vocabs/rvk-de")
    print("annif loadvoc qucosa-de data/comparison_experiment/subjects.tsv")
    print("annif loadvoc qucosa-de data/comparison_experiment/subjects.flat.ttl")
    print("annif loadvoc qucosa-de data/comparison_experiment/subjects.hierarchical.ttl")
    print("annif train qucosa-de data/comparison_experiment/training_data.tsv")
    print("annif eval --limit 5 qucosa-de data/comparison_experiment/test_data.tsv")
