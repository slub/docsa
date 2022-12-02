"""Runs a single model for the k10plus dataset."""

# pylint: disable=invalid-name

from itertools import islice
import logging
import os

from slub_docsa.common.dataset import dataset_from_samples
from slub_docsa.common.paths import get_cache_dir, get_figures_dir
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.evaluation.classification.score.batched import BatchedF1Score, BatchedIncidenceDecisionConfusionScore

from slub_docsa.experiments.common.vectorizer import get_cached_tfidf_stemming_vectorizer
from slub_docsa.evaluation.classification.incidence import threshold_incidence_decision, unique_subject_order
from slub_docsa.evaluation.classification.incidence import subject_incidence_matrix_from_targets
from slub_docsa.evaluation.classification.split import scikit_kfold_train_test_split
# from slub_docsa.models.classification.ann_torch import TorchSingleLayerDenseReluModel
from slub_docsa.models.classification.natlibfi_annif import AnnifModel
# from slub_docsa.data.load.k10plus.slub import k10plus_slub_samples_generator
from slub_docsa.data.load.k10plus.samples import k10plus_public_samples_generator

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    random_state = 123
    language = "de"
    schema = "rvk"
    min_samples = 100
    limit = 500000

    plot_training_history_filepath = os.path.join(get_figures_dir(), "k10plus/classify_one_ann_history")
    stemming_cache_filepath = os.path.join(get_cache_dir(), "stemming/k10plus_cache.sqlite")

    logger.debug("load k10plus dataset from samples")
    dataset = dataset_from_samples(
        k10plus_public_samples_generator(languages=[language], schemas=[schema], limit=limit)
    )
    # dataset = dataset_from_samples(
    #     k10plus_slub_samples_generator(languages=[language], schemas=[schema], require_toc=True, limit=limit)
    # )
    logger.debug("filter dataset by subjects with insufficient samples")
    dataset = filter_subjects_with_insufficient_samples(dataset, min_samples)
    dataset_name = f"k10plus_public_{language}_{schema}"

    logger.debug("calculate unique subject order")
    subject_order = unique_subject_order(dataset.subjects)
    logger.debug("there are %d unique subjects", len(subject_order))

    logger.debug("split dataset into training and test set")
    train_dataset, test_dataset = scikit_kfold_train_test_split(0.9, dataset, random_state=random_state)

    logger.debug("calculate subject incidence matrices")
    train_incidence = subject_incidence_matrix_from_targets(train_dataset.subjects, subject_order)

    logger.debug("initialize vectorizer")
    # vectorizer = get_qucosa_dbmdz_bert_vectorizer(subtext_samples=1, hidden_states=1)
    vectorizer = get_cached_tfidf_stemming_vectorizer(cache_prefix=dataset_name, max_features=10000)

    logger.debug("initialize model")
    model = AnnifModel(model_type="omikuji", lang_code="de")
    # model = TorchSingleLayerDenseTanhModel(
    # model = TorchBertSequenceClassificationHeadModel(
    # model = TorchSingleLayerDenseReluModel(
    #     vectorizer=vectorizer,
    #     batch_size=16,
    #     epochs=50,
    #     lr=0.0001,
    #     plot_training_history_filepath=plot_training_history_filepath
    # )

    logger.debug("start training")
    model.fit(train_dataset.documents, train_incidence)

    logger.debug("predict test data")
    batched_score = BatchedIncidenceDecisionConfusionScore(
        incidence_decision=threshold_incidence_decision(0.5),
        confusion_score=BatchedF1Score()
    )
    test_document_generator = iter(test_dataset.documents)
    test_subjects_generator = iter(test_dataset.subjects)
    while True:
        logger.debug("evaluate chunk of test documents")
        test_document_chunk = list(islice(test_document_generator, 100))
        test_subjects_chunk = list(islice(test_subjects_generator, 100))
        if not test_document_chunk:
            break

        test_incidence_chunk = subject_incidence_matrix_from_targets(test_subjects_chunk, subject_order)
        predicted_probabilities_chunk = model.predict_proba(test_document_chunk)
        batched_score.add_batch(test_incidence_chunk, predicted_probabilities_chunk)

    logger.debug("calculate score")
    print(batched_score())
