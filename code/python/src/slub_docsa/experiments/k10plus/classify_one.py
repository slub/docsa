"""Runs a single model for the k10plus dataset."""

# pylint: disable=invalid-name

import logging
import os

from slub_docsa.common.dataset import dataset_from_samples
from slub_docsa.common.paths import get_cache_dir, get_figures_dir
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.evaluation.classification.pipeline import score_classification_models_for_dataset_with_splits
from slub_docsa.evaluation.classification.score.batched import BatchedBestThresholdScore, BatchedF1Score
from slub_docsa.evaluation.classification.score.batched import BatchedIncidenceDecisionConfusionScore
from slub_docsa.evaluation.classification.score.batched import BatchedPrecisionScore, BatchedRecallScore
from slub_docsa.experiments.common.vectorizer import get_cached_tfidf_stemming_vectorizer
from slub_docsa.evaluation.classification.incidence import threshold_incidence_decision, unique_subject_order
from slub_docsa.evaluation.classification.split import scikit_kfold_splitter, scikit_kfold_train_test_split
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
    min_samples = 10
    limit = 100000

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

    logger.debug("initialize vectorizer")
    # vectorizer = get_qucosa_dbmdz_bert_vectorizer(subtext_samples=1, hidden_states=1)
    vectorizer = get_cached_tfidf_stemming_vectorizer(cache_prefix=dataset_name, max_features=10000)

    logger.debug("initialize model")

    def model_generator():
        return AnnifModel(model_type="omikuji", lang_code="de")
    # model = TorchSingleLayerDenseTanhModel(
    # model = TorchBertSequenceClassificationHeadModel(
    # model = TorchSingleLayerDenseReluModel(
    #     vectorizer=vectorizer,
    #     batch_size=16,
    #     epochs=50,
    #     lr=0.0001,
    #     plot_training_history_filepath=plot_training_history_filepath
    # )

    batched_scores = [
        BatchedIncidenceDecisionConfusionScore(
            incidence_decision=threshold_incidence_decision(0.5),
            confusion_score=BatchedF1Score()
        ),
        BatchedIncidenceDecisionConfusionScore(
            incidence_decision=threshold_incidence_decision(0.5),
            confusion_score=BatchedPrecisionScore()
        ),
        BatchedIncidenceDecisionConfusionScore(
            incidence_decision=threshold_incidence_decision(0.5),
            confusion_score=BatchedRecallScore()
        ),
        BatchedBestThresholdScore(
            score_generator=BatchedF1Score
        ),
        BatchedBestThresholdScore(
            score_generator=BatchedPrecisionScore
        ),
        BatchedBestThresholdScore(
            score_generator=BatchedRecallScore
        )
    ]
    scores, per_class_scores = score_classification_models_for_dataset_with_splits(
        10, scikit_kfold_splitter(10, random_state=123),
        subject_order, dataset, [model_generator], batched_scores, [],
        stop_after_evaluating_split=1
    )

    logger.debug("print score")
    print(scores, per_class_scores)
