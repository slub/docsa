"""Runs a single model for the k10plus dataset."""

# pylint: disable=invalid-name

from functools import partial
import logging
import os

from slub_docsa.common.paths import get_cache_dir, get_figures_dir
from slub_docsa.data.store.subject import cached_unique_subject_order
from slub_docsa.evaluation.classification.pipeline import score_classification_models_for_dataset_with_splits
from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets
from slub_docsa.experiments.common.scores import default_named_per_class_score_list, default_named_score_list
from slub_docsa.experiments.common.scores import initialize_named_score_tuple_list
from slub_docsa.evaluation.classification.split import scikit_kfold_splitter
from slub_docsa.experiments.k10plus.datasets import k10plus_named_sample_generators
from slub_docsa.serve.models.classification.common import get_all_classification_model_types

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    random_state = 123
    language = "de"
    variant = "public"
    limit = 100000
    min_samples = 50
    wordpiece_vocabulary_size = 40000
    dataset_name = f"k10plus_{variant}_{language}_rvk_ms={min_samples}_limit={limit}"
    model_name = "tiny_bert_torch_ann"
    # model_name = "tfidf_10k_knn_k=1"
    # model_name = "dbmdz_bert_sts1_torch_ann"

    plot_training_history_filepath = os.path.join(get_figures_dir(), "k10plus/classify_one_ann_history")
    stemming_cache_filepath = os.path.join(get_cache_dir(), "stemming/k10plus_cache.sqlite")

    logger.debug("load k10plus dataset from samples")
    named_dataset = next(filter_and_cache_named_datasets(
        k10plus_named_sample_generators(variants=[(variant, limit)], min_samples=min_samples),
        [dataset_name]
    ))

    logger.debug("load or generate subject order")
    subject_order = cached_unique_subject_order(dataset_name, named_dataset.dataset.subjects)
    logger.debug("there are %d unique subjects", len(subject_order))

    logger.debug("initialize model")
    subject_hierarchy = named_dataset.schema_generator()
    model_generator = partial(get_all_classification_model_types()[model_name], subject_hierarchy, subject_order)

    scores = initialize_named_score_tuple_list(default_named_score_list(
        # subject_order, subject_hierarchy_generator()
    ))
    per_class_scores = initialize_named_score_tuple_list(default_named_per_class_score_list())

    scores, per_class_scores = score_classification_models_for_dataset_with_splits(
        10, scikit_kfold_splitter(10, random_state=123),
        subject_order, named_dataset.dataset, [model_generator], scores.generators, per_class_scores.generators,
        stop_after_evaluating_split=0,
        use_test_data_as_validation_data=False,
        check_minimum_samples=False,
        check_split_distribution=False
    )

    logger.debug("print score")
    print(scores)
