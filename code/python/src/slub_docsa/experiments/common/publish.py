"""Publish a model trained on artifical data."""

# pylint: disable=invalid-name,too-many-locals, too-many-arguments

import logging
import os

from sklearn.metrics import f1_score

import slub_docsa

from slub_docsa.data.load.subjects.common import default_schema_generators
from slub_docsa.evaluation.classification.incidence import subject_incidence_matrix_from_targets, unique_subject_order
from slub_docsa.evaluation.classification.score.scikit import scikit_metric_for_best_threshold_based_on_f1score
from slub_docsa.evaluation.classification.split import scikit_kfold_train_test_split
from slub_docsa.serve.common import PublishedClassificationModelStatistics, current_date_as_model_creation_date
from slub_docsa.serve.rest.service.models import classify_with_limit_and_threshold
from slub_docsa.data.store.model import PublishedClassificationModelInfo
from slub_docsa.data.store.model import default_published_classification_models_directory
from slub_docsa.data.store.model import load_published_classification_model
from slub_docsa.data.store.model import save_as_published_classification_model


logger = logging.getLogger(__name__)


def _evaluate_model(model, test_dataset, subject_order):
    predicted_probabilities = model.predict_proba(test_dataset.documents)
    test_incidence = subject_incidence_matrix_from_targets(test_dataset.subjects, subject_order)
    score = scikit_metric_for_best_threshold_based_on_f1score(
        f1_score, average="micro", zero_division=0
    )(test_incidence, predicted_probabilities)
    return score


def _default_model_directory(model_id):
    return os.path.join(default_published_classification_models_directory(), model_id)


def _default_model_id(dataset_name, model_type):
    """Return model id constructed from dataset name and model type."""
    return dataset_name + "__" + model_type


def _evaluate_published_model(model_id, model_types, test_dataset, subject_order):
    logger.info("load and evaluate persisted model")
    model_directory = _default_model_directory(model_id)
    schema_generators = default_schema_generators()
    published_model = load_published_classification_model(model_directory, model_types, schema_generators)
    classify_with_limit_and_threshold(
        published_model.model,
        test_dataset.documents,
        limit=3
    )
    persisted_f1_score = _evaluate_model(published_model.model, test_dataset, subject_order)
    logger.info(
        "score after persisting is %.5f for model '%s'", persisted_f1_score, model_id
    )


def publish_model(model_generator, model_type, dataset, dataset_name, subject_hierarchy_generator, random_state=123):
    """Publish a model."""
    model_id = _default_model_id(dataset_name, model_type)
    model_directory = _default_model_directory(model_id)
    if os.path.exists(model_directory):
        logger.info("skip existing persisted model '%s'", model_id)
        return

    logger.info("prepare training data")
    subject_order = unique_subject_order(dataset.subjects)
    train_dataset, test_dataset = scikit_kfold_train_test_split(0.9, dataset, random_state=random_state)
    train_incidence = subject_incidence_matrix_from_targets(train_dataset.subjects, subject_order)
    subject_hierarchy = subject_hierarchy_generator()

    logger.info("loading model '%s'", model_id)
    model = model_generator(subject_hierarchy, subject_order)
    logger.info("loaded model class %s", str(model))

    logger.info("train model %s", str(model))
    model.fit(train_dataset.documents, train_incidence)

    logger.info("evaluate model %s", str(model))
    test_dataset_f1_score = _evaluate_model(model, test_dataset, subject_order)
    logger.info("f1 score before persisting %f", test_dataset_f1_score)

    logger.info("save model with id '%s'", model_id)
    save_as_published_classification_model(
        directory=model_directory,
        model=model,
        subject_order=subject_order,
        model_info=PublishedClassificationModelInfo(
            model_id=model_id,
            model_type=model_type,
            model_version="v1",
            schema_id="rvk",
            creation_date=current_date_as_model_creation_date(),
            supported_languages=["de"],
            description=f"model trained for dataset variant '{dataset_name}' "
                      + f"with classifiation model '{model_type}'",
            tags=["qucosa", "only_titles"],
            slub_docsa_version=slub_docsa.__version__,
            statistics=PublishedClassificationModelStatistics(
                number_of_training_samples=len(train_dataset.subjects),
                number_of_test_samples=len(test_dataset.subjects),
                scores={
                    "f1_t=best": test_dataset_f1_score
                }
            )
        )
    )

    _evaluate_published_model(model_id, {model_type: model_generator}, test_dataset, subject_order)
