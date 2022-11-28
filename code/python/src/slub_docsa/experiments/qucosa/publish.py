"""Publish a model trained on artifical data."""

# pylint: disable=invalid-name

import logging
import os

from sklearn.metrics import f1_score

import slub_docsa

from slub_docsa.common.paths import get_serve_dir
from slub_docsa.evaluation.classification.incidence import subject_incidence_matrix_from_targets, unique_subject_order
from slub_docsa.evaluation.classification.score import scikit_metric_for_best_threshold_based_on_f1score
from slub_docsa.evaluation.classification.split import scikit_kfold_train_test_split
from slub_docsa.experiments.qucosa.datasets import qucosa_named_datasets
from slub_docsa.serve.common import current_date_as_model_creation_date
from slub_docsa.serve.models.classification.classic import get_classic_classification_models_map
from slub_docsa.serve.rest.service.models import classify_with_limit_and_threshold
from slub_docsa.serve.store.models import PublishedClassificationModelInfo, load_published_classification_model
from slub_docsa.serve.store.models import save_as_published_classification_model


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    random_state = 123
    split_function_name = "random"
    n_splits = 10
    min_samples = 10
    dataset_name = "qucosa_de_titles_rvk"

    model_type = "tfidf_snowball_de_10k_rforest"
    model_id = dataset_name + "__" + model_type

    named_datasets = qucosa_named_datasets([dataset_name])
    _, dataset, subject_hierarchy = next(named_datasets)

    model = get_classic_classification_models_map()[model_type]()
    logger.info("loaded model %s", str(model))

    subject_order = unique_subject_order(dataset.subjects)

    train_dataset, test_dataset = scikit_kfold_train_test_split(0.9, dataset, random_state=random_state)
    train_incidence = subject_incidence_matrix_from_targets(train_dataset.subjects, subject_order)
    test_incidence = subject_incidence_matrix_from_targets(test_dataset.subjects, subject_order)

    model.fit(train_dataset.documents, train_incidence, test_dataset.documents, test_incidence)

    def _evaluate_model(m):
        predicted_probabilities = m.predict_proba(test_dataset.documents)

        score = scikit_metric_for_best_threshold_based_on_f1score(
            f1_score, average="micro", zero_division=0
        )(test_incidence, predicted_probabilities)

        return score

    logger.info("score before persisting %f", _evaluate_model(model))

    logger.info("save model")
    models_directory = os.path.join(get_serve_dir(), "classification_models")
    model_directory = os.path.join(models_directory, model_id)
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
            description=f"""qucosa model trained for dataset variant '{dataset_name}'
                with classifiation model '{model_type}'""",
            tags=["qucosa", "only_titles"],
            slub_docsa_version=slub_docsa.__version__,
        )
    )

    logger.info("load model")
    published_model = load_published_classification_model(model_directory, get_classic_classification_models_map())
    results = classify_with_limit_and_threshold(
        published_model.model,
        test_dataset.documents,
        limit=3
    )
    logger.info("score after persisting %f", _evaluate_model(published_model.model))
