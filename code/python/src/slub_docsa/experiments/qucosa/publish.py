"""Publish a model trained on a qucosa dataset."""

# pylint: disable=invalid-name,too-many-locals

import logging

from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets
from slub_docsa.experiments.common.publish import publish_model
from slub_docsa.experiments.qucosa.datasets import qucosa_named_sample_generators
from slub_docsa.serve.models.classification.common import get_all_classification_model_types


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    check_qucosa_download = False
    dataset_name = "qucosa_de_titles_rvk"
    model_type = "something"

    model_generator = get_all_classification_model_types()[model_type]

    _, dataset, subject_hierarchy_genereator = next(filter_and_cache_named_datasets(
        qucosa_named_sample_generators(check_qucosa_download), [dataset_name]
    ))

    publish_model(model_generator, model_type, dataset, dataset_name, subject_hierarchy_genereator)
