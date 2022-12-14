"""Common methods for datasets and models related to qucosa."""

import argparse

from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets
from slub_docsa.experiments.qucosa.datasets import qucosa_named_sample_generators
from slub_docsa.experiments.qucosa.models import default_qucosa_named_clustering_models_tuple_list
from slub_docsa.experiments.qucosa.models import default_qucosa_named_classification_model_list
from slub_docsa.serve.models.classification.common import get_all_classification_model_types


def available_qucosa_classification_model_names():
    """Return all classificaation models for the qucosa dataset."""
    model_types = get_all_classification_model_types()
    return list(model_types.keys())


def available_qucosa_clustering_model_names():
    """Return all available clustering models for the qucosa dataset."""
    all_models = default_qucosa_named_clustering_models_tuple_list(1)
    return [n for n, _ in all_models]


def available_qucosa_dataset_names():
    """Return all dataset variants for qucosa."""
    return [named_sample_generator.name for named_sample_generator in qucosa_named_sample_generators()]


def load_qucosa_dataset_by_name(dataset_name: str, check_qucosa_download: bool):
    """Return a single dataset retrieving it by its name."""
    named_dataset = next(filter_and_cache_named_datasets(
        qucosa_named_sample_generators(check_qucosa_download), [dataset_name]
    ))
    if named_dataset is not None:
        return named_dataset.dataset, named_dataset.schema_generator()
    raise ValueError(f"dataset with name '{dataset_name}' not known")


def load_qucosa_classification_model_by_name(model_name: str):
    """Return a single model retrieving it by its name."""
    for name, model in default_qucosa_named_classification_model_list():
        if name == model_name:
            return model()
    raise ValueError(f"model with name '{model_name}' not known")


def add_common_qucosa_arguments(parser: argparse.ArgumentParser):
    """Add cli argument that allows to check whether qucosa data is fully downloaded."""
    parser.add_argument(
        "--check_qucosa_download",
        help="checks whether qucosa document count matches SLUB elasticsearch",
        action="store_true",
        default=False
    )
