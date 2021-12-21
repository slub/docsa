"""Common methods for datasets and models related to qucosa."""

import argparse

from slub_docsa.common.model import PersistableClassificationModel
from slub_docsa.experiments.annif.models import default_annif_named_model_list
from slub_docsa.experiments.dummy.models import default_dummy_named_model_list
from slub_docsa.experiments.qucosa.datasets import qucosa_named_datasets, qucosa_named_datasets_tuple_list
from slub_docsa.experiments.qucosa.models import default_qucosa_named_clustering_models_tuple_list
from slub_docsa.experiments.qucosa.models import default_qucosa_named_classification_model_list


def available_qucosa_classification_model_names(only_persistable: bool = False):
    """Return all classificaation models for the qucosa dataset."""
    if only_persistable:
        return [
            n for n, m in default_qucosa_named_classification_model_list()
            if isinstance(m(), PersistableClassificationModel)
        ]

    all_models = default_dummy_named_model_list() \
        + default_qucosa_named_classification_model_list() \
        + default_annif_named_model_list("de")
    return [n for n, _ in all_models]


def available_qucosa_clustering_model_names():
    """Return all available clustering models for the qucosa dataset."""
    all_models = default_qucosa_named_clustering_models_tuple_list(1)
    return [n for n, _ in all_models]


def available_qucosa_dataset_names():
    """Return all dataset variants for qucosa."""
    return [n for n, _, _ in qucosa_named_datasets_tuple_list()]


def load_qucosa_dataset_by_name(dataset_name: str, check_qucosa_download: bool):
    """Return a single dataset retrieving it by its name."""
    dataset_tuple = next(qucosa_named_datasets([dataset_name], check_qucosa_download))
    if dataset_tuple is not None:
        return dataset_tuple[1], dataset_tuple[2]
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
