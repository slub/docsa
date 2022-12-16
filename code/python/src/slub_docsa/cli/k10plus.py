"""Common methods for k10plus data."""

from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets
from slub_docsa.experiments.k10plus.datasets import k10plus_named_sample_generators


def available_k10plus_dataset_names():
    """Return dataset variants available for k10plus or qucosa."""
    return [named_dataset.name for named_dataset in k10plus_named_sample_generators()]


def load_k10plus_dataset_by_name(dataset_name: str):
    """Return a single dataset retrieving it by its name."""
    named_dataset = next(filter_and_cache_named_datasets(
        k10plus_named_sample_generators(), [dataset_name]
    ))
    if named_dataset is not None:
        return named_dataset.dataset, named_dataset.schema_generator()
    raise ValueError(f"dataset with name '{dataset_name}' not known")
