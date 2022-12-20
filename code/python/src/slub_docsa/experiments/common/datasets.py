"""Common methods when experimenting with datasets."""

import os
import logging

from typing import Callable, Iterator, NamedTuple, Optional, Sequence

from slub_docsa.common.sample import Sample
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.common.dataset import Dataset, dataset_from_samples, samples_from_dataset
from slub_docsa.common.paths import get_cache_dir
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level, prune_subject_targets_to_minimum_samples
from slub_docsa.data.store.dataset import load_persisted_dataset_from_lazy_sample_iterator

logger = logging.getLogger(__name__)


class NamedSamplesGenerator(NamedTuple):
    """Combines a sample generator and subject hierarchy with their names."""

    name: str
    samples_generator: Callable[[], Iterator[Sample]]
    schema_id: str
    schema_generator: Callable[[], SubjectHierarchy]
    languages: Sequence[str]


class NamedDataset(NamedTuple):
    """Combines a dataset and subject hierarchy with their names."""

    name: str
    dataset: Dataset
    schema_id: str
    schema_generator: Callable[[], SubjectHierarchy]
    languages: Sequence[str]


def filter_and_cache_named_datasets(
    named_sample_generators: Sequence[NamedSamplesGenerator],
    name_subset: Optional[Sequence[str]] = None,
) -> Iterator[NamedDataset]:
    """Filter and cache named datasets to an sqlite database.

    Parameters
    ----------
    dataset_list : Sequence[NamedDataset]
        the dataset list as list of tuples
    name_subset : Optional[Sequence[str]], optional
        optional name subset that is used to filter the full dataset list

    Yields
    ------
    NamedDataset
        tuples of dataset name, the dataset loaded as sqlite database, and a generator to load the corresponding
        subject hierarchy
    """
    cache_dir = os.path.join(get_cache_dir(), "datasets")

    # filter data sets based on name subset parameter
    if name_subset is not None:
        named_sample_generators = list(filter(lambda i: i.name in name_subset, named_sample_generators))

    for named_sample_generator in named_sample_generators:
        # load and persist each dataset
        dataset_name = named_sample_generator.name
        logger.debug("load and save persisted dataset %s", dataset_name)
        filepath = os.path.join(cache_dir, f"{dataset_name}.sqlite")
        dataset = load_persisted_dataset_from_lazy_sample_iterator(named_sample_generator.samples_generator, filepath)
        yield NamedDataset(
            dataset_name,
            dataset,
            named_sample_generator.schema_id,
            named_sample_generator.schema_generator,
            named_sample_generator.languages
        )


def filter_min_samples(samples_iterator, min_samples):
    """Apply standard minimum samples pruning to a sample iterator."""
    dataset = dataset_from_samples(samples_iterator)
    dataset = filter_subjects_with_insufficient_samples(dataset, min_samples)
    return samples_from_dataset(dataset)


def prune_by_level(samples_iterator, prune_level, min_samples, subject_hierarchy):
    """Apply level-based pruning to a sample iterator."""
    if prune_level < 1:
        raise ValueError("prune level must be at least 1")
    dataset = dataset_from_samples(samples_iterator)
    dataset.subjects = prune_subject_targets_to_level(prune_level, dataset.subjects, subject_hierarchy)
    pruned_iterator = samples_from_dataset(dataset)
    filtered_iterator = filter_min_samples(pruned_iterator, min_samples)
    return filtered_iterator


def prune_min_samples(samples_iterator, min_samples, subject_hierarchy):
    """Combine hierarchical and standard minimum sample pruning for a samples iterator."""
    # prune hierarchy
    dataset = dataset_from_samples(samples_iterator)
    dataset.subjects = prune_subject_targets_to_minimum_samples(min_samples, dataset.subjects, subject_hierarchy)
    pruned_iterator = samples_from_dataset(dataset)
    # filter min samples
    filtered_iterator = filter_min_samples(pruned_iterator, min_samples)
    return filtered_iterator
