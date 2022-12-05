import os
import logging

from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from slub_docsa.common.sample import Sample
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.common.dataset import Dataset, dataset_from_samples, samples_from_dataset
from slub_docsa.common.paths import get_cache_dir
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level, prune_subject_targets_to_minimum_samples
from slub_docsa.data.store.dataset import load_persisted_dataset_from_lazy_sample_iterator

logger = logging.getLogger(__name__)


DatasetTupleList = List[Tuple[str, Callable[[], Iterator[Sample]], Callable[[], SubjectHierarchy]]]


def filter_and_cache_named_datasets(
    dataset_list: DatasetTupleList,
    name_subset: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[str, Dataset, Callable[[], SubjectHierarchy]]]:
    """Return default qucosa dataset variants."""
    cache_dir = os.path.join(get_cache_dir(), "datasets")

    # filter data sets based on name subset parameter
    if name_subset is not None:
        dataset_list = list(filter(lambda i: i[0] in name_subset, dataset_list))

    for dataset_name, lazy_sample_iterator, lazy_subject_hierarchy in dataset_list:
        # load and persist each dataset
        logger.debug("load and save persisted dataset %s", dataset_name)
        filepath = os.path.join(cache_dir, f"{dataset_name}.sqlite")
        dataset = load_persisted_dataset_from_lazy_sample_iterator(lazy_sample_iterator, filepath)
        yield dataset_name, dataset, lazy_subject_hierarchy


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
