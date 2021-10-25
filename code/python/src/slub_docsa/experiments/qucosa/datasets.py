"""Provides various variants of the qucosa dataset."""

import logging
import os

from typing import Callable, Iterator, List, Tuple

from slub_docsa.common.dataset import Dataset, dataset_from_samples, samples_from_dataset
from slub_docsa.common.paths import CACHE_DIR
from slub_docsa.common.sample import SampleIterator
from slub_docsa.common.subject import SubjectHierarchyType
from slub_docsa.data.load.qucosa import read_qucosa_abstracts_rvk_samples, read_qucosa_fulltext_rvk_samples
from slub_docsa.data.load.qucosa import read_qucosa_documents_from_directory, read_qucosa_titles_rvk_samples
from slub_docsa.data.load.rvk import RvkSubjectNode, get_rvk_subject_store
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.data.preprocess.language import filter_samples_by_detected_fulltext_language_via_langid
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level, prune_subject_targets_to_minimum_samples
from slub_docsa.data.store.dataset import load_persisted_dataset_from_lazy_sample_iterator

logger = logging.getLogger(__name__)

QUCOSA_DATASET_CACHE_DIRECTORY = os.path.join(CACHE_DIR, "qucosa")


def prune_hierarchical_min_samples(samples_iterator, min_samples, subject_hierarchy):
    """Apply hierarchical minimum samples pruning to a sample iterator."""
    dataset = dataset_from_samples(samples_iterator)
    dataset.subjects = prune_subject_targets_to_minimum_samples(min_samples, dataset.subjects, subject_hierarchy)
    return samples_from_dataset(dataset)


def prune_min_samples(samples_iterator, min_samples):
    """Apply standard minimum samples pruning to a sample iterator."""
    dataset = dataset_from_samples(samples_iterator)
    dataset = filter_subjects_with_insufficient_samples(dataset, min_samples)
    return samples_from_dataset(dataset)


def prune_by_level(samples_iterator, prune_level, subject_hierarchy):
    """Apply level-based pruning to a sample iterator."""
    if prune_level < 1:
        raise ValueError("prune level must be at least 1")
    dataset = dataset_from_samples(samples_iterator)
    dataset.subjects = prune_subject_targets_to_level(prune_level, dataset.subjects, subject_hierarchy)
    return samples_from_dataset(dataset)


def default_pruning(samples_iterator, min_samples, subject_hierarchy):
    """Combine hierarchical and standard minimum sample pruning for a samples iterator."""
    pruned_iterator = prune_hierarchical_min_samples(samples_iterator, min_samples, subject_hierarchy)
    pruned_iterator = prune_min_samples(pruned_iterator, min_samples)
    return pruned_iterator


def load_qucosa_titles_samples(subject_hierarchy) -> SampleIterator:
    """Load qucosa documents only considering the title."""
    samples_iterator = read_qucosa_titles_rvk_samples(read_qucosa_documents_from_directory())
    return default_pruning(samples_iterator, 10, subject_hierarchy)


def load_qucosa_abstracts_samples(lang_code, subject_hierarchy) -> SampleIterator:
    """Load qucosa documents consisting of both the title and abstract."""
    samples_iterator = read_qucosa_abstracts_rvk_samples(read_qucosa_documents_from_directory(), lang_code)
    return default_pruning(samples_iterator, 10, subject_hierarchy)


def load_qucosa_fulltexts_samples(lang_code, subject_hierarchy) -> SampleIterator:
    """Load qucosa documents consiting both of the title and fulltext."""
    samples_iterator = read_qucosa_fulltext_rvk_samples(read_qucosa_documents_from_directory(), lang_code)
    filtered_iterator = filter_samples_by_detected_fulltext_language_via_langid(samples_iterator, lang_code)
    return default_pruning(filtered_iterator, 10, subject_hierarchy)


def default_named_qucosa_datasets(
    name_subset: List[str] = None
) -> Iterator[Tuple[str, Dataset, SubjectHierarchyType[RvkSubjectNode]]]:
    """Return default qucosa dataset variants."""
    rvk_hierarchy = get_rvk_subject_store()

    named_sample_iterators: List[Tuple[str, Callable[[], SampleIterator]]] = [
        ("qucosa_all_titles_rvk", lambda: load_qucosa_titles_samples(rvk_hierarchy)),
        ("qucosa_de_abstracts_unprocessed_rvk", lambda: load_qucosa_abstracts_samples("de", rvk_hierarchy)),
        ("qucosa_de_fulltexts_unprocessed_rvk", lambda: load_qucosa_fulltexts_samples("de", rvk_hierarchy)),
    ]

    # filter data sets based on name subset parameter
    if name_subset is not None:
        named_sample_iterators = list(filter(lambda i: i[0] in name_subset, named_sample_iterators))

    for dataset_name, lazy_sample_iterator in named_sample_iterators:
        # load and persist each dataset
        logger.info("load and save persisted dataset %s", dataset_name)
        filepath = os.path.join(QUCOSA_DATASET_CACHE_DIRECTORY, f"{dataset_name}.dbm")
        dataset = load_persisted_dataset_from_lazy_sample_iterator(lazy_sample_iterator, filepath)
        yield dataset_name, dataset, rvk_hierarchy


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # loads all data sets and generates persistent storage for them
    for ds in default_named_qucosa_datasets():
        pass
