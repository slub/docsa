"""Various variants of artificial data prepared for experimentation."""

import logging
import os

from typing import Callable, Iterator, List, Optional, Tuple

from slub_docsa.common.dataset import Dataset, samples_from_dataset
from slub_docsa.common.paths import CACHE_DIR
from slub_docsa.common.subject import SubjectHierarchyType
from slub_docsa.data.artificial.hierarchical import generate_hierarchical_random_dataset_from_dbpedia
from slub_docsa.data.artificial.simple import generate_easy_random_dataset_from_dbpedia, generate_random_dataset
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_minimum_samples
from slub_docsa.data.store.dataset import load_persisted_dataset_from_lazy_sample_iterator
from slub_docsa.data.store.subject import load_persisted_subject_hierarchy_from_lazy_subject_generator

logger = logging.getLogger(__name__)

ARTIFICIAL_DATASETS_CACHE_DIR = os.path.join(CACHE_DIR, "artificial")


def generate_pruned_random_no_correlations_dataset(n_token, n_docs, n_subjects, min_samples):
    """Generate random data without any subject correlations, but with minimum samples per subject."""
    dataset = generate_random_dataset(n_token, n_docs, n_subjects)
    return filter_subjects_with_insufficient_samples(dataset, min_samples), None


def generate_pruned_easy_random_dataset_from_dbpedia(n_docs, n_subjects, min_samples):
    """Generate easy to predict random data that is pruned to have minimum samples per subject."""
    dataset = generate_easy_random_dataset_from_dbpedia("english", n_docs, n_subjects)
    return filter_subjects_with_insufficient_samples(dataset, min_samples), None


def generate_pruned_hierarchical_random_dataset(n_token, n_docs, n_subjects, min_samples):
    """Generate hierarchical random that that is hierarchical pruned to have minimum samples per subject."""
    dataset, subject_hierarchy = generate_hierarchical_random_dataset_from_dbpedia(
            "english", n_token, n_docs, n_subjects
    )
    dataset.subjects = prune_subject_targets_to_minimum_samples(min_samples, dataset.subjects, subject_hierarchy)
    dataset = filter_subjects_with_insufficient_samples(dataset, min_samples)
    return dataset, subject_hierarchy


def default_named_artificial_datasets(
    n_token: int,
    n_docs: int,
    n_subjects: int,
    min_samples: int,
    name_subset: List[str] = None,
) -> Iterator[Tuple[str, Dataset, Optional[SubjectHierarchyType]]]:
    """Return several persisted default artificial datasets."""
    lazy_named_datasets: List[Tuple[str, Callable[[], Tuple[Dataset, Optional[SubjectHierarchyType]]]]] = [
        (f"random_no_correlations_t={n_token}_d={n_docs}_s={n_subjects}_min={min_samples}", lambda:
            generate_pruned_random_no_correlations_dataset(n_token, n_docs, n_subjects, min_samples)),
        (f"random_easy_to_predict_dbpedia_d={n_docs}_s={n_subjects}_min={min_samples}", lambda:
            generate_pruned_easy_random_dataset_from_dbpedia(n_docs, n_subjects, min_samples)),
        (f"random_hierarchical_t={n_token}_d={n_docs}_s={n_subjects}_min={min_samples}", lambda:
            generate_pruned_hierarchical_random_dataset(n_token, n_docs, n_subjects, min_samples)),
    ]

    # filter data sets based on name subset parameter
    if name_subset is not None:
        lazy_named_datasets = list(filter(lambda i: i[0] in name_subset, lazy_named_datasets))

    for dataset_name, lazy_dataset_generator in lazy_named_datasets:
        # load and persist each dataset
        logger.info("load and save persisted random dataset %s", dataset_name)
        os.makedirs(ARTIFICIAL_DATASETS_CACHE_DIR, exist_ok=True)
        dataset_fp = os.path.join(ARTIFICIAL_DATASETS_CACHE_DIR, f"{dataset_name}_dataset.sqlite")
        subject_hierarchy_fp = os.path.join(ARTIFICIAL_DATASETS_CACHE_DIR, f"{dataset_name}_subject_hierarchy.dbm")

        # if dataset and subject hierarchy are stored, use stub definitions and load from files instead
        dataset = Dataset()
        subject_hierarchy = {} if os.path.exists(subject_hierarchy_fp) else None

        if not os.path.exists(dataset_fp):
            # dataset does not exist, generate it
            dataset, subject_hierarchy = lazy_dataset_generator()

        # store and load dataset
        dataset = load_persisted_dataset_from_lazy_sample_iterator(
            lambda d=dataset: samples_from_dataset(d),
            dataset_fp
        )

        # store and load subject hierarchy, if exists
        if subject_hierarchy is not None:
            subject_hierarchy = load_persisted_subject_hierarchy_from_lazy_subject_generator(
                lambda s=subject_hierarchy: s.values(),
                subject_hierarchy_fp
            )

        yield dataset_name, dataset, subject_hierarchy


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    for ds in default_named_artificial_datasets(1000, 10000, 10, 10):
        pass
