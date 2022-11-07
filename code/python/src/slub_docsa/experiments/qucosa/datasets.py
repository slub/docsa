"""Provides various variants of the qucosa dataset."""

# pylint: disable=too-many-arguments

import logging
import os

from typing import Callable, Iterator, List, Tuple, Union, Sequence, Optional
from typing_extensions import Literal

from slub_docsa.common.dataset import Dataset, dataset_from_samples, samples_from_dataset
from slub_docsa.common.paths import get_cache_dir
from slub_docsa.common.sample import Sample
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.load.qucosa import qucosa_subject_hierarchy_by_subject_schema, read_qucosa_samples
from slub_docsa.data.load.qucosa import read_qucosa_documents_from_directory
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.data.preprocess.language import filter_samples_by_detected_language_via_langid
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level, prune_subject_targets_to_minimum_samples
from slub_docsa.data.store.dataset import load_persisted_dataset_from_lazy_sample_iterator
from slub_docsa.evaluation.incidence import unique_subject_order

logger = logging.getLogger(__name__)


def _filter_min_samples(samples_iterator, min_samples):
    """Apply standard minimum samples pruning to a sample iterator."""
    dataset = dataset_from_samples(samples_iterator)
    dataset = filter_subjects_with_insufficient_samples(dataset, min_samples)
    return samples_from_dataset(dataset)


def _prune_by_level(samples_iterator, prune_level, min_samples, subject_hierarchy):
    """Apply level-based pruning to a sample iterator."""
    if prune_level < 1:
        raise ValueError("prune level must be at least 1")
    dataset = dataset_from_samples(samples_iterator)
    dataset.subjects = prune_subject_targets_to_level(prune_level, dataset.subjects, subject_hierarchy)
    pruned_iterator = samples_from_dataset(dataset)
    filtered_iterator = _filter_min_samples(pruned_iterator, min_samples)
    return filtered_iterator


def _prune_min_samples(samples_iterator, min_samples, subject_hierarchy):
    """Combine hierarchical and standard minimum sample pruning for a samples iterator."""
    # prune hierarchy
    dataset = dataset_from_samples(samples_iterator)
    dataset.subjects = prune_subject_targets_to_minimum_samples(min_samples, dataset.subjects, subject_hierarchy)
    pruned_iterator = samples_from_dataset(dataset)
    # filter min samples
    filtered_iterator = _filter_min_samples(pruned_iterator, min_samples)
    return filtered_iterator


def _load_qucosa_samples(
    subject_schema: Union[Literal["rvk"], Literal["ddc"]],
    text_source: str = "titles",
    lang_code: Optional[str] = None,
    langid_check: bool = False,
    pruning_method: str = "min_samples_10",
    check_qucosa_download: bool = False,
) -> Iterator[Sample]:
    subject_hierarchy = qucosa_subject_hierarchy_by_subject_schema(subject_schema)
    qucosa_iterator = read_qucosa_documents_from_directory(
        check_elasticsearch_document_count=check_qucosa_download,
    )
    sample_iterator = read_qucosa_samples(qucosa_iterator, text_source, subject_schema, lang_code)

    if langid_check:
        if lang_code is None:
            raise ValueError("lang code can not be None when lang check is requested")
        sample_iterator = filter_samples_by_detected_language_via_langid(sample_iterator, lang_code)

    if pruning_method == "no_pruning":
        return sample_iterator
    if pruning_method == "filter_samples_10":
        return _filter_min_samples(sample_iterator, 10)
    if pruning_method == "min_samples_10":
        return _prune_min_samples(sample_iterator, 10, subject_hierarchy)
    if pruning_method == "level_1":
        return _prune_by_level(sample_iterator, 1, 10, subject_hierarchy)
    if pruning_method == "level_2":
        return _prune_by_level(sample_iterator, 2, 10, subject_hierarchy)
    raise RuntimeError("unknown pruning method")


def qucosa_named_datasets_tuple_list(
    check_qucosa_download: bool = False,
):
    """Return list of qucosa datasets as tuples."""
    def lazy_rvk():
        return qucosa_subject_hierarchy_by_subject_schema("rvk")

    def lazy_ddc():
        return qucosa_subject_hierarchy_by_subject_schema("ddc")

    cqd = check_qucosa_download

    datasets: List[Tuple[str, Callable[[], Iterator[Sample]], Callable[[], SubjectHierarchy]]] = [
        ("qucosa_all_titles_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "titles", None, False, "min_samples_10", cqd
            ), lazy_rvk),
        ("qucosa_all_titles_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "titles", None, False, "min_samples_10", cqd
            ), lazy_ddc),
        ("qucosa_de_titles_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "titles", "de", False, "min_samples_10", cqd
            ), lazy_rvk),
        ("qucosa_de_titles_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "titles", "de", False, "min_samples_10", cqd
            ), lazy_ddc),
        ("qucosa_de_titles_langid_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "titles", "de", True, "min_samples_10", cqd
            ), lazy_rvk),
        ("qucosa_de_titles_langid_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "titles", "de", True, "min_samples_10", cqd
            ), lazy_ddc),
        ("qucosa_de_complete_but_only_titles_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "complete_but_only_titles", "de", False, "min_samples_10", cqd
            ), lazy_rvk),
        ("qucosa_de_abstracts_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "abstracts", "de", False, "min_samples_10", cqd
            ), lazy_rvk),
        ("qucosa_de_abstracts_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "abstracts", "de", False, "min_samples_10", cqd
            ), lazy_ddc),
        ("qucosa_de_abstracts_langid_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "abstracts", "de", True, "min_samples_10", cqd
            ), lazy_rvk),
        ("qucosa_de_abstracts_langid_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "abstracts", "de", True, "min_samples_10", cqd
            ), lazy_ddc),
        ("qucosa_de_complete_but_only_abstracts_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "complete_but_only_abstracts", "de", False, "min_samples_10", cqd
            ), lazy_rvk),
        ("qucosa_de_fulltexts_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "fulltexts", "de", False, "min_samples_10", cqd
            ), lazy_rvk),
        ("qucosa_de_fulltexts_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "fulltexts", "de", False, "min_samples_10", cqd
            ), lazy_ddc),
        ("qucosa_de_fulltexts_langid_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "fulltexts", "de", True, "min_samples_10", cqd
            ), lazy_rvk),
        ("qucosa_de_fulltexts_langid_rvk_level_1",
            lambda: _load_qucosa_samples(
                "rvk", "fulltexts", "de", True, "level_1", cqd
            ), lazy_rvk),
        ("qucosa_de_fulltexts_langid_rvk_level_2",
            lambda: _load_qucosa_samples(
                "rvk", "fulltexts", "de", True, "level_2", cqd
            ), lazy_rvk),
        ("qucosa_de_fulltexts_langid_rvk_no_pruning",
            lambda: _load_qucosa_samples(
                "rvk", "fulltexts", "de", True, "no_pruning", cqd
            ), lazy_rvk),
        ("qucosa_de_fulltexts_langid_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "fulltexts", "de", True, "min_samples_10", cqd
            ), lazy_ddc),
        ("qucosa_de_complete_but_only_fulltexts_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "complete_but_only_fulltexts", "de", False, "min_samples_10", cqd
            ), lazy_rvk),
    ]
    return datasets


def qucosa_named_datasets(
    name_subset: Optional[Sequence[str]] = None,
    check_qucosa_download: bool = False,
) -> Iterator[Tuple[str, Dataset, SubjectHierarchy]]:
    """Return default qucosa dataset variants."""
    quocsa_cache_dir = os.path.join(get_cache_dir(), "qucosa")
    dataset_list = qucosa_named_datasets_tuple_list(check_qucosa_download)

    # filter data sets based on name subset parameter
    if name_subset is not None:
        dataset_list = list(filter(lambda i: i[0] in name_subset, dataset_list))

    for dataset_name, lazy_sample_iterator, lazy_subject_hierarchy in dataset_list:
        # load and persist each dataset
        logger.info("load and save persisted dataset %s", dataset_name)
        filepath = os.path.join(quocsa_cache_dir, f"{dataset_name}.sqlite")
        dataset = load_persisted_dataset_from_lazy_sample_iterator(lazy_sample_iterator, filepath)
        yield dataset_name, dataset, lazy_subject_hierarchy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # loads all data sets and generates persistent storage for them
    for dn, ds, _ in qucosa_named_datasets():
        n_unique_subjects = len(unique_subject_order(ds.subjects))
        logger.info(
            "dataset %s has %d documents and %d unique subjects",
            dn, len(ds.documents), n_unique_subjects
        )
