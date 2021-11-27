"""Provides various variants of the qucosa dataset."""

import logging
import os

from typing import Callable, Iterator, List, Tuple, Union
from typing_extensions import Literal

from slub_docsa.common.dataset import Dataset, dataset_from_samples, samples_from_dataset
from slub_docsa.common.paths import CACHE_DIR
from slub_docsa.common.sample import Sample
from slub_docsa.common.subject import SubjectHierarchyType
from slub_docsa.data.load.qucosa import read_qucosa_abstracts_rvk_samples, read_qucosa_fulltext_rvk_samples
from slub_docsa.data.load.qucosa import read_qucosa_documents_from_directory, read_qucosa_titles_rvk_samples
from slub_docsa.data.load.rvk import RvkSubjectNode, get_rvk_subject_store
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.data.preprocess.document import apply_nltk_snowball_stemming_to_document_samples_iterator
from slub_docsa.data.preprocess.language import filter_samples_by_detected_language_via_langid
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_level, prune_subject_targets_to_minimum_samples
from slub_docsa.data.preprocess.vectorizer import TfidfStemmingVectorizer
from slub_docsa.data.preprocess.vectorizer import CachedVectorizer
from slub_docsa.data.store.dataset import load_persisted_dataset_from_lazy_sample_iterator
from slub_docsa.evaluation.incidence import unique_subject_order

logger = logging.getLogger(__name__)

QUCOSA_DATASET_CACHE_DIRECTORY = os.path.join(CACHE_DIR, "qucosa")
VECTORIZATION_CACHE = os.path.join(CACHE_DIR, "vectorizer")


def _prune_hierarchical_min_samples(samples_iterator, min_samples, subject_hierarchy):
    """Apply hierarchical minimum samples pruning to a sample iterator."""
    dataset = dataset_from_samples(samples_iterator)
    dataset.subjects = prune_subject_targets_to_minimum_samples(min_samples, dataset.subjects, subject_hierarchy)
    return samples_from_dataset(dataset)


def _prune_min_samples(samples_iterator, min_samples):
    """Apply standard minimum samples pruning to a sample iterator."""
    dataset = dataset_from_samples(samples_iterator)
    dataset = filter_subjects_with_insufficient_samples(dataset, min_samples)
    return samples_from_dataset(dataset)


def _prune_by_level(samples_iterator, prune_level, subject_hierarchy):
    """Apply level-based pruning to a sample iterator."""
    if prune_level < 1:
        raise ValueError("prune level must be at least 1")
    dataset = dataset_from_samples(samples_iterator)
    dataset.subjects = prune_subject_targets_to_level(prune_level, dataset.subjects, subject_hierarchy)
    return samples_from_dataset(dataset)


def _default_pruning(samples_iterator, min_samples, subject_hierarchy):
    """Combine hierarchical and standard minimum sample pruning for a samples iterator."""
    pruned_iterator = _prune_hierarchical_min_samples(samples_iterator, min_samples, subject_hierarchy)
    pruned_iterator = _prune_min_samples(pruned_iterator, min_samples)
    return pruned_iterator


def _load_qucosa_samples(
    subject_hierarchy,
    text_source: Union[Literal["titles"], Literal["abstracts"], Literal["fulltexts"]] = "titles",
    lang_code: str = None,
    langid_check: bool = False,
    stemming: bool = False,
) -> Iterator[Sample]:
    qucosa_iterator = read_qucosa_documents_from_directory()
    sample_iterator = read_qucosa_titles_rvk_samples(qucosa_iterator, lang_code)
    if text_source == "abstracts":
        sample_iterator = read_qucosa_abstracts_rvk_samples(qucosa_iterator, lang_code)
    if text_source == "fulltexts":
        sample_iterator = read_qucosa_fulltext_rvk_samples(qucosa_iterator, lang_code)

    if langid_check:
        if lang_code is None:
            raise ValueError("lang code can not be None when lang check is requested")
        sample_iterator = filter_samples_by_detected_language_via_langid(sample_iterator, lang_code)

    if stemming:
        if lang_code is None:
            raise ValueError("lang code can not be None when stemming is requested")
        sample_iterator = apply_nltk_snowball_stemming_to_document_samples_iterator(sample_iterator, lang_code)

    return _default_pruning(sample_iterator, 10, subject_hierarchy)


def default_named_qucosa_datasets(
    name_subset: List[str] = None
) -> Iterator[Tuple[str, Dataset, SubjectHierarchyType[RvkSubjectNode]]]:
    """Return default qucosa dataset variants."""
    rvk = get_rvk_subject_store()

    named_sample_iterators: List[Tuple[str, Callable[[], Iterator[Sample]]]] = [
        ("qucosa_all_titles_rvk", lambda: _load_qucosa_samples(rvk, "titles", None, False, False)),
        ("qucosa_de_titles_rvk", lambda: _load_qucosa_samples(rvk, "titles", "de", False, False)),
        ("qucosa_de_titles_langid_rvk", lambda: _load_qucosa_samples(rvk, "titles", "de", True, False)),
        ("qucosa_de_abstracts_rvk", lambda: _load_qucosa_samples(rvk, "abstracts", "de", False, False)),
        ("qucosa_de_abstracts_langid_rvk", lambda: _load_qucosa_samples(rvk, "abstracts", "de", True, False)),
        ("qucosa_de_fulltexts_rvk", lambda: _load_qucosa_samples(rvk, "fulltexts", "de", False, False)),
        ("qucosa_de_fulltexts_langid_rvk", lambda: _load_qucosa_samples(rvk, "fulltexts", "de", True, False))
    ]

    # filter data sets based on name subset parameter
    if name_subset is not None:
        named_sample_iterators = list(filter(lambda i: i[0] in name_subset, named_sample_iterators))

    for dataset_name, lazy_sample_iterator in named_sample_iterators:
        # load and persist each dataset
        logger.info("load and save persisted dataset %s", dataset_name)
        filepath = os.path.join(QUCOSA_DATASET_CACHE_DIRECTORY, f"{dataset_name}.sqlite")
        dataset = load_persisted_dataset_from_lazy_sample_iterator(lazy_sample_iterator, filepath)
        yield dataset_name, dataset, rvk


def get_qucosa_tfidf_stemming_vectorizer(max_features: int = 10000, cache_vectors=False, fit_only_once: bool = False):
    """Load the tfidf stemming vectorizer that persists stemmed texts for caching."""
    stemming_cache_filepath = os.path.join(CACHE_DIR, "stemming/global_cache.sqlite")

    tfidf_vectorizer = TfidfStemmingVectorizer(
        lang_code="de",
        max_features=max_features,
        stemming_cache_filepath=stemming_cache_filepath,
        ngram_range=(1, 1),
    )
    if not cache_vectors:
        return tfidf_vectorizer

    return CachedVectorizer(tfidf_vectorizer, fit_only_once=fit_only_once)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # loads all data sets and generates persistent storage for them
    for dn, ds, _ in default_named_qucosa_datasets():
        n_unique_subjects = len(unique_subject_order(ds.subjects))
        logger.info(
            "dataset %s has %d documents and %d unique subjects",
            dn, len(ds.documents), n_unique_subjects
        )
