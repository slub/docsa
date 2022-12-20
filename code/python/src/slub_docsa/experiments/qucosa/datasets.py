"""Provides various variants of the qucosa dataset."""

# pylint: disable=too-many-arguments

import logging

from typing import Iterator, Sequence, Union, Optional
from typing_extensions import Literal

from slub_docsa.common.sample import Sample
from slub_docsa.data.load.qucosa import read_qucosa_samples
from slub_docsa.data.load.qucosa import read_qucosa_documents_from_directory
from slub_docsa.data.load.subjects.common import subject_hierarchy_by_subject_schema
from slub_docsa.data.preprocess.language import filter_samples_by_detected_language_via_langid
from slub_docsa.evaluation.classification.incidence import unique_subject_order
from slub_docsa.experiments.common.datasets import NamedSamplesGenerator, filter_and_cache_named_datasets
from slub_docsa.experiments.common.datasets import filter_min_samples, prune_by_level, prune_min_samples

logger = logging.getLogger(__name__)


def _load_qucosa_samples(
    subject_schema: Union[Literal["rvk"], Literal["ddc"]],
    text_source: str = "titles",
    lang_code: Optional[str] = None,
    langid_check: bool = False,
    pruning_method: str = "min_samples_10",
    check_qucosa_download: bool = False,
) -> Iterator[Sample]:
    subject_hierarchy = subject_hierarchy_by_subject_schema(subject_schema)
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
        return filter_min_samples(sample_iterator, 10)
    if pruning_method == "min_samples_10":
        return prune_min_samples(sample_iterator, 10, subject_hierarchy)
    if pruning_method == "level_1":
        return prune_by_level(sample_iterator, 1, 10, subject_hierarchy)
    if pruning_method == "level_2":
        return prune_by_level(sample_iterator, 2, 10, subject_hierarchy)
    raise RuntimeError("unknown pruning method")


def qucosa_named_sample_generators(
    check_qucosa_download: bool = False,
) -> Sequence[NamedSamplesGenerator]:
    """Return list of qucosa datasets as tuples."""
    def lazy_rvk():
        return subject_hierarchy_by_subject_schema("rvk")

    def lazy_ddc():
        return subject_hierarchy_by_subject_schema("ddc")

    cqd = check_qucosa_download

    datasets: Sequence[NamedSamplesGenerator] = [
        NamedSamplesGenerator(
            "qucosa_all_titles_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "titles", None, False, "min_samples_10", cqd
            ), "rvk", lazy_rvk, ["de", "en"]),
        NamedSamplesGenerator(
            "qucosa_all_titles_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "titles", None, False, "min_samples_10", cqd
            ), "ddc", lazy_ddc, ["de", "en"]),
        NamedSamplesGenerator(
            "qucosa_de_titles_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "titles", "de", False, "min_samples_10", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_titles_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "titles", "de", False, "min_samples_10", cqd
            ), "ddc", lazy_ddc, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_titles_langid_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "titles", "de", True, "min_samples_10", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_titles_langid_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "titles", "de", True, "min_samples_10", cqd
            ), "ddc", lazy_ddc, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_complete_but_only_titles_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "complete_but_only_titles", "de", False, "min_samples_10", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_abstracts_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "abstracts", "de", False, "min_samples_10", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_abstracts_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "abstracts", "de", False, "min_samples_10", cqd
            ), "ddc", lazy_ddc, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_abstracts_langid_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "abstracts", "de", True, "min_samples_10", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_abstracts_langid_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "abstracts", "de", True, "min_samples_10", cqd
            ), "ddc", lazy_ddc, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_complete_but_only_abstracts_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "complete_but_only_abstracts", "de", False, "min_samples_10", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_fulltexts_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "fulltexts", "de", False, "min_samples_10", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_fulltexts_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "fulltexts", "de", False, "min_samples_10", cqd
            ), "ddc", lazy_ddc, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_fulltexts_langid_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "fulltexts", "de", True, "min_samples_10", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_fulltexts_langid_rvk_level_1",
            lambda: _load_qucosa_samples(
                "rvk", "fulltexts", "de", True, "level_1", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_fulltexts_langid_rvk_level_2",
            lambda: _load_qucosa_samples(
                "rvk", "fulltexts", "de", True, "level_2", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_fulltexts_langid_rvk_no_pruning",
            lambda: _load_qucosa_samples(
                "rvk", "fulltexts", "de", True, "no_pruning", cqd
            ), "rvk", lazy_rvk, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_fulltexts_langid_ddc",
            lambda: _load_qucosa_samples(
                "ddc", "fulltexts", "de", True, "min_samples_10", cqd
            ), "ddc", lazy_ddc, ["de"]),
        NamedSamplesGenerator(
            "qucosa_de_complete_but_only_fulltexts_rvk",
            lambda: _load_qucosa_samples(
                "rvk", "complete_but_only_fulltexts", "de", False, "min_samples_10", cqd
            ), "rvk", lazy_rvk, ["de"]),
    ]
    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # loads all data sets and generates persistent storage for them
    named_sample_generators = qucosa_named_sample_generators(check_qucosa_download=False)
    for named_dataset in filter_and_cache_named_datasets(named_sample_generators):
        n_unique_subjects = len(unique_subject_order(named_dataset.dataset.subjects))
        logger.info(
            "dataset %s has %d documents and %d unique subjects",
            named_dataset.name, len(named_dataset.dataset.documents), n_unique_subjects
        )
