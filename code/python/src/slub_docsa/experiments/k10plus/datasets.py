"""k10plus datasets."""

from functools import partial
import logging
from typing import Sequence, Tuple

from slub_docsa.evaluation.classification.incidence import unique_subject_order
from slub_docsa.data.load.k10plus.samples import k10plus_public_samples_generator
from slub_docsa.data.load.k10plus.slub import k10plus_slub_samples_generator
from slub_docsa.data.load.subjects.common import subject_hierarchy_by_subject_schema
from slub_docsa.experiments.common.datasets import NamedSamplesGenerator
from slub_docsa.experiments.common.datasets import filter_and_cache_named_datasets, prune_min_samples

logger = logging.getLogger(__name__)

DEFAULT_VARIANTS = (
    ("public", 100000), ("public", None),
    ("slub_titles", 20000), ("slub_titles", None),
    ("slub_raw", 20000), ("slub_raw", None),
    ("slub_clean", 20000), ("slub_clean", None),
)


def _filter_k10plus_samples(samples_generator, min_samples, subject_hierarchy_generator):
    return prune_min_samples(samples_generator(), min_samples, subject_hierarchy_generator())


def k10plus_named_sample_generators(
    languages: Sequence[str] = ("de", "en"),
    schemas: Sequence[str] = ("rvk", "ddc", "bk"),
    variants: Sequence[Tuple[str, int]] = DEFAULT_VARIANTS,
    min_samples: int = 50,
) -> Sequence[NamedSamplesGenerator]:
    """Return list of k10plus datasets as tuples."""
    sample_generators: Sequence[NamedSamplesGenerator] = []
    for language in languages:
        for schema in schemas:
            for variant in variants:
                if variant[1] is not None:
                    dataset_name = f"k10plus_{variant[0]}_{language}_{schema}_ms={min_samples}_limit={variant[1]}"
                else:
                    dataset_name = f"k10plus_{variant[0]}_{language}_{schema}_ms={min_samples}"
                if variant[0] == "public":
                    samples_generator = partial(
                        k10plus_public_samples_generator, languages=[language], schemas=[schema], limit=variant[1]
                    )
                elif variant[0] == "slub_raw":
                    samples_generator = partial(
                        k10plus_slub_samples_generator,
                        languages=[language],
                        schemas=[schema],
                        limit=variant[1],
                        clean_toc=False
                    )
                elif variant[0] == "slub_clean":
                    samples_generator = partial(
                        k10plus_slub_samples_generator,
                        languages=[language],
                        schemas=[schema],
                        limit=variant[1],
                        clean_toc=True
                    )
                elif variant[0] == "slub_titles":
                    samples_generator = partial(
                        k10plus_slub_samples_generator,
                        languages=[language],
                        schemas=[schema],
                        limit=variant[1],
                        titles_only=True,
                    )
                else:
                    raise ValueError(f"unknown variant '{variant}'")
                subject_hierarchy_generator = partial(subject_hierarchy_by_subject_schema, schema=schema)
                samples_generator = partial(
                    _filter_k10plus_samples,
                    samples_generator=samples_generator,
                    min_samples=min_samples,
                    subject_hierarchy_generator=subject_hierarchy_generator
                )
                sample_generators.append(
                    NamedSamplesGenerator(
                        dataset_name, samples_generator, schema, subject_hierarchy_generator, [language]
                    )
                )

    return sample_generators


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("sqlitedict").setLevel(logging.WARNING)

    # loads all data sets and generates persistent storage for them
    dataset_list = k10plus_named_sample_generators(
        languages=["de"], variants=[("public", None), ("slub_titles", None), ("slub_raw", None), ("slub_clean", None)]
    )
    for named_dataset in filter_and_cache_named_datasets(dataset_list):
        n_unique_subjects = len(unique_subject_order(named_dataset.dataset.subjects))
        logger.info(
            "dataset %s has %d documents and %d unique subjects",
            named_dataset.name, len(named_dataset.dataset.documents), n_unique_subjects
        )
