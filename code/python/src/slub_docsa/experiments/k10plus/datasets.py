"""k10plus datasets."""

from functools import partial
import logging
from typing import Sequence, Tuple

from slub_docsa.evaluation.classification.incidence import unique_subject_order
from slub_docsa.data.load.k10plus.samples import k10plus_public_samples_generator
from slub_docsa.data.load.k10plus.slub import k10plus_slub_samples_generator
from slub_docsa.data.load.subjects.common import subject_hierarchy_by_subject_schema
from slub_docsa.experiments.common.datasets import DatasetTupleList, filter_and_cache_named_datasets, prune_min_samples

logger = logging.getLogger(__name__)


def _filter_k10plus_samples(samples_generator, min_samples, subject_hierarchy_generator):
    return prune_min_samples(samples_generator(), min_samples, subject_hierarchy_generator())


def k10plus_named_datasets_tuple_list(
    languages: Sequence[str] = ("de", "en"),
    schemas: Sequence[str] = ("rvk", "ddc", "bk"),
    variants: Sequence[Tuple[str, int]] = (("public", 100000), ("slub_raw", 20000), ("slub_clean", 20000)),
    min_samples: int = 50,
) -> DatasetTupleList:
    """Return list of k10plus datasets as tuples."""
    datasets: DatasetTupleList = []
    for language in languages:
        for schema in schemas:
            for variant in variants:
                if variant[1] is not None:
                    public_name = f"k10plus_{variant[0]}_{language}_{schema}_ms={min_samples}_limit={variant[1]}"
                else:
                    public_name = f"k10plus_{variant[0]}_{language}_{schema}_ms={min_samples}"
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
                else:
                    raise ValueError(f"unknown variant '{variant}'")
                subject_hierarchy_generator = partial(subject_hierarchy_by_subject_schema, schema=schema)
                samples_generator = partial(
                    _filter_k10plus_samples,
                    samples_generator=samples_generator,
                    min_samples=min_samples,
                    subject_hierarchy_generator=subject_hierarchy_generator
                )
                datasets.append((public_name, samples_generator, subject_hierarchy_generator))

    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("sqlitedict").setLevel(logging.WARNING)

    # loads all data sets and generates persistent storage for them
    dataset_list = k10plus_named_datasets_tuple_list(
        languages=["en"], variants=[("public", None), ("slub_raw", None), ("slub_clean", None)]
    )
    for dn, ds, _ in filter_and_cache_named_datasets(dataset_list):
        n_unique_subjects = len(unique_subject_order(ds.subjects))
        logger.info(
            "dataset %s has %d documents and %d unique subjects",
            dn, len(ds.documents), n_unique_subjects
        )
