"""k10plus datasets."""

from functools import partial
import logging
from typing import Optional

from slub_docsa.evaluation.classification.incidence import unique_subject_order
from slub_docsa.data.load.k10plus.samples import k10plus_public_samples_generator
from slub_docsa.data.load.k10plus.slub import k10plus_slub_samples_generator
from slub_docsa.data.load.subjects.common import subject_hierarchy_by_subject_schema
from slub_docsa.experiments.common.datasets import DatasetTupleList, filter_and_cache_named_datasets

logger = logging.getLogger(__name__)


def k10plus_named_datasets_tuple_list(limit: Optional[int] = 10000) -> DatasetTupleList:
    """Return list of k10plus datasets as tuples."""
    datasets: DatasetTupleList = []
    for language in ["de", "en"]:
        for schema in ["rvk", "ddc", "bk"]:
            for variant in ["public", "slub"]:
                if limit is not None:
                    public_name = f"k10plus_{variant}_{language}_{schema}_limit={limit}"
                else:
                    public_name = f"k10plus_{variant}_{language}_{schema}"
                if variant == "public":
                    sample_generator = partial(
                        k10plus_public_samples_generator, languages=[language], schemas=[schema], limit=limit
                    )
                else:
                    sample_generator = partial(
                        k10plus_slub_samples_generator, languages=[language], schemas=[schema], limit=limit
                    )
                subject_hierarchy_generator = partial(subject_hierarchy_by_subject_schema, schema=schema)
                datasets.append((public_name, sample_generator, subject_hierarchy_generator))

    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # loads all data sets and generates persistent storage for them
    dataset_list = k10plus_named_datasets_tuple_list()
    for dn, ds, _ in filter_and_cache_named_datasets(dataset_list):
        n_unique_subjects = len(unique_subject_order(ds.subjects))
        logger.info(
            "dataset %s has %d documents and %d unique subjects",
            dn, len(ds.documents), n_unique_subjects
        )
