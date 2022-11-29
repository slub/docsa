"""Generates a plot illustrating the subject distribution of the Qucosa dataset."""

# pylint: disable=invalid-name

import os
import logging

from slub_docsa.data.load.subjects.ddc import load_ddc_subject_hierarchy
from slub_docsa.data.load.subjects.rvk import load_rvk_subject_hierarchy_from_sqlite

from slub_docsa.data.load.qucosa import read_qucosa_documents_from_directory, read_qucosa_samples
from slub_docsa.evaluation.dataset.subject_distribution import generate_subject_sunburst
from slub_docsa.evaluation.dataset.subject_distribution import number_of_documents_by_subjects

logger = logging.getLogger(__name__)


def _basic_statistics():
    samples_generator = read_qucosa_samples(read_qucosa_documents_from_directory(), "titles", "rvk", None)
    subjects = number_of_documents_by_subjects(current_subject_hierarchy, samples_generator)
    qucosa_doc_count = sum(1 for x in read_qucosa_documents_from_directory())

    count_no_value = subjects["uri://no_value"]
    count_not_found = subjects["uri://not_found"]

    if subject_schema == "ddc":
        n_known_subjects = "unknown"
    elif subject_schema == "rvk":
        n_known_subjects = sum(1 for _ in current_subject_hierarchy)
    else:
        n_known_subjects = None

    print(f"qucosa has {qucosa_doc_count} documents")
    print(f"qucosa uses only {len(subjects) - 2} of in total {n_known_subjects} unique {subject_schema} subjects")
    print(f"qucosa has {count_no_value} documents containing no subject annotation")
    print(f"qucosa has {count_not_found} documents with invalid annotations")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from slub_docsa.common.paths import get_figures_dir

    # subject_schema = "ddc"
    subject_schema = "rvk"
    language = "de"

    current_subject_hierarchy = {
        "rvk": load_rvk_subject_hierarchy_from_sqlite,
        "ddc": load_ddc_subject_hierarchy,
    }[subject_schema]()

    os.makedirs(os.path.join(get_figures_dir(), "qucosa"), exist_ok=True)
    samples = read_qucosa_samples(
        metadata_variant="titles", subject_schema="rvk", lang_code=None, require_subjects=False
    )
    generate_subject_sunburst(current_subject_hierarchy, language, samples).write_html(
        os.path.join(get_figures_dir(), f"qucosa/{subject_schema}_distribution.html"),
        include_plotlyjs="cdn",
    )

    logger.info("done plotting")
