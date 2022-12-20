"""Generates a plot illustrating the subject distribution of the Qucosa dataset."""

# pylint: disable=invalid-name

import os
import logging

from slub_docsa.data.load.subjects.jskos import load_jskos_subject_hierarchy_from_sqlite
from slub_docsa.data.load.subjects.rvk import load_rvk_subject_hierarchy_from_sqlite
# from slub_docsa.data.load.k10plus.samples import k10plus_samples_generator
from slub_docsa.data.load.k10plus.slub import k10plus_slub_samples_generator

from slub_docsa.evaluation.dataset.subject_distribution import generate_subject_sunburst

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from slub_docsa.common.paths import get_figures_dir

    subject_schema = "bk"
    doc_lang_code = "en"
    label_lang_code = "de"

    current_subject_hierarchy = {
        "rvk": load_rvk_subject_hierarchy_from_sqlite,
        "ddc": lambda: load_jskos_subject_hierarchy_from_sqlite("ddc", preload_contains=True),
        "bk": lambda: load_jskos_subject_hierarchy_from_sqlite("bk", preload_contains=True),
    }[subject_schema]()

    os.makedirs(os.path.join(get_figures_dir(), "k10plus"), exist_ok=True)
    languages = None if doc_lang_code == "any" else [doc_lang_code]
    # samples = k10plus_public_samples_generator(schemas=[subject_schema], languages=languages, limit=None)
    samples = k10plus_slub_samples_generator(schemas=[subject_schema], languages=languages)

    generate_subject_sunburst(
        current_subject_hierarchy,
        label_lang_code,
        samples,
        max_depth=2,
        use_breadcrumb=False
    ).write_html(
        os.path.join(
            get_figures_dir(),
            f"k10plus/k10plus_slub_with_toc_{subject_schema}_{doc_lang_code}_distribution.html"
        ),
        include_plotlyjs="cdn",
    )

    logger.info("done plotting")
