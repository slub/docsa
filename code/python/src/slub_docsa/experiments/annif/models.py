"""Default Annif models that can be used for experimentation."""

from typing import Optional, Sequence

from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.models.classification.natlibfi_annif import AnnifModel
from slub_docsa.experiments.common.models import NamedClassificationModelTupleList


def default_annif_named_model_list(
    lang_code: str,
    subject_order: Optional[Sequence[str]] = None,
    subject_hierarchy: Optional[SubjectHierarchy] = None
) -> NamedClassificationModelTupleList:
    """Return a list of common annif models."""
    models: NamedClassificationModelTupleList = [
        ("annif_tfidf", lambda: AnnifModel(model_type="tfidf", lang_code=lang_code)),
        ("annif_svc", lambda: AnnifModel(model_type="svc", lang_code=lang_code)),
        ("annif_fasttext", lambda: AnnifModel(model_type="fasttext", lang_code=lang_code)),
        ("annif_omikuji", lambda: AnnifModel(model_type="omikuji", lang_code=lang_code)),
        ("annif_vw_multi", lambda: AnnifModel(model_type="vw_multi", lang_code=lang_code)),
        ("annif_mllm", lambda: AnnifModel(
            model_type="mllm",
            lang_code=lang_code,
            subject_order=subject_order,
            subject_hierarchy=subject_hierarchy
        )),
        ("annif_yake", lambda: AnnifModel(
            model_type="yake",
            lang_code=lang_code,
            subject_order=subject_order,
            subject_hierarchy=subject_hierarchy
        )),
        ("annif_stwfsa", lambda: AnnifModel(
            model_type="stwfsa",
            lang_code=lang_code,
            subject_order=subject_order,
            subject_hierarchy=subject_hierarchy
        ))
    ]

    return models
