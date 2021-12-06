"""Default Annif models that can be used for experimentation."""

from typing import Optional, Sequence

from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.models.classification.natlibfi_annif import AnnifModel
from slub_docsa.experiments.common.models import NamedClassificationModelTupleList


def default_annif_named_model_list(
    lang_code: str,
    subject_order: Optional[Sequence[str]] = None,
    subject_hierarchy: Optional[SubjectHierarchyType[SubjectNodeType]] = None
) -> NamedClassificationModelTupleList:
    """Return a list of common annif models."""
    models: NamedClassificationModelTupleList = [
        ("annif tfidf", lambda: AnnifModel(model_type="tfidf", lang_code=lang_code)),
        ("annif svc", lambda: AnnifModel(model_type="svc", lang_code=lang_code)),
        ("annif fasttext", lambda: AnnifModel(model_type="fasttext", lang_code=lang_code)),
        ("annif omikuji", lambda: AnnifModel(model_type="omikuji", lang_code=lang_code)),
        ("annif vw_multi", lambda: AnnifModel(model_type="vw_multi", lang_code=lang_code)),
        ("annif mllm", lambda: AnnifModel(
            model_type="mllm",
            lang_code=lang_code,
            subject_order=subject_order,
            subject_hierarchy=subject_hierarchy
        )),
        ("annif yake", lambda: AnnifModel(
            model_type="yake",
            lang_code=lang_code,
            subject_order=subject_order,
            subject_hierarchy=subject_hierarchy
        )),
        ("annif stwfsa", lambda: AnnifModel(
            model_type="stwfsa",
            lang_code=lang_code,
            subject_order=subject_order,
            subject_hierarchy=subject_hierarchy
        ))
    ]

    return models
