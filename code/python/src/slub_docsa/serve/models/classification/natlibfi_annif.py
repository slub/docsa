"""Setup annif models."""

from slub_docsa.models.classification.natlibfi_annif import AnnifModel
from slub_docsa.serve.common import ModelTypeMapping


def get_annif_classification_models_map() -> ModelTypeMapping:
    """Return a map of classification model types and their generator functions."""
    return {
        "annif_tfidf_de": lambda subject_hierarchy, subject_order: AnnifModel(model_type="tfidf", lang_code="de"),
        "annif_svc_de": lambda subject_hierarchy, subject_order: AnnifModel(model_type="svc", lang_code="de"),
        "annif_fasttext_de": lambda subject_hierarchy, subject_order: AnnifModel(model_type="fasttext", lang_code="de"),
        "annif_omikuji_de": lambda subject_hierarchy, subject_order: AnnifModel(model_type="omikuji", lang_code="de"),
        "annif_yake_de": lambda subject_hierarchy, subject_order: AnnifModel(
            model_type="yake", lang_code="de", subject_hierarchy=subject_hierarchy, subject_order=subject_order
        ),
        "annif_mllm_de": lambda subject_hierarchy, subject_order: AnnifModel(
            model_type="mllm", lang_code="de", subject_hierarchy=subject_hierarchy, subject_order=subject_order
        ),
        # "annif_stwfsa_de": lambda subject_hierarchy, subject_order: AnnifModel(
        #     model_type="stwfsa", lang_code="de", subject_hierarchy=subject_hierarchy, subject_order=subject_order
        # ),
    }
