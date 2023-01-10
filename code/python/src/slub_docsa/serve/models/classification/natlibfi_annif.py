"""Setup annif models."""

from slub_docsa.models.classification.natlibfi_annif import AnnifModel
from slub_docsa.serve.common import ModelTypeMapping


def get_annif_classification_models_map() -> ModelTypeMapping:
    """Return a map of classification model types and their generator functions."""
    models = {}
    for lang_code in ["de", "en"]:
        models.update({
            f"annif_tfidf_{lang_code}": lambda subject_hierarchy, subject_order:
                AnnifModel(model_type="tfidf", lang_code=lang_code),
            f"annif_svc_{lang_code}": lambda subject_hierarchy, subject_order:
                AnnifModel(model_type="svc", lang_code=lang_code),
            f"annif_fasttext_{lang_code}": lambda subject_hierarchy, subject_order:
                AnnifModel(model_type="fasttext", lang_code=lang_code),
            f"annif_omikuji_{lang_code}": lambda subject_hierarchy, subject_order:
                AnnifModel(model_type="omikuji", lang_code=lang_code),
            f"annif_yake_{lang_code}": lambda subject_hierarchy, subject_order:
                AnnifModel(
                    model_type="yake", lang_code=lang_code,
                    subject_hierarchy=subject_hierarchy, subject_order=subject_order
                ),
            f"annif_mllm_{lang_code}": lambda subject_hierarchy, subject_order:
                AnnifModel(
                    model_type="mllm", lang_code=lang_code,
                    ubject_hierarchy=subject_hierarchy, subject_order=subject_order
                ),
            # f"annif_stwfsa_{lang_code}": lambda subject_hierarchy, subject_order:
            #     AnnifModel(
            #        model_type="stwfsa", lang_code=lang_code,
            #        subject_hierarchy=subject_hierarchy, subject_order=subject_order
            #     ),
        })
    return models
