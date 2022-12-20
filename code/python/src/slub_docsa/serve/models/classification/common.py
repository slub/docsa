"""All model types."""

from slub_docsa.serve.models.classification.ann import get_ann_classification_models_map
from slub_docsa.serve.models.classification.classic import get_classic_classification_models_map
from slub_docsa.serve.models.classification.dbmdz import get_dbmdz_classification_models_map
from slub_docsa.serve.models.classification.natlibfi_annif import get_annif_classification_models_map


def get_all_classification_model_types():
    """Return all available model types.

    Model types include:
    - classic models, see `slub_docsa.serve.models.classification.classic`
    - annif models, see `slub_docsa.serve.models.classification.natlibfi_annif`
    - models based on artificial neural networks, see `slub_docsa.serve.models.classification.ann`
    - models using the pre-trained Dbmdz BERT transformer for vectorization, see
      `slub_docsa.serve.models.classification.dbmdz`
    """
    model_types = {}
    model_types.update(get_classic_classification_models_map())
    model_types.update(get_annif_classification_models_map())
    model_types.update(get_ann_classification_models_map())
    model_types.update(get_dbmdz_classification_models_map())
    return model_types
