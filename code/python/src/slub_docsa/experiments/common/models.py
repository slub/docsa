"""Simple interface for named models."""

from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

from slub_docsa.common.model import ClassificationModel, ClusteringModel
from slub_docsa.serve.common import ModelTypeMapping

NamedClassificationModelTupleList = List[Tuple[str, Callable[[], ClassificationModel]]]
NamedClusteringModelTupleList = List[Tuple[str, Callable[[], ClusteringModel]]]


class NamedClassificationModels(NamedTuple):
    """Stores names and classes for classification models."""

    names: List[str]
    generators: List[Callable[[], ClassificationModel]]


class NamedClusteringModels(NamedTuple):
    """Stores names and classes for a list of clustering models."""

    names: List[str]
    classes: List[ClusteringModel]


def filter_model_type_mapping(
    model_types: ModelTypeMapping,
    name_subset: Sequence[str],
):
    """Filter a model type map by a subset of model types."""
    return {name: generator for name, generator in model_types.items() if name in name_subset}


def initialize_classification_models_from_tuple_list(
    model_list: NamedClassificationModelTupleList,
    name_subset: Optional[Sequence[str]] = None,
) -> NamedClassificationModels:
    """Return an instance of NamedClassificationModels for a list of named model tuples."""
    if name_subset is not None:
        model_list = list(filter(lambda i: i[0] in name_subset, model_list))

    model_names = [i[0] for i in model_list]
    model_generators = [i[1] for i in model_list]
    return NamedClassificationModels(model_names, model_generators)


def initialize_clustering_models_from_tuple_list(
    model_list: NamedClusteringModelTupleList,
    name_subset: Optional[Sequence[str]] = None,
) -> NamedClusteringModels:
    """Return an instance of NamedClusteringModels for a list of named model tuples."""
    if name_subset is not None:
        model_list = list(filter(lambda i: i[0] in name_subset, model_list))

    model_names = [i[0] for i in model_list]
    model_classes = [i[1]() for i in model_list]
    return NamedClusteringModels(model_names, model_classes)
