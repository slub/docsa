"""Simple interface for named models."""

from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

from slub_docsa.common.model import ClassificationModel

NamedModelTupleList = List[Tuple[str, Callable[[], ClassificationModel]]]


class NamedClassificationModels(NamedTuple):
    """Stores names and classes for default models."""

    names: List[str]
    classes: List[ClassificationModel]


def initialize_models_from_tuple_list(
    model_list: NamedModelTupleList,
    name_subset: Optional[Sequence[str]] = None,
):
    """Return an instance of NamedClassificationModels for a list of named model tuples."""
    if name_subset is not None:
        model_list = list(filter(lambda i: i[0] in name_subset, model_list))

    model_names = [i[0] for i in model_list]
    model_classes = [i[1]() for i in model_list]
    return NamedClassificationModels(model_names, model_classes)
