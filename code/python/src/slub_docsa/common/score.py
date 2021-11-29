"""Base class describing a scoring function for evaluation.

Note: Type aliases `MultiClassScoreFunctionType`, `BinaryClassScoreFunctionType` and `IncidenceDecisionFunctionType`
are not correctly described in API documentation due to [issue in pdoc](https://github.com/pdoc3/pdoc/issues/229).
"""

from typing import Callable, Optional, Sequence

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectTargets


GenericScoreFunctionType = Callable[[np.ndarray, np.ndarray], float]

MultiClassScoreFunctionType = Callable[[np.ndarray, np.ndarray], float]
"""Score function comparing true and predicted subject probabilities for multiple classes for each document."""

BinaryClassScoreFunctionType = Callable[[np.ndarray, np.ndarray], float]
"""Score function comparing true and predicted subject probabilities for a single class for each document."""

IncidenceDecisionFunctionType = Callable[[np.ndarray], np.ndarray]
"""Convert a subject probabilities matrix to an incidence matrix by applying some decision."""

ClusteringScoreFunction = Callable[[Sequence[Document], np.ndarray, Optional[SubjectTargets]], float]
"""Score function type used to evaluate a clustering of documents."""
