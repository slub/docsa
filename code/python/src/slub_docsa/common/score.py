"""Base class describing a scoring function for evaluation."""

from typing import Callable

import numpy as np


ScoreFunctionType = Callable[[np.ndarray, np.ndarray], float]
"""Score function comparing true and predicted subject lists for each document."""

IncidenceDecisionFunctionType = Callable[[np.ndarray], np.ndarray]
"""Convert a subject probabilities matrix to an incidence matrix by applying some decision."""
