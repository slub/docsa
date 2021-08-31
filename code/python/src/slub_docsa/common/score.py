"""Base class describing a scoring function for evaluation."""

from typing import Callable, Sequence

from slub_docsa.common.subject import SubjectUriList

ScoreFunctionType = Callable[[Sequence[SubjectUriList], Sequence[SubjectUriList]], float]
"""Score function comparing true and predicted subject lists for each document."""
