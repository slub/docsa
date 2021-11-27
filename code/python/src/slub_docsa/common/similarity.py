"""Type definitions for similarity and distance functions between documents."""

from typing import Callable, Optional, Sequence

import numpy as np

from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectTargets

DocumentSimilarityFunction = Callable[[Document, Document], float]

IndexedDocumentSimilarityFunction = Callable[[int, int], float]

DocumentDistanceFunction = Callable[[Document, Document], float]

IndexedDocumentDistanceFunction = Callable[[int, int], float]

IndexedDocumentDistanceGenerator = Callable[
    [Sequence[Document], np.ndarray, Optional[SubjectTargets]],
    IndexedDocumentDistanceFunction
]
