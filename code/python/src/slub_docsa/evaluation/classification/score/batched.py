"""Batched score functions."""

import numpy as np

from slub_docsa.common.score import BatchedMultiClassIncidenceScore, BatchedMultiClassProbabilitiesScore
from slub_docsa.common.score import IncidenceDecisionFunction
from slub_docsa.evaluation.classification.score.common import f1_score, precision_score, recall_score


class BatchedConfusionScore(BatchedMultiClassIncidenceScore):
    """Abstract implementation of a score based on simple incidence counts."""

    def __init__(self):
        """Initialize the batched confusion score with counts."""
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def add_batch(self, true_incidences: np.ndarray, predicted_incidences: np.ndarray):
        """Add a batch of incidence matrices."""
        self.true_positive += (predicted_incidences * true_incidences).sum()
        self.false_positive += ((1 - true_incidences) * predicted_incidences).sum()
        self.false_negative += (true_incidences * (1 - predicted_incidences)).sum()

    def __call__(self) -> float:
        """Abstract method that will calculate a score based on the collected confusion counts."""
        raise NotImplementedError()


class BatchedF1Score(BatchedConfusionScore):
    """Batched score calculating the total f1 score."""

    def __call__(self):
        """Return the f1 score."""
        return f1_score(self.true_positive, self.false_positive, self.false_negative)


class BatchedPrecisionScore(BatchedConfusionScore):
    """Batched score calculating the total precision score."""

    def __call__(self):
        """Return the precision score."""
        return precision_score(self.true_positive, self.false_positive)


class BatchedRecallScore(BatchedConfusionScore):
    """Batched score calculating the total recall score."""

    def __call__(self):
        """Return the recall score."""
        return recall_score(self.true_positive, self.false_negative)


class BatchedIncidenceDecisionConfusionScore(BatchedMultiClassProbabilitiesScore):
    """Combines a batched confusion score and a decision function to score probability matrices."""

    def __init__(self, incidence_decision: IncidenceDecisionFunction, confusion_score: BatchedConfusionScore):
        """Initialize with an incidence matrix and confusion based score."""
        self.incidence_decision = incidence_decision
        self.confusion_score = confusion_score

    def add_batch(self, true_probabilities: np.ndarray, predicted_probabilities: np.ndarray):
        """Add batch of probability matrices."""
        true_incidence = self.incidence_decision(true_probabilities)
        predicted_incidence = self.incidence_decision(predicted_probabilities)
        self.confusion_score.add_batch(true_incidence, predicted_incidence)

    def __call__(self) -> float:
        """Return the confusion based total score."""
        return self.confusion_score.__call__()
