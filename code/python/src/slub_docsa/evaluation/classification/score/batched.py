"""Batched score functions measuring classification performance."""

import logging

from typing import Callable, Optional, Sequence

import numpy as np

from slub_docsa.common.score import BatchedMultiClassIncidenceScore, BatchedMultiClassProbabilitiesScore
from slub_docsa.common.score import IncidenceDecisionFunction, BatchedPerClassIncidenceScore
from slub_docsa.common.score import BatchedPerClassProbabilitiesScore
from slub_docsa.evaluation.classification.score.common import f1_score, precision_score, recall_score
from slub_docsa.evaluation.classification.incidence import ThresholdIncidenceDecision

logger = logging.getLogger(__name__)


class BatchedConfusionScore(BatchedMultiClassIncidenceScore):
    """Abstract implementation of a score based on simple incidence counts."""

    def __init__(self):
        """Initialize the batched confusion score with counts."""
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def add_batch(self, true_incidences: np.ndarray, predicted_incidences: np.ndarray):
        """Add a batch of incidence matrices.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject incidences in shape (document_batch, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject incidences in shape (document_batch, subjects).
        """
        self.true_positive += (predicted_incidences * true_incidences).sum()
        self.false_positive += ((1 - true_incidences) * predicted_incidences).sum()
        self.false_negative += (true_incidences * (1 - predicted_incidences)).sum()

    def __call__(self) -> float:
        """Abstract method that will calculate a score based on the collected confusion counts."""
        raise NotImplementedError()

    def __str__(self):
        """Return a string representation of the scoring method used for caching."""
        return f"<{self.__class__.__name__}>"


class BatchedF1Score(BatchedConfusionScore):
    """Batched score calculating the total f1 score."""

    def __call__(self) -> float:
        """Return the f1 score."""
        return f1_score(self.true_positive, self.false_positive, self.false_negative)


class BatchedPrecisionScore(BatchedConfusionScore):
    """Batched score calculating the total precision score."""

    def __call__(self) -> float:
        """Return the precision score."""
        return precision_score(self.true_positive, self.false_positive)


class BatchedRecallScore(BatchedConfusionScore):
    """Batched score calculating the total recall score."""

    def __call__(self) -> float:
        """Return the recall score."""
        return recall_score(self.true_positive, self.false_negative)


class BatchedIncidenceDecisionScore(BatchedMultiClassProbabilitiesScore):
    """Combines a batched confusion score and a decision function to score probability matrices."""

    def __init__(self, incidence_decision: IncidenceDecisionFunction, incidence_score: BatchedMultiClassIncidenceScore):
        """Initialize with an incidence matrix and confusion based score.

        Parameters
        ----------
        incidence_decision : IncidenceDecisionFunction
            the incidence decision function that is applied to true and predicted probability matrices before scoring
        incidence_score : BatchedMultiClassIncidenceScore
            the incidence score that is applied to the generated incidence matrices
        """
        self.incidence_decision = incidence_decision
        self.incidence_score = incidence_score

    def add_batch(self, true_probabilities: np.ndarray, predicted_probabilities: np.ndarray):
        """Add batch of probability matrices.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject probabilities in shape (document_batch, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject probabilities in shape (document_batch, subjects).
        """
        true_incidence = self.incidence_decision(true_probabilities)
        predicted_incidence = self.incidence_decision(predicted_probabilities)
        self.incidence_score.add_batch(true_incidence, predicted_incidence)

    def __call__(self) -> float:
        """Return the total score."""
        return self.incidence_score.__call__()

    def __str__(self):
        """Return a string representation of the scoring method used for caching."""
        return f"<BatchedIncidenceDecisionConfusionScore incidence_decision={str(self.incidence_decision)} " \
            + f"incidence_score={str(self.incidence_score)}>"


class BatchedBestThresholdScore(BatchedMultiClassProbabilitiesScore):
    """Calculates score for multiple incidence thresholds and returns the best one."""

    def __init__(
        self,
        score_generator: Callable[[], BatchedMultiClassIncidenceScore],
        optimizer_generator: Optional[Callable[[], BatchedMultiClassIncidenceScore]] = None,
        thresholds: Sequence[float] = None,
    ):
        """Initialize with incidence score whose score is returned for the best threshold.

        Parameters
        ----------
        score_generator : Callable[[], BatchedMultiClassIncidenceScore]
            the incidence score as generator function
        optimizer_generator : Optional[Callable[[], BatchedMultiClassIncidenceScore]], optional
            the incidence score that is used to find the best score, if None, the f1 score is used
        thresholds : Sequence[float], optional
            the sequence of thresholds that are checked, by default [0.1, 0.2, ..., 0.9]
        """
        self.optimizer_generator = optimizer_generator or BatchedF1Score
        self.score_generator = score_generator
        self.thresholds = thresholds or [i / 10.0 + 0.1 for i in range(9)]

        self.optimizers = [self.optimizer_generator() for _ in self.thresholds]
        self.scores = [self.score_generator() for _ in self.thresholds]

    def add_batch(self, true_probabilities: np.ndarray, predicted_probabilities: np.ndarray):
        """Add batch of probability matrices.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject probabilities in shape (document_batch, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject probabilities in shape (document_batch, subjects).
        """
        for threshold in self.thresholds:
            true_incidence = ThresholdIncidenceDecision(threshold)(true_probabilities)
            predicted_incidence = ThresholdIncidenceDecision(threshold)(predicted_probabilities)

            for optimizer in self.optimizers:
                optimizer.add_batch(true_incidence, predicted_incidence)

            for score in self.scores:
                score.add_batch(true_incidence, predicted_incidence)

    def __call__(self) -> float:
        """Return the best threshold score."""
        optimizer_values = [optimizer() for optimizer in self.optimizers]
        max_threshold_idx = np.argmax(optimizer_values)
        logger.debug("return score for best threshold=%s", str(self.thresholds[max_threshold_idx]))
        return self.scores[max_threshold_idx]()

    def __str__(self):
        """Return a string representation of the scoring method used for caching."""
        return f"<BatchedBestThresholdScore score_generator={str(self.score_generator())} " \
            + f"optimizer_generator={str(self.optimizer_generator())} thresholds={str(self.thresholds)}>"


class BatchedPerClassConfusionScore(BatchedPerClassIncidenceScore):
    """Abstract implementation of a score based on simple incidence counts."""

    def __init__(self):
        """Initialize the batched confusion score with counts."""
        self.true_positive = None
        self.false_positive = None
        self.false_negative = None

    def add_batch(self, true_incidences: np.ndarray, predicted_incidences: np.ndarray):
        """Add a batch of incidence matrices.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject probabilities in shape (document_batch, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject probabilities in shape (document_batch, subjects).
        """
        if self.true_positive is None or self.false_positive is None or self.false_negative is None:
            self.true_positive = np.zeros(true_incidences.shape[1])
            self.false_positive = np.zeros(true_incidences.shape[1])
            self.false_negative = np.zeros(true_incidences.shape[1])
        self.true_positive += (predicted_incidences * true_incidences).sum(axis=0)
        self.false_positive += ((1 - true_incidences) * predicted_incidences).sum(axis=0)
        self.false_negative += (true_incidences * (1 - predicted_incidences)).sum(axis=0)

    def __call__(self) -> Sequence[float]:
        """Abstract method that will calculate a score based on the collected confusion counts."""
        raise NotImplementedError()

    def __str__(self):
        """Return a string representation of the scoring method used for caching."""
        return f"<{self.__class__.__name__}>"


class BatchedPerClassF1Score(BatchedPerClassConfusionScore):
    """Batched score calculating the f1 score for each subject."""

    def __call__(self) -> Sequence[float]:
        """Return the f1 score for each subject."""
        return f1_score(self.true_positive, self.false_positive, self.false_negative)


class BatchedPerClassPrecisionScore(BatchedPerClassConfusionScore):
    """Batched score calculating the precision score for each subject."""

    def __call__(self) -> Sequence[float]:
        """Return the precision score for each subject."""
        return precision_score(self.true_positive, self.false_positive)


class BatchedPerClassRecallScore(BatchedPerClassConfusionScore):
    """Batched score calculating the recall score for each subject."""

    def __call__(self) -> Sequence[float]:
        """Return the recall score for each subject."""
        return recall_score(self.true_positive, self.false_negative)


class BatchedIncidenceDecisionPerClassScore(BatchedPerClassProbabilitiesScore):
    """Combines a batched incidence score and a decision function to score probability matrices for each class."""

    def __init__(self, incidence_decision: IncidenceDecisionFunction, incidence_score: BatchedPerClassIncidenceScore):
        """Initialize with an incidence matrix and incidence score.

        Parameters
        ----------
        incidence_decision : IncidenceDecisionFunction
            the incidence decision function that is applied to true and predicted probability matrices before scoring
        incidence_score : BatchedPerClassIncidenceScore
            the incidence score that is applied to the generated incidence matrices
        """
        self.incidence_decision = incidence_decision
        self.confusion_score = incidence_score

    def add_batch(self, true_probabilities: np.ndarray, predicted_probabilities: np.ndarray):
        """Add batch of probability matrices.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject probabilities in shape (document_batch, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject probabilities in shape (document_batch, subjects).
        """
        true_incidence = self.incidence_decision(true_probabilities)
        predicted_incidence = self.incidence_decision(predicted_probabilities)
        self.confusion_score.add_batch(true_incidence, predicted_incidence)

    def __call__(self) -> Sequence[float]:
        """Return the total score for each class."""
        return self.confusion_score.__call__()

    def __str__(self):
        """Return a string representation of the scoring method used for caching."""
        return f"<BatchedIncidenceDecisionPerClassConfusionScore incidence_decision={str(self.incidence_decision)} " \
            + f"confusion_score={str(self.confusion_score)}>"


class BatchedNumberOfTestExamplesPerClass(BatchedPerClassProbabilitiesScore):
    """Count the number of test examples for each subject."""

    def __init__(self):
        """Initialize."""
        self.counts = None

    def add_batch(self, true_probabilities: np.ndarray, predicted_probabilities: np.ndarray):
        """Add batch of probability matrices.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject probabilities in shape (document_batch, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject probabilities in shape (document_batch, subjects).
        """
        if self.counts is None:
            self.counts = np.zeros(true_probabilities.shape[1])
        self.counts += (true_probabilities > 0.0).sum(axis=0)

    def __call__(self) -> Sequence[float]:
        """Return the number of test examples for each subject."""
        return self.counts

    def __str__(self):
        """Return a string representation of the scoring method used for caching."""
        return "<BatchedNumberOfTestExamplesPerClass>"
