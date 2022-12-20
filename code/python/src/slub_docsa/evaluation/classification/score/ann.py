"""Torch based F1 scoring."""

import torch
import numpy as np

from slub_docsa.evaluation.classification.score.common import f1_score, subject_score_distribution_from_scores


class TorchF1Score:
    """Batched F1 score based on torch Tensors comparing true and predicted subject probabilities."""

    def __init__(self):
        """Initialize torch based f1 score."""
        self.t01_counts = None
        self.t05_counts = None

    def _initalialize_counts(self, number_of_subjects):
        self.t01_counts = [
            np.zeros(number_of_subjects, dtype=np.int64),  # true positive
            np.zeros(number_of_subjects, dtype=np.int64),  # false positive
            np.zeros(number_of_subjects, dtype=np.int64),  # false negative
        ]
        self.t05_counts = [
            np.zeros(number_of_subjects, dtype=np.int64),  # true positive
            np.zeros(number_of_subjects, dtype=np.int64),  # false positive
            np.zeros(number_of_subjects, dtype=np.int64),  # false negative
        ]

    def _add_counts(self, counts, true_inc, pred_inc):
        counts[0] += (true_inc * pred_inc).sum(dim=0).cpu().detach().numpy().astype(np.uint32)
        counts[1] += ((~true_inc) * pred_inc).sum(dim=0).cpu().detach().numpy().astype(np.uint32)
        counts[2] += (true_inc * (~pred_inc)).sum(dim=0).cpu().detach().numpy().astype(np.uint32)

    def add_batch(self, true_probabilities: torch.Tensor, predicted_probabilities: torch.Tensor):
        """Add multi-class subject probability matrices for a batch of documents to be processed for scoring.

        Parameters
        ----------
        true_probabilities: np.ndarray
            the matrix containing the true subject probabilities in shape (document_batch, subjects).
        predicted_probabilities: np.ndarray
            the matrix containing predicted subject probabilities in shape (document_batch, subjects).
        """
        if self.t01_counts is None or self.t05_counts is None:
            self._initalialize_counts(true_probabilities.shape[1])

        true_incidences = true_probabilities >= 0.5
        t01_predicted_incidences = predicted_probabilities >= 0.1
        t05_predicted_incidences = predicted_probabilities >= 0.5

        self._add_counts(self.t01_counts, true_incidences, t01_predicted_incidences)
        self._add_counts(self.t05_counts, true_incidences, t05_predicted_incidences)

    def __call__(self) -> float:
        """Return f1 scores as tuples of (total, distribution) for both t=0.1 and t=0.5."""
        if self.t01_counts is None or self.t05_counts is None:
            distribution = subject_score_distribution_from_scores([])
            return (0.0, distribution), (0.0, distribution)
        t01_f1 = f1_score(*map(np.sum, self.t01_counts))
        t05_f1 = f1_score(*map(np.sum, self.t05_counts))
        t01_f1_distribution = subject_score_distribution_from_scores(f1_score(*self.t01_counts))
        t05_f1_distribution = subject_score_distribution_from_scores(f1_score(*self.t05_counts))
        return (t01_f1, t01_f1_distribution), (t05_f1, t05_f1_distribution)


if __name__ == "__main__":

    true = torch.Tensor([[0, 1], [1, 0], [1, 1]]).to("cuda")
    pred = torch.Tensor([[0, 0.8], [0.2, 0], [0.3, 0.7]]).to("cuda")

    score = TorchF1Score()
    score.add_batch(true, pred)
    print(score())
