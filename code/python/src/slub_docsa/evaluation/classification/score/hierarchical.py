"""Defines hierarchical scores that can be used to judge the classification performance."""

# pylint: disable=too-many-locals, unused-argument

import logging
import math
from typing import Callable, Optional, Sequence

import numpy as np

from slub_docsa.common.score import BatchedMultiClassIncidenceScore, MultiClassIncidenceScore
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.preprocess.subject import subject_ancestors_list
from slub_docsa.evaluation.classification.incidence import extend_incidence_list_to_ancestors

logger = logging.getLogger(__name__)


def cesa_bianchi_loss_for_sample_generator(
    subject_hierarchy: Optional[SubjectHierarchy],
    subject_order: Optional[Sequence[str]],
    log_factor: Optional[float] = None,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return function that calculates the h-loss according to cesa-bianchi et al. for a single sample.

    The h-loss is a hierarchical score that considers not only whether subjects are predicted correctly, but also uses
    the subject hierarchy to weigh mistakes based on their severness. Mistakes near the root of the hierarchy are
    judged to be more severe, and thus, add a higher loss to the overall score.

    A detailed description can be found in:

    N. Cesa-Bianchi, C. Gentile, and L. Zaniboni.
    [Hierarchical classification: Combining Bayes with SVM](
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.98.7332&rep=rep1&type=pdf) (2006)

    Parameters
    ----------
    subject_hierarchy: Optional[SubjectHierarchy]
        the subject hierarchy used to evaluate the serverness of prediction mistakes; if no subject hierarchy is
        provided, a score of `numpy.nan` is returned
    subject_order: Optional[Sequence[str]]
        an subject list mapping subject URIs to the columns of the provided target / predicted incidence matrices; if
        no subject order is provided, a score of `numpy.nan` is returned
    log_factor: Optional[float] = None
        factor used for logartihmic scaling, which helps to visualize the h-loss in plots

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
        a function that scores two incidence arrays for a single sample (target incidence, predicted incidence) using
        the cesa-bianchi h-loss given the provided subject hierarchy and subject order
    """

    def _nan_results(_true_incidence: np.ndarray, _predicted_incidence: np.ndarray) -> float:
        return np.nan

    if subject_hierarchy is None or subject_order is None:
        # always return nan if there is no subject hierarchy provided
        return _nan_results

    number_of_root_nodes = sum(1 for _ in subject_hierarchy.root_subjects())

    def _find_ancestor_with_error(
            subject_uri: str,
            subject_hierarchy: SubjectHierarchy,
            incidence_list: Sequence[int],
    ) -> str:
        if subject_order is None:
            raise ValueError("can't find ancestor without subject order")
        ancestors = subject_ancestors_list(subject_uri, subject_hierarchy)
        previous_ancestor = ancestors[-1]
        for ancestor in reversed(ancestors[:-1]):
            if ancestor in subject_order:
                ancestor_id = subject_order.index(ancestor)
                if incidence_list[ancestor_id] == 1:
                    break
            previous_ancestor = ancestor
        return previous_ancestor

    def _h_loss(true_array: np.ndarray, predicted_array: np.ndarray) -> float:
        subjects_with_errors = set()
        true_list = true_array.tolist()
        pred_list = predicted_array.tolist()

        ext_true_list = extend_incidence_list_to_ancestors(subject_hierarchy, subject_order, true_list)
        ext_pred_list = extend_incidence_list_to_ancestors(subject_hierarchy, subject_order, pred_list)

        # look for false negative errors
        for i, value in enumerate(true_list):
            if value == 1 and ext_pred_list[i] == 0:
                # there is an error here, lets find out at which level
                subject_uri = subject_order[i]
                subjects_with_errors.add(_find_ancestor_with_error(
                    subject_uri,
                    subject_hierarchy,
                    ext_pred_list
                ))

        # look for false positive errors
        for i, value in enumerate(pred_list):
            if value == 1 and ext_true_list[i] == 0:
                subject_uri = subject_order[i]
                subjects_with_errors.add(_find_ancestor_with_error(
                    subject_uri,
                    subject_hierarchy,
                    ext_true_list
                ))

        nonlocal number_of_root_nodes
        loss = 0.0
        for subject_uri in subjects_with_errors:
            ancestors = subject_ancestors_list(subject_uri, subject_hierarchy)
            max_level_loss = 1.0
            for ancestor in ancestors:
                ancestor_parent = subject_hierarchy.subject_parent(ancestor)
                if ancestor_parent is None:
                    max_level_loss *= 1.0 / number_of_root_nodes
                else:
                    number_of_siblings = len(subject_hierarchy.subject_children(ancestor_parent))
                    max_level_loss *= 1.0 / number_of_siblings
            loss += max_level_loss

        if log_factor is not None:
            return math.log(1 + loss * log_factor, 2) / math.log(log_factor + 1, 2)
        return loss

    return _h_loss


def cesa_bianchi_loss_generator(
    subject_hierarchy: Optional[SubjectHierarchy],
    subject_order: Optional[Sequence[str]],
    log_factor: Optional[float] = None,
) -> MultiClassIncidenceScore:
    """Return Cesa-Bianchi score function comparing true and predicted incidence matrices.

    See `cesa_bianchi_loss_for_sample_generator` for additional information.

    Parameters
    ----------
    subject_hierarchy: Optional[SubjectHierarchy]
        the subject hierarchy used to evaluate the serverness of prediction mistakes; if no subject hierarchy is
        provided, a score of `numpy.nan` is returned
    subject_order: Optional[Sequence[str]]
        an subject list mapping subject URIs to the columns of the provided target / predicted incidence matrices;
        if no subject order is provided, a score of `numpy.nan` is returned
    log_factor: Optional[float] = None
        factor used for logartihmic scaling, which helps to visualize the h-loss in plots
    """
    loss_per_sample = cesa_bianchi_loss_for_sample_generator(subject_hierarchy, subject_order, log_factor)

    def _loss(true_incidences: np.ndarray, predicted_incidences: np.ndarray) -> float:
        h_losses = list(map(loss_per_sample, true_incidences, predicted_incidences))
        return np.average(h_losses)

    return _loss


class BatchedCesaBianchiIncidenceLoss(BatchedMultiClassIncidenceScore):
    """Batched Cesa-Bianchi incidence score calculation.

    See `cesa_bianchi_loss_for_sample_generator` for additional information.
    """

    def __init__(
        self,
        subject_hierarchy: Optional[SubjectHierarchy],
        subject_order: Optional[Sequence[str]],
        log_factor: Optional[float] = None,
    ):
        """Initialize a Cesa-Bianchi incidence loss.

        Parameters
        ----------
        subject_hierarchy: Optional[SubjectHierarchy]
            the subject hierarchy used to evaluate the serverness of prediction mistakes; if no subject hierarchy is
            provided, a score of `numpy.nan` is returned
        subject_order: Optional[Sequence[str]]
            an subject list mapping subject URIs to the columns of the provided target / predicted incidence matrices;
            if no subject order is provided, a score of `numpy.nan` is returned
        log_factor: Optional[float] = None
            factor used for logartihmic scaling, which helps to visualize the h-loss in plots
        """
        self.loss_function = cesa_bianchi_loss_for_sample_generator(subject_hierarchy, subject_order, log_factor)
        self.losses = []

    def add_batch(self, true_incidences: np.ndarray, predicted_incidences: np.ndarray):
        """Add multi-class subject incidence matrices for a batch of documents to be processed for scoring.

        Parameters
        ----------
        true_incidences: np.ndarray
            the matrix containing the true subject incidences in shape (document_batch, subjects).
        predicted_incidences: np.ndarray
            the matrix containing predicted subject incidences in shape (document_batch, subjects).
        """
        self.losses.extend([self.loss_function(ti, pi) for ti, pi in zip(true_incidences, predicted_incidences)])

    def __call__(self) -> float:
        """Return current score comparing multi-class subject incidences that were added in batches."""
        return np.average(self.losses)
