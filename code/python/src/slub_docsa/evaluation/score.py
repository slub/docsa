"""Defines various scores that can be used to judge the performance of models."""

from typing import Sequence, Callable
from slub_docsa.common.subject import SubjectUriList

from slub_docsa.evaluation.incidence import unique_subject_list, subject_incidence_matrix_from_list


def scikit_metric(
    metric_function,
    **kwargs
) -> Callable[[Sequence[SubjectUriList], Sequence[SubjectUriList]], float]:
    """Return a scikit-learn metric transformed to score lists of subject URIs."""

    def _metric(
        true_subject_list: Sequence[SubjectUriList],
        predicted_subject_list: Sequence[SubjectUriList],
    ) -> float:

        subject_order = unique_subject_list(true_subject_list)
        true_incidence = subject_incidence_matrix_from_list(true_subject_list, subject_order)
        predicted_incidence = subject_incidence_matrix_from_list(predicted_subject_list, subject_order)

        score = metric_function(true_incidence, predicted_incidence, **kwargs)
        if isinstance(score, float):
            return score

        raise RuntimeError("sklearn metric output is not a float")

    return _metric
