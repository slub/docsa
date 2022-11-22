"""Test methods for scores."""

import numpy as np

from slub_docsa.common.subject import SubjectTuple
from slub_docsa.data.preprocess.subject import build_subject_hierarchy_from_subject_tuples
from slub_docsa.evaluation.score import cesa_bianchi_h_loss


def test_cesa_bianchi_h_loss():
    """Test h-loss for specific example hierarchy and incidence matrices.

    Subject Hierarchy
    - subject 1
    - subject 2
      - subject 3
        - subject 5
        - subject 6
      - subject 4
    """
    subject_order = [
        "uri://subject1",
        "uri://subject2",
        "uri://subject3",
        "uri://subject4",
        "uri://subject5",
        "uri://subject6",
    ]

    subject_hierarchy = build_subject_hierarchy_from_subject_tuples([
        SubjectTuple("uri://subject1", {"en": "subject 1"}, None),
        SubjectTuple("uri://subject2", {"en": "subject 2"}, None),
        SubjectTuple("uri://subject3", {"en": "subject 3"}, "uri://subject2"),
        SubjectTuple("uri://subject4", {"en": "subject 4"}, "uri://subject2"),
        SubjectTuple("uri://subject5", {"en": "subject 5"}, "uri://subject3"),
        SubjectTuple("uri://subject6", {"en": "subject 6"}, "uri://subject3"),
    ])

    test_cases = [
        # score, true incidence, predicted incidence
        (0.0, [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]),
        (0.5, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
        (1.0, [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]),
        (1.0, [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1]),
        #
        (0.50, [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]),
        (0.25, [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 0]),
        (0.25, [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0]),
        (0.00, [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0]),
        #
        (0.125, [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1]),
    ]

    h_loss = cesa_bianchi_h_loss(subject_hierarchy, subject_order)

    for target_score, true_incidence_list, predicted_incidence_list in test_cases:
        assert target_score == h_loss(np.array([true_incidence_list]), np.array([predicted_incidence_list]))
