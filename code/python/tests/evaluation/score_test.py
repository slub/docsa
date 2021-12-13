"""Test methods for scores."""

import numpy as np

from slub_docsa.common.subject import SubjectHierarchy, SubjectNode
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

    subject_hierarchy: SubjectHierarchy = {
        "uri://subject1": SubjectNode(uri="uri://subject1", label="subject 1", parent_uri=None),
        "uri://subject2": SubjectNode(uri="uri://subject2", label="subject 2", parent_uri=None),
        "uri://subject3": SubjectNode(uri="uri://subject3", label="subject 3", parent_uri="uri://subject2"),
        "uri://subject4": SubjectNode(uri="uri://subject4", label="subject 4", parent_uri="uri://subject2"),
        "uri://subject5": SubjectNode(uri="uri://subject5", label="subject 5", parent_uri="uri://subject3"),
        "uri://subject6": SubjectNode(uri="uri://subject5", label="subject 6", parent_uri="uri://subject3"),
    }

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
        (0.125, [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]),
    ]

    h_loss = cesa_bianchi_h_loss(subject_hierarchy, subject_order)

    for target_score, true_incidence_list, predicted_incidence_list in test_cases:
        assert target_score == h_loss(np.array([true_incidence_list]), np.array([predicted_incidence_list]))
