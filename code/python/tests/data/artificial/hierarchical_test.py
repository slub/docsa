"""Test hierarhical random data generation methods."""

from slub_docsa.data.artificial.hierarchical import generate_hierarchical_subject_token_probabilities
from slub_docsa.data.artificial.hierarchical import generate_hierarchical_random_dataset
from slub_docsa.data.artificial.tokens import generate_random_token_probabilties
from slub_docsa.evaluation.classification.incidence import unique_subject_order


def test_generate_hierarchical_subject_token_probabilities():
    """Test that token probabilities are generated to the specified parameters."""
    n_subjects = 10
    n_tokens = 200

    token_probabilities = generate_random_token_probabilties(n_tokens)
    subject_tp, subject_hierarchy = generate_hierarchical_subject_token_probabilities(n_subjects, token_probabilities)

    assert n_subjects == len(subject_tp)
    assert n_subjects == sum(1 for _ in subject_hierarchy)
    assert set(subject_tp.keys()) == set(subject_hierarchy)


def test_generate_hierarchical_random_dataset():
    """Test that hierarchical dataset is generated to specified parameters."""
    n_tokens = 200
    n_documents = 100
    n_subjects = 10

    dataset, subject_hierarchy = generate_hierarchical_random_dataset(n_tokens, n_documents, n_subjects)

    assert len(dataset.documents) == n_documents
    assert len(dataset.subjects) == n_documents
    assert sum(1 for _ in subject_hierarchy) == n_subjects
    assert min([s_uri in subject_hierarchy for s_uri in unique_subject_order(dataset.subjects)])
