"""Tests for random evaluation data generators."""

import numpy as np

from slub_docsa.data.artificial.tokens import generate_random_token_probabilties


def test_generate_random_token_probabilties():
    """Test random token generator."""
    n_tokens = 1000
    token_probabilities = generate_random_token_probabilties(n_tokens)
    assert len(token_probabilities) == n_tokens
    np.testing.assert_almost_equal(np.sum(list(token_probabilities.values())), 1.0)
