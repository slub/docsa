"""Tests simple data generation methods."""

import itertools
from typing import Mapping, Set

import pytest

from slub_docsa.data.artificial.simple import generate_easy_random_dataset
from slub_docsa.data.artificial.simple import generate_easy_random_dataset_from_token_probabilities
from slub_docsa.data.artificial.simple import generate_easy_random_dataset_from_dbpedia
from slub_docsa.data.artificial.tokens import TokenProbabilities
from slub_docsa.evaluation.incidence import unique_subject_order


def _check_dataset_documents_do_not_share_tokens(dataset):
    """Verify that documents of different subjects do not share a single token."""
    unique_subjects_list = unique_subject_order(dataset.subjects)
    tokens_by_subject: Mapping[str, Set[str]] = {}
    for s_uri in unique_subjects_list:
        docs_list = [d for d, s in zip(dataset.documents, dataset.subjects) if s_uri in s]
        tokens_by_subject[s_uri] = {t for d in docs_list for t in d.title.split(" ")}

    for s_1, s_2 in itertools.combinations(unique_subjects_list, 2):
        assert len(tokens_by_subject[s_1].intersection(tokens_by_subject[s_2])) == 0


def test_generate_easy_random_dataset_from_token_probabilities():
    """Test easy random dataset and verify that documents do not share any tokens."""
    n_subjects = 3
    n_docs = 5

    token_probabilities: TokenProbabilities = {
        "one": 0.1,
        "two": 0.3,
        "three": 0.1,
        "four": 0.2,
        "five": 0.1,
        "six": 0.1,
        "seven": 0.1,
    }

    dataset = generate_easy_random_dataset_from_token_probabilities(
        token_probabilities,
        n_docs=n_docs,
        n_subjects=n_subjects
    )

    _check_dataset_documents_do_not_share_tokens(dataset)


def test_generate_easy_random_dataset():
    """Test easy random dataset and verify that documents do not share any tokens."""
    _check_dataset_documents_do_not_share_tokens(
        generate_easy_random_dataset(100, 50, 10)
    )


@pytest.mark.skip(reason="requires large download of dbpedia")
def test_generate_easy_random_dataset_from_dbpedia():
    """Test easy random dataset and verify that documents do not share any tokens."""
    _check_dataset_documents_do_not_share_tokens(
        generate_easy_random_dataset_from_dbpedia("en", 100, 10)
    )
