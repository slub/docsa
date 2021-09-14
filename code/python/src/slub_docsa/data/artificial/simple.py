"""Methods to generate simple artificial data."""

import logging

from typing import Sequence

import numpy as np

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.document import Document
from slub_docsa.data.artificial.tokens import TokenProbabilities, generate_random_token_probabilties
from slub_docsa.data.artificial.tokens import token_probabilities_from_dbpedia

logger = logging.getLogger(__name__)


def generate_random_text(n_tokens: int, tokens: Sequence[str], probabilities: Sequence[float]) -> str:
    """Return random text with tokens chosen based on token probabilities."""
    token_list = np.random.choice(
        tokens,
        size=n_tokens,
        p=probabilities,
        replace=True
    )
    return " ".join(token_list)


def generate_random_dataset_from_token_probabilities(
    token_probabilties: TokenProbabilities,
    n_docs: int,
    n_subjects: int
):
    """Return a random dataset by generating random documents according to token probabilties."""
    token_list = list(token_probabilties.keys())
    token_probabilty_list = list(map(lambda t: token_probabilties[t], token_list))

    title_length_array = np.random.default_rng().integers(
        low=5,
        high=20,
        size=n_docs
    )

    subject_count_array = np.random.default_rng().integers(
        low=1,
        high=4,
        size=n_docs
    )

    subject_list = [f"uri://random/subject/{i}" for i in range(n_subjects)]

    documents = []
    subject_targets = []

    for i in range(n_docs):

        doc_uri = f"uri://random/document/{i}"
        doc_title = generate_random_text(title_length_array[i], token_list, token_probabilty_list)
        documents.append(Document(uri=doc_uri, title=doc_title))

        random_subjects = np.random.choice(
            subject_list,
            size=subject_count_array[i],
            replace=False
        ).tolist()

        subject_targets.append(random_subjects)

    return Dataset(documents=documents, subjects=subject_targets)


def generate_random_dataset(n_tokens: int, n_docs: int, n_subjects: int):
    """Generate random dataset with tokens from exponential distribution."""
    token_probabilities = generate_random_token_probabilties(n_tokens)
    return generate_random_dataset_from_token_probabilities(token_probabilities, n_docs, n_subjects)


def generate_random_dataset_from_dbpedia(language: str, n_docs: int, n_subjects: int):
    """Generate random dataset from with tokens from DBpedia."""
    token_probabilities = token_probabilities_from_dbpedia(language)
    return generate_random_dataset_from_token_probabilities(token_probabilities, n_docs, n_subjects)


def get_static_mini_dataset():
    """Return a collection of 5 documents and their target subjects."""
    documents = [
        Document(uri="uri://test_document1", title="This is a document"),
        Document(uri="uri://test_document2", title="Another document with interesting topics"),
        Document(uri="uri://test_document3", title="Document describing Magdeburg and its Landmarks"),
        Document(uri="uri://test_document4", title="History of Magdeburg "),
        Document(uri="uri://test_document5", title="History of Dresden"),
        Document(uri="uri://test_document6", title="Yet another document describing interseting things"),
        Document(uri="uri://test_document7", title="A fascinating example of a document"),
        Document(uri="uri://test_document8", title="Boring document title"),
        Document(uri="uri://test_document9", title="Towards writing good documents"),
        Document(uri="uri://test_document10", title="Artifical document generation")
    ]

    subjects = [
        ["uri://subject1", "uri://subject2"],
        ["uri://subject1", "uri://subject2"],
        ["uri://subject1", "uri://subject2"],
        ["uri://subject2", "uri://subject3"],
        ["uri://subject2", "uri://subject3"],
        ["uri://subject2", "uri://subject3"],
        ["uri://subject1", "uri://subject3"],
        ["uri://subject1", "uri://subject3"],
        ["uri://subject1", "uri://subject3"],
        ["uri://subject1", "uri://subject4"],
    ]

    return Dataset(
        documents=documents,
        subjects=subjects
    )
