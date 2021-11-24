"""Methods to generate simple artificial random data.

This module provides various methods that allow to generate artificial random documents and their subject annotations
based on trivial hypotheses, e.g., that there is no correlation between the tokens of a document and its subject
annotations (fully random), or that there is a correlation based on token probabilities.

As a consequence, these datasets can be used to validate various properties of different models, e.g.:
- a model should not outperform random predictions if the data was generated fully randomly and does not contain any
  correlations between documents and their subject annotations
- a model should outperform random predictions if there are simple correlations based on token occurance probabilities
"""

# pylint: disable=too-many-locals

import logging

from typing import Mapping, Sequence, Tuple

import numpy as np

from slub_docsa.common.dataset import Dataset, SimpleDataset
from slub_docsa.common.document import Document
from slub_docsa.data.artificial.tokens import TokenProbabilities, generate_random_token_probabilties
from slub_docsa.data.artificial.tokens import token_probabilities_from_dbpedia

logger = logging.getLogger(__name__)


def generate_random_text(n_tokens: int, tokens: Sequence[str], probabilities: Sequence[float]) -> str:
    """Return random text with tokens chosen based on token probabilities.

    Parameters
    ----------
    n_tokens: int
        the length of the text in number of tokens
    tokens: Sequence[str]
        the list of tokens as sequence, list ordering must match `probabilities` parameter
    probabilities: Sequence[float]
        the occurance probabilities for each token, list ordering must match `tokens` parameter

    Returns
    -------
    str
        The random text that was generated
    """
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
    n_subjects: int,
    n_subjects_per_document_interval: Tuple[int, int] = (1, 4),
    n_title_tokens_interval: Tuple[int, int] = (5, 20),
) -> Dataset:
    """Return a random dataset by generating random documents according to token probabilties.

    However, documents are assigned randomly to subjects. Therefore, there is not correlation between subjects and
    documents, and thus, no model should be able to learn any relationship and outperform random predictions.

    Parameters
    ----------
    token_probabilities: TokenProbabilities
        base tokens and their probabilities used to describe subjects
    n_docs: int
        the number of documents to generate
    n_subjects: int
        the number of subjects to generate
    n_subjects_per_document_interval: Tuple[int, int] = (1, 4)
        an interval (min, max+1) describing the number of subjects each document is generated from. The exact number is
        drawn randomly from this interval for each document.
    n_title_tokens_interval: Tuple[int, int] = (5, 20)
        an interval (min, max+1) describing how many tokens are used to generate a title for each document. The exact
        number is drawn randomly from this interval for each document.

    Returns
    -------
    Dataset
        the dataset containing all generated documents and their corresponding subject annotations
    """
    token_list = list(token_probabilties.keys())
    token_probabilty_list = list(map(lambda t: token_probabilties[t], token_list))

    title_length_array = np.random.default_rng().integers(
        low=n_title_tokens_interval[0],
        high=n_title_tokens_interval[1],
        size=n_docs
    )

    subject_count_array = np.random.default_rng().integers(
        low=n_subjects_per_document_interval[0],
        high=n_subjects_per_document_interval[1],
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

    return SimpleDataset(documents=documents, subjects=subject_targets)


def generate_easy_random_dataset_from_token_probabilities(
    token_probabilties: TokenProbabilities,
    n_docs: int,
    n_subjects: int
) -> Dataset:
    """Return a random dataset that is very easy to predict because subjects area assigned a unique vocabulary.

    Each token of `token_probabilities` is assigned to one subject. Documents are generated from these non-overlapping
    token sets. Therefore, predicting a document's subject is as simple as creating a reverse lookup table from tokens
    to subjects. Most models should achieve a high performance for this data.

    Parameters
    ----------
    token_probabilities: TokenProbabilities
        base token occurance probabilities that are divided into distinct subsets
    n_docs: int
        the number of documents to generate
    n_subjects: int
        the number of subjects to generate

    Returns
    -------
    Dataset
        the dataset containing all generated documents and their corresponding subject annotations
    """
    if n_subjects > len(token_probabilties):
        raise ValueError("can not assign tokens to subjects if there are more subjects than tokens")

    token_list = list(token_probabilties.keys())
    subject_list = [f"uri://random/subject/{i}" for i in range(n_subjects)]

    # assign tokens to subjects
    token_to_subject_index = np.random.default_rng().integers(
        low=0,
        high=n_subjects,
        size=len(token_probabilties)
    )

    # check that every subject was chosen at least once
    subject_appear_once = False
    while not subject_appear_once:
        subject_appear_once = True
        for i in range(n_subjects):
            if i not in token_to_subject_index:
                # pick random position
                token_id = np.random.default_rng().integers(low=0, high=len(token_probabilties), size=1)[0]
                token_to_subject_index[token_id] = i
                subject_appear_once = False

    # collect tokens and their probabilities that belong to each subject
    token_list_by_subject: Mapping[int, Sequence[str]] = {}
    token_probabilities_list_by_subject: Mapping[int, Sequence[float]] = {}
    for i in range(n_subjects):
        token_list_by_subject[i] = [t for j, t in enumerate(token_list) if token_to_subject_index[j] == i]
        if len(token_list_by_subject[i]) == 0:
            raise RuntimeError("subject has no tokens, this should not happen")

        probabilities_list = list(map(lambda t: token_probabilties[t], token_list_by_subject[i]))
        token_probabilities_list_by_subject[i] = list(np.array(probabilities_list) / np.sum(probabilities_list))

    title_length_array = np.random.default_rng().integers(
        low=5,
        high=20,
        size=n_docs
    )

    documents = []
    subject_targets = []

    for i in range(n_docs):

        subject_id = np.random.default_rng().integers(low=0, high=n_subjects, size=1)[0]
        doc_uri = f"uri://random/document/{i}"
        doc_title = generate_random_text(
            title_length_array[i],
            token_list_by_subject[subject_id],
            token_probabilities_list_by_subject[subject_id]
        )
        documents.append(Document(uri=doc_uri, title=doc_title))
        subject_targets.append([subject_list[subject_id]])

    return SimpleDataset(documents=documents, subjects=subject_targets)


def generate_random_dataset(n_tokens: int, n_docs: int, n_subjects: int) -> Dataset:
    """Generate random uncorrelated dataset with tokens from exponential distribution.

    Wrapper around `generate_random_dataset_from_token_probabilities` with token probabilities from
    `generate_random_token_probabilties`.

    Parameters
    ----------
    n_tokens: int
        The number of tokens to generate
    n_docs: int
        The number of documents to generate
    n_subjects: int
        The number of subjects to generate

    Returns
    -------
    Dataset
        the dataset containing all generated documents and their corresponding subject annotations
    """
    token_probabilities = generate_random_token_probabilties(n_tokens)
    return generate_random_dataset_from_token_probabilities(token_probabilities, n_docs, n_subjects)


def generate_easy_random_dataset(n_tokens: int, n_docs: int, n_subjects: int):
    """Generate random dataset that is easy to predict with tokens from exponential distribution.

    Wrapper around `generate_easy_random_dataset_from_token_probabilities` with token probabilities from
    `generate_random_token_probabilties`.

    Parameters
    ----------
    n_tokens: int
        The number of tokens to generate
    n_docs: int
        The number of documents to generate
    n_subjects: int
        The number of subjects to generate

    Returns
    -------
    Dataset
        the dataset containing all generated documents and their corresponding subject annotations
    """
    token_probabilities = generate_random_token_probabilties(n_tokens)
    return generate_easy_random_dataset_from_token_probabilities(token_probabilities, n_docs, n_subjects)


def generate_random_dataset_from_dbpedia(lang_code: str, n_docs: int, n_subjects: int):
    """Generate random uncorrelated dataset with tokens from DBpedia.

    Wrapper around `generate_random_dataset_from_token_probabilities` with token probabilities from
    `token_probabilities_from_dbpedia`.

    Parameters
    ----------
    lang_code: str
        The language code of dbpedia resources that are used to extract token probabilities
    n_docs: int
        The number of documents to generate
    n_subjects: int
        The number of subjects to generate

    Returns
    -------
    Dataset
        the dataset containing all generated documents and their corresponding subject annotations
    """
    token_probabilities = token_probabilities_from_dbpedia(lang_code)
    return generate_random_dataset_from_token_probabilities(token_probabilities, n_docs, n_subjects)


def generate_easy_random_dataset_from_dbpedia(lang_code: str, n_docs: int, n_subjects: int):
    """Generate random dataset that is easy to predict from with tokens from DBpedia.

    Wrapper around `generate_easy_random_dataset_from_token_probabilities` with token probabilities from
    `token_probabilities_from_dbpedia`.

    Parameters
    ----------
    lang_code: str
        The language code of dbpedia resources that are used to extract token probabilities
    n_docs: int
        The number of documents to generate
    n_subjects: int
        The number of subjects to generate

    Returns
    -------
    Dataset
        the dataset containing all generated documents and their corresponding subject annotations
    """
    token_probabilities = token_probabilities_from_dbpedia(lang_code)
    return generate_easy_random_dataset_from_token_probabilities(token_probabilities, n_docs, n_subjects)


def get_static_mini_dataset() -> Dataset:
    """Return a collection of 10 documents and their target subjects of 4 different subjects.

    Returns
    -------
    Dataset
        the static dataset consisting of 10 documents and 4 subjects
    """
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

    return SimpleDataset(
        documents=documents,
        subjects=subjects
    )
