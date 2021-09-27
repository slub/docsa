"""Generate random data with a hierarchical subject relationship."""

# pylint: disable=too-many-locals

import logging
import math

from typing import List, Mapping, Optional, Tuple
import numpy as np

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectHierarchyType, SubjectNode, print_subject_hierarchy
from slub_docsa.data.artificial.simple import generate_random_text
from slub_docsa.data.artificial.tokens import TokenProbabilities, choose_tokens_by_probabilities
from slub_docsa.data.artificial.tokens import generate_random_token_probabilties, combine_token_probabilities
from slub_docsa.data.artificial.tokens import token_probabilities_from_dbpedia


logger = logging.getLogger(__name__)


def generate_hierarchical_subject_token_probabilities(
    n_subjects: int,
    root_token_probabilities: TokenProbabilities,
    label_from_tokens: bool = True,
) -> Tuple[Mapping[str, TokenProbabilities], SubjectHierarchyType]:
    """Generate a number of hierarchical subjects each represented by random token probabilities."""
    subject_token_probabilities = {}
    subject_hierarchy = {}
    parent_backlog: List[Tuple[Optional[str], int, TokenProbabilities]] = [
        (None, 0, root_token_probabilities)
    ]
    i = 0
    while len(subject_token_probabilities) < n_subjects:
        # select new subject as parent
        parent_subject_uri, current_level, current_token_probabilties = parent_backlog.pop(0)
        current_subject_uri = None

        if current_level > 0:
            # remember parent for document generation
            current_subject_uri = "uri://random/subject/" + str(i)
            if label_from_tokens:
                tokens = list(current_token_probabilties.keys())
                probabilities = [current_token_probabilties[t] for t in tokens]
                current_subject_label = " ".join(np.random.choice(tokens, size=5, p=probabilities, replace=False))
            else:
                current_subject_label = "subject " + str(i)
            subject_token_probabilities[current_subject_uri] = current_token_probabilties
            subject_hierarchy[current_subject_uri] = SubjectNode(
                uri=current_subject_uri,
                label=current_subject_label,
                parent_uri=parent_subject_uri
            )

        # create children
        n_children = np.random.default_rng().integers(
            low=2,
            high=5,
            size=1
        )[0]

        for _ in range(n_children):
            n_child_tokens = np.random.default_rng().integers(
                low=math.floor(len(current_token_probabilties) * 0.5),
                high=math.ceil(len(current_token_probabilties) * 0.9),
                size=1
            )[0]
            child_token_probabilities = choose_tokens_by_probabilities(n_child_tokens, current_token_probabilties)
            parent_backlog.append((current_subject_uri, current_level + 1, child_token_probabilities))

        i += 1

    return subject_token_probabilities, subject_hierarchy


def generate_hierarchical_random_dataset_from_token_probabilities(
    token_probabilities: TokenProbabilities,
    n_documents: int,
    n_subjects: int,
) -> Tuple[Dataset, SubjectHierarchyType]:
    """Generate a random hierarchical dataset based on token probabilities."""
    logger.debug("generate hierarchical dataset with %d documents", n_documents)
    subject_token_probabilities, subject_hierarchy = generate_hierarchical_subject_token_probabilities(
        n_subjects,
        token_probabilities
    )

    subject_uri_list = list(subject_token_probabilities.keys())
    subject_probabilities = np.random.default_rng().exponential(size=len(subject_uri_list))
    subject_probabilities = subject_probabilities / np.sum(subject_probabilities)
    n_subjects_per_document = np.random.default_rng().integers(low=1, high=4, size=n_documents)
    n_tokens_per_document = np.random.default_rng().integers(low=10, high=50, size=n_documents)

    documents = []
    subject_targets = []
    for i in range(n_documents):
        chosen_subjects = list(np.random.choice(
            subject_uri_list, size=n_subjects_per_document[i], p=subject_probabilities
        ))
        chosen_subject_token_probabilities = [subject_token_probabilities[uri] for uri in chosen_subjects]
        combined_token_probabilities = combine_token_probabilities(chosen_subject_token_probabilities)

        token_list = list(combined_token_probabilities.keys())
        token_probabilty_list = [combined_token_probabilities[t] for t in token_list]

        document = generate_random_text(n_tokens_per_document[i], token_list, token_probabilty_list)
        document_uri = "uri://random/document/" + str(i)
        documents.append(Document(uri=document_uri, title=document))
        subject_targets.append(chosen_subjects)

    return Dataset(documents=documents, subjects=subject_targets), subject_hierarchy


def generate_hierarchical_random_dataset(
    n_tokens: int,
    n_documents: int,
    n_subjects: int,
):
    """Generate a random hierarchical dataset based on artificially generated tokens."""
    token_probabilities = generate_random_token_probabilties(n_tokens)
    return generate_hierarchical_random_dataset_from_token_probabilities(
        token_probabilities,
        n_documents,
        n_subjects,
    )


def generate_hierarchical_random_dataset_from_dbpedia(
    language: str,
    n_tokens: int,
    n_documents: int,
    n_subjects: int,
):
    """Generate a random hierarchical dataset based on token probabilities extracted from DBpedia abstracts."""
    token_probabilities = token_probabilities_from_dbpedia(language)
    token_probabilities = choose_tokens_by_probabilities(n_tokens, token_probabilities)
    return generate_hierarchical_random_dataset_from_token_probabilities(
        token_probabilities,
        n_documents,
        n_subjects,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_token_probabilities = generate_random_token_probabilties(1000)
    subject_tp, subject_h = generate_hierarchical_subject_token_probabilities(10, test_token_probabilities)
    print_subject_hierarchy(subject_h)
