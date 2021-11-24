"""Generate random data with a hierarchical subject relationship.

This module allows to generate documents from artificial hierarchical subjects. Subject hierarchies are designed such
that child subjects share a random subset of tokens with its parent subject. Therefore, sibling subjects may also
share some tokens, but are likely to also contain different tokens. A parent subject however always contains all
tokens of all of its child subjects, and potentially more.

Of course, in real world data sets, there might be different token correlations between child and parent subjects.
However, for first experiments, this simplified design was implemented.

Artificial hierarchical random data allows to:
- Test respective hierarchical methods (hierarchical models, scores, etc.) without using large-scale real word data sets
- Investigate whether models are able to utilize this particular type of correlations between hierarchical subjects
"""

# pylint: disable=too-many-locals

import logging
import math

from typing import List, Mapping, Optional, Tuple
import numpy as np

from slub_docsa.common.dataset import Dataset, SimpleDataset
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
    children_per_level_interval: Tuple[int, int] = (2, 5),
    child_token_factor_interval: Tuple[float, float] = (0.5, 0.9),
) -> Tuple[Mapping[str, TokenProbabilities], SubjectHierarchyType]:
    """Generate a number of hierarchical subjects each represented by random token probabilities.

    Subjects are generated incremetally, i.e., starting with the root subject a number of child subjects are generated.
    Each child subject consists of a random selection of tokens and their occurance probabilities from the parent
    subject. If more subjects are needed, further child subjects are generated, using previous child subjects are
    parent subjects.

    Parameters
    ----------
    n_subjects: int
        the number of subjects to generate
    root_token_probabilities: TokenProbabilities
        the base token probabilities used to infer token probabilities
    label_from_tokens: bool = True
        if false, use a generic label (e.g. "subject123"), otherwise draw a random selection of tokens and use those
        as a label (useful when evaluating lexicographic approaches, e.g., the Annif yake model)
    children_per_level_interval: Tuple[int, int] = (2, 5)
        an interval (min, max+1) describing how many child subjects are generated from a parent subject. The exact
        number of child subjects is drawn randomly from this interval each time children are generated for a parent.
    child_token_factor_interval: Tuple[float, float] = (0.5, 0.9)
        an interval (min factor, max factor) describing a factor that defines how many tokens are drawn from a parent
        subject to be used for a child subject (0.5 means half of the number of tokens a parent subject consists of,
        1.0 means all tokens of the parent subject). The factor is drawn randomly from the provided interval.

    Returns
    -------
    Tuple[Mapping[str, TokenProbabilities], SubjectHierarchyType]
        A tuple containing the token probabilities for each subject (first element) as well as the corresponding
        subject hierarchy (second element)
    """
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
            low=children_per_level_interval[0],
            high=children_per_level_interval[1],
            size=1
        )[0]

        for _ in range(n_children):
            n_child_tokens = np.random.default_rng().integers(
                low=math.floor(len(current_token_probabilties) * child_token_factor_interval[0]),
                high=math.ceil(len(current_token_probabilties) * child_token_factor_interval[1]),
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
    n_subjects_per_document_interval: Tuple[int, int] = (1, 4),
    n_title_tokens_interval: Tuple[int, int] = (10, 50),
) -> Tuple[Dataset, SubjectHierarchyType]:
    """Generate a random hierarchical dataset based on given token probabilities.

    Parameters
    ----------
    token_probabilities: TokenProbabilities
        the base token probabilities which will be sampled from to generate child subjects according to
        `generate_hierarchical_subject_token_probabilities`
    n_documents: int
        the total number of documents to generate from all subjects. Each document belongs to one or multiple randomly
        selected subjects.
    n_subjects: int
        the total number of subjects to generate
    n_subjects_per_document_interval: Tuple[int, int] = (1, 4)
        an interval (min, max+1) describing the number of subjects each document is generated from. The exact number is
        drawn randomly from this interval for each document.
    n_title_tokens_interval: Tuple[int, int] = (10, 50)
        an interval (min, max+1) describing how many tokens are used to generate a title for each document. The exact
        number is drawn randomly from this interval for each document.

    Returns
    -------
    Tuple[Dataset, SubjectHierarchyType]
        a tuple containing the dataset of artificial documents and subject annotations, as well as the subject hierarchy
    """
    logger.debug("generate hierarchical dataset with %d documents", n_documents)
    subject_token_probabilities, subject_hierarchy = generate_hierarchical_subject_token_probabilities(
        n_subjects,
        token_probabilities
    )

    subject_uri_list = list(subject_token_probabilities.keys())
    subject_probabilities = np.random.default_rng().exponential(size=len(subject_uri_list))
    subject_probabilities = subject_probabilities / np.sum(subject_probabilities)
    n_subjects_per_document = np.random.default_rng().integers(
        low=n_subjects_per_document_interval[0],
        high=n_subjects_per_document_interval[1],
        size=n_documents
    )
    n_tokens_per_document = np.random.default_rng().integers(
        low=n_title_tokens_interval[0],
        high=n_title_tokens_interval[1],
        size=n_documents
    )

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

    return SimpleDataset(documents=documents, subjects=subject_targets), subject_hierarchy


def generate_hierarchical_random_dataset(
    n_tokens: int,
    n_documents: int,
    n_subjects: int,
) -> Tuple[Dataset, SubjectHierarchyType]:
    """Generate a random hierarchical dataset based on artificially generated tokens.

    A wrapper around `generate_hierarchical_random_dataset_from_token_probabilities` using random tokens from
    `slub_docsa.data.artificial.tokens.generate_random_token_probabilties`.

    Parameters
    ----------
    n_tokens: int
        the number of random tokens to generate
    n_documents: int
        the number of random documents to generate
    n_subjects: int
        the number of random subjects to generate

    Returns
    -------
    Tuple[Dataset, SubjectHierarchyType]
        Both the generated dataset and subject hierarchy
    """
    token_probabilities = generate_random_token_probabilties(n_tokens)
    return generate_hierarchical_random_dataset_from_token_probabilities(
        token_probabilities,
        n_documents,
        n_subjects,
    )


def generate_hierarchical_random_dataset_from_dbpedia(
    lang_code: str,
    n_tokens: int,
    n_documents: int,
    n_subjects: int,
) -> Tuple[Dataset, SubjectHierarchyType]:
    """Generate a random hierarchical dataset based on token probabilities extracted from DBpedia abstracts.

    A wrapper around `generate_hierarchical_random_dataset_from_token_probabilities` using tokens extracted from
    Dbpedia via `slub_docsa.data.artificial.tokens.token_probabilities_from_dbpedia`.

    Parameters
    ----------
    lang_code: str
        the language code of the dbpedia resources, which are used to extract token probabilities
    n_tokens: int
        the number of tokens to extract and use
    n_documents: int
        the number of documents to generate
    n_subjects: int
        the number of subjects to generate

    Returns
    -------
    Tuple[Dataset, SubjectHierarchyType]
        Both the generated dataset and subject hierarchy
    """
    token_probabilities = token_probabilities_from_dbpedia(lang_code)
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
