"""Methods to generate random token distributions or extract them from existing corpora."""

import re
import string
import logging

from typing import Iterable, Mapping

import numpy as np

from slub_docsa.data.load.dbpedia import read_dbpedia_abstracts

logger = logging.getLogger(__name__)

TokenProbabilities = Mapping[str, float]
"""A mapping describing the occurance probability for each available token.
The sum of all probabilities should equal 1.
"""


def generate_random_token_probabilties(n_tokens: int, token_length=10, exp_scale=1.0) -> TokenProbabilities:
    """Generate random tokens of random lowercase letters with probabilities from an exponential distribution.

    Parameters
    ----------
    n_tokens: int
        The number of random tokens to generate.
    token_length: int = 10
        The size of the tokens as the number of random letters.
    exp_scale: float = 1.0
        The scale parameter of the exponential distribution used to draw token probabilities.

    Returns
    -------
    TokenProbabilities
        A map of random tokens and their random probabilities

    Examples
    --------
    Generate 3 random tokens of 5 random letters with random probabilities:

    >>> generate_random_token_probabilties(3, token_length=5)
    {'hghby': 0.04076535563169729, 'kncyv': 0.6835635904749359, 'aovqt': 0.275671053893367}
    """
    tokens: TokenProbabilities = {}
    while len(tokens) < n_tokens:
        n_tokens_left = n_tokens - len(tokens)
        token_list = np.random.choice(list(string.ascii_lowercase), size=(n_tokens_left, token_length))
        probabilities = np.random.default_rng().exponential(exp_scale, n_tokens_left)
        tokens.update({"".join(t): p for t, p in zip(token_list, probabilities)})
    probability_sum = np.sum(list(tokens.values()))
    return {t: p / probability_sum for t, p in tokens.items()}


def token_probabilities_from_corpus(corpus: Iterable[str]) -> TokenProbabilities:
    """Extract token probabilities by counting tokens in an existing corpus.

    Parameters
    ----------
    corpus: Iterable[str]
        An iterable of strings containing texts from some corpus.

    Returns
    -------
    TokenProbabilities
        The occurance probabilities of each lowercased token of the provided corpus.

    Examples
    --------
    Count token occurances of arbitrary text:

    >>> token_probabilities_from_corpus(["a corpus with", "tokens tokens"])
    {'a': 0.2, 'corpus': 0.2, 'with': 0.2, 'tokens': 0.4}
    """
    token_pattern = re.compile(r"^[a-z0-9]+$")
    token_counts = {}

    for document in corpus:
        for token in document.split(" "):
            token_normalized = token.strip().lower()
            if token_normalized in token_counts:
                token_counts[token_normalized] += 1
            elif token_normalized and token_pattern.match(token_normalized):
                token_counts[token_normalized] = 1

    count_sum = sum(token_counts.values())

    return {t: c / count_sum for t, c in token_counts.items()}


def token_probabilities_from_dbpedia(lang_code: str, n_docs=10000) -> TokenProbabilities:
    """Return token probabilities by counting tokens in DBpedia abstracts.

    Parameters
    ----------
    lang_code: str
        the language code of dbpedia resources, which are used to extract token probabilities
    n_docs: int = 10000
        the maximum number of dbpedia documents to process and collect tokens from

    Returns
    -------
    TokenProbabilities
        the token probabilities as extracted from Dbpedia
    """
    logger.debug("extract token probabilties from %s dbpedia abstracts", lang_code)
    corpus = read_dbpedia_abstracts(lang_code, limit=n_docs)
    return token_probabilities_from_corpus(corpus)


def choose_tokens_by_probabilities(
    k: int,
    token_probabilities: TokenProbabilities,
) -> TokenProbabilities:
    """Select `k` tokens according to their probability and return a subset of token probabilities.

    The resulting token probabilities are normalized again to achieve a sum of 1.

    Parameters
    ----------
    k: int
        the number of tokens to randomly select from `token_probabilities`
    token_probabilities: TokenProbabilities
        the based token probabilities from which tokens are sampled

    Returns
    -------
    TokenProbabilities
        a subset of `k` tokens and their occurance probabilities normalized to 1
    """
    tokens = list(token_probabilities.keys())
    probabilities = list(map(lambda t: token_probabilities[t], tokens))
    tokens_idx = list(range(len(tokens)))

    # do selection
    chosen_token_idx = np.random.choice(
        tokens_idx,
        size=min(k, len(tokens)),
        replace=False,
        p=probabilities
    )

    # build new dictionary
    chosen_tokens = [tokens[i] for i in chosen_token_idx]
    chosen_probabilities = [probabilities[i] for i in chosen_token_idx]
    chosen_total_sum = sum(chosen_probabilities)
    return {t: p / chosen_total_sum for t, p in zip(chosen_tokens, chosen_probabilities)}


def combine_token_probabilities(
    token_probabilities_list: Iterable[TokenProbabilities]
) -> TokenProbabilities:
    """Add token probabilities from a list of token probabilities.

    Normalizes the union of all token probabilities to a sum of 1.

    Parameters
    ----------
    token_probabilities_list: Iterable[TokenProbabilities]
        the list of token probablities that will be combined into one

    Returns
    -------
    TokenProbabilities
        the combined token probablities
    """
    new_token_probabilities = {}
    for token_probabilities in token_probabilities_list:
        for token, probability in token_probabilities.items():
            new_token_probabilities[token] = new_token_probabilities.get(token, 0) + probability

    new_probability_sum = np.sum(list(new_token_probabilities.values()))
    return {t: p / new_probability_sum for t, p in new_token_probabilities.items()}
