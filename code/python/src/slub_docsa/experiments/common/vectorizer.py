"""Cached vectorizer for various experiments."""

# pylint: disable=too-many-arguments

import os

from typing import Optional

from slub_docsa.data.preprocess.vectorizer import TfidfStemmingVectorizer, CachedVectorizer, PersistedCachedVectorizer
from slub_docsa.data.preprocess.vectorizer import HuggingfaceBertVectorizer, TfidfVectorizer, WordpieceVectorizer
from slub_docsa.common.paths import get_cache_dir


def get_cached_dbmdz_bert_vectorizer(
        cache_dir: Optional[str] = None,
        subtext_samples: int = 1,
        hidden_states: int = 1,
):
    """Load persisted dbmdz bert vectorizer."""
    if cache_dir is None:
        cache_dir = os.path.join(get_cache_dir(), "vectorizer")
    print(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"dbmdz_bert_qucosa_sts={subtext_samples}_hs={hidden_states}.sqlite"
    dbmdz_bert_cache_fp = os.path.join(cache_dir, filename)
    return PersistedCachedVectorizer(dbmdz_bert_cache_fp, HuggingfaceBertVectorizer(subtext_samples=subtext_samples))


def get_cached_tfidf_vectorizer(
    max_features: int = 10000,
    fit_only_once: bool = False
):
    """Load the tfidf vectorizer without stemming that caches vectorizations."""
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),
    )

    return CachedVectorizer(tfidf_vectorizer, fit_only_once=fit_only_once)


def get_cached_tfidf_stemming_vectorizer(
    lang_code: str = "de",
    cache_filepath: Optional[str] = None,
    cache_prefix: str = "global",
    max_features: int = 10000,
    fit_only_once: bool = False
):
    """Load the tfidf stemming vectorizer that persists stemmed texts for caching."""
    if cache_filepath is None:
        cache_filepath = os.path.join(get_cache_dir(), f"stemming/{cache_prefix}_features={max_features}.sqlite")

    tfidf_vectorizer = TfidfStemmingVectorizer(
        lang_code=lang_code,
        max_features=max_features,
        stemming_cache_filepath=cache_filepath,
        ngram_range=(1, 1),
    )

    return CachedVectorizer(tfidf_vectorizer, fit_only_once=fit_only_once)


def get_static_wikipedia_wordpiece_vectorizer(
    lang_code: str,
    max_length: int = 32,
    vocabulary_size: int = 10000,
    uncased: bool = True,
    limit: int = None,
    cache_directory: str = None,
):
    """Load or generate a static Wordpiece vectorizer solely based on Wikipedia texts."""
    if cache_directory is None:
        props = "_".join(
            [lang_code, f"vs{vocabulary_size}"]
            + ["uncased" if uncased else "cased"]
            + ([f"l{limit}"] if limit is not None else [])
        )
        cache_directory = os.path.join(get_cache_dir(), f"wordpiece/wikpedia_{props}")

    if not os.path.exists(cache_directory):
        vectorizer = WordpieceVectorizer(
            lang_code,
            vocabulary_size=vocabulary_size,
            max_length=max_length,
            uncased=uncased,
            wikipedia_texts_limit=limit
        )
        vectorizer.fit(iter([]))
        vectorizer.save(cache_directory)

    vectorizer = WordpieceVectorizer(
        lang_code, vocabulary_size=vocabulary_size, max_length=max_length, uncased=uncased, ignore_fit=True
    )
    vectorizer.load(cache_directory)
    return vectorizer
