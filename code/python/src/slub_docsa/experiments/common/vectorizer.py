"""Cached vectorizer for various experiments."""

import os

from typing import Optional

from slub_docsa.data.preprocess.vectorizer import TfidfStemmingVectorizer, CachedVectorizer, PersistedCachedVectorizer
from slub_docsa.data.preprocess.vectorizer import HuggingfaceBertVectorizer, TfidfVectorizer
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
