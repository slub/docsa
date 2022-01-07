"""Specialized vectorizer for the Qucosa datasets."""

import os

from slub_docsa.data.preprocess.vectorizer import TfidfStemmingVectorizer, CachedVectorizer, PersistedCachedVectorizer
from slub_docsa.data.preprocess.vectorizer import HuggingfaceBertVectorizer, TfidfVectorizer
from slub_docsa.common.paths import get_cache_dir


def get_qucosa_dbmdz_bert_vectorizer(subtext_samples: int = 1, hidden_states: int = 1):
    """Load persisted dbmdz bert vectorizer."""
    vectorizer_cache_dir = os.path.join(get_cache_dir(), "vectorizer")
    os.makedirs(vectorizer_cache_dir, exist_ok=True)
    filename = f"dbmdz_bert_qucosa_sts={subtext_samples}_hs={hidden_states}.sqlite"
    dbmdz_bert_cache_fp = os.path.join(vectorizer_cache_dir, filename)
    return PersistedCachedVectorizer(dbmdz_bert_cache_fp, HuggingfaceBertVectorizer(subtext_samples=subtext_samples))


def get_qucosa_tfidf_vectorizer(max_features: int = 10000, cache_vectors=False, fit_only_once: bool = False):
    """Load the tfidf vectorizer without stemming that optionally caches vectorizations."""
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),
    )
    if not cache_vectors:
        return tfidf_vectorizer

    return CachedVectorizer(tfidf_vectorizer, fit_only_once=fit_only_once)


def get_qucosa_tfidf_stemming_vectorizer(max_features: int = 10000, cache_vectors=False, fit_only_once: bool = False):
    """Load the tfidf stemming vectorizer that persists stemmed texts for caching."""
    stemming_cache_filepath = os.path.join(get_cache_dir(), "stemming/global_cache.sqlite")

    tfidf_vectorizer = TfidfStemmingVectorizer(
        lang_code="de",
        max_features=max_features,
        stemming_cache_filepath=stemming_cache_filepath,
        ngram_range=(1, 1),
    )
    if not cache_vectors:
        return tfidf_vectorizer

    return CachedVectorizer(tfidf_vectorizer, fit_only_once=fit_only_once)
