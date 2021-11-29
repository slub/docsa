"""Specialized vectorizer for the Qucosa datasets."""

import os

from slub_docsa.data.preprocess.vectorizer import TfidfStemmingVectorizer, CachedVectorizer, PersistedCachedVectorizer
from slub_docsa.data.preprocess.vectorizer import HuggingfaceBertVectorizer
from slub_docsa.common.paths import CACHE_DIR

VECTORIZATION_CACHE = os.path.join(CACHE_DIR, "vectorizer")


def get_qucosa_dbmdz_bert_vectorizer(subtext_samples: int = 1, hidden_states: int = 1):
    """Load persisted dbmdz bert vectorizer."""
    os.makedirs(VECTORIZATION_CACHE, exist_ok=True)
    filename = f"dbmdz_bert_qucosa_sts={subtext_samples}_hs={hidden_states}.sqlite"
    dbmdz_bert_cache_fp = os.path.join(VECTORIZATION_CACHE, filename)
    return PersistedCachedVectorizer(dbmdz_bert_cache_fp, HuggingfaceBertVectorizer(subtext_samples=subtext_samples))


def get_qucosa_tfidf_stemming_vectorizer(max_features: int = 10000, cache_vectors=False, fit_only_once: bool = False):
    """Load the tfidf stemming vectorizer that persists stemmed texts for caching."""
    stemming_cache_filepath = os.path.join(CACHE_DIR, "stemming/global_cache.sqlite")

    tfidf_vectorizer = TfidfStemmingVectorizer(
        lang_code="de",
        max_features=max_features,
        stemming_cache_filepath=stemming_cache_filepath,
        ngram_range=(1, 1),
    )
    if not cache_vectors:
        return tfidf_vectorizer

    return CachedVectorizer(tfidf_vectorizer, fit_only_once=fit_only_once)
