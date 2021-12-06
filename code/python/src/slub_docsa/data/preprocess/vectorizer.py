"""Methods to vectorize text."""

# pylint: disable=too-many-instance-attributes, too-many-arguments

import logging
import os
import pickle  # nosec

from typing import Iterator, Optional, cast
from itertools import islice

import torch
import numpy as np

from sqlitedict import SqliteDict
from sklearn.feature_extraction.text import TfidfVectorizer as ScikitTfidfVectorizer
from torch.nn.modules.module import Module
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel

from slub_docsa.common.paths import CACHE_DIR
from slub_docsa.data.preprocess.document import nltk_snowball_text_stemming_function
from slub_docsa.data.preprocess.document import persisted_nltk_snowball_text_stemming_function
from slub_docsa.data.store.array import bytes_to_numpy_array, numpy_array_to_bytes
from slub_docsa.data.store.document import sha1_hash_from_text

logger = logging.getLogger(__name__)

HUGGINGFACE_CACHE_DIR = os.path.join(CACHE_DIR, "huggingface")


class AbstractVectorizer:
    """Represents a vectorizer model that can be fitted to a corpus."""

    def fit(self, texts: Iterator[str]):
        """Optionally train a vectorizer model using the provided texts.

        Parameters
        ----------
        texts: Iterator[str]
            the texts that are used to build a vectorization model

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def transform(self, texts: Iterator[str]) -> Iterator[np.ndarray]:
        """Return vector representation of texts as array.

        Parameters
        ----------
        texts: Iterator[str]
            the texts that are vectorized

        Returns
        -------
        Iterator[np.ndarray]
            an iterator over the vectorized texts as numpy array
        """
        raise NotImplementedError()


class PersistableVectorizer(AbstractVectorizer):
    """Extend a vectorizer to support save/load methods that can be used to persist it."""

    def load(self, persist_dir: str):
        """Load a persisted vectorizer state from disk.

        Parameters
        ----------
        persist_dir: str
            the path to the directory that the persisted vectorizer state is loaded from
        """
        raise NotImplementedError()

    def save(self, persist_dir: str):
        """Save the vectorizer state to disk.

        Parameters
        ----------
        persist_dir: str
            a path to a directory that is used to save the vectorizer state
        """
        raise NotImplementedError()


class TfidfVectorizer(PersistableVectorizer):
    """Vectorizer using Scikit TfidfVectorizer."""

    PERSIST_FILENAME = "tfidf_vectorizer.pickle"

    def __init__(self, max_features=10000, **kwargs):
        """Initialize vectorizer.

        Parameters
        ----------
        max_features: int = 10000
            The maximum number of unique tokens to extract from text during fit.
        kwargs: Any
            additional arguments that are passed to the scikit tfidf vectorizer
        """
        self.max_features = max_features
        self.fitted_once = False
        self.vectorizer = ScikitTfidfVectorizer(max_features=max_features, **kwargs)

    def fit(self, texts: Iterator[str]):
        """Fit vectorizer."""
        corpus = list(texts)
        logger.debug("do scikit tfidf vectorization of %d texts and %d features", len(corpus), self.max_features)
        self.vectorizer.fit(corpus)
        logger.debug("done with scikit tfidf vectorization")
        self.fitted_once = True
        # logger.debug("fitted tfidf vectorizer with vocabulary of %s", str(self.vectorizer.get_feature_names()))

    def transform(self, texts: Iterator[str]) -> Iterator[np.ndarray]:
        """Return vectorized texts."""
        for row in self.vectorizer.transform(list(texts)).toarray():
            if not np.any(row):
                # add random value if row is all zero, so cosine distance is well defined
                row[np.random.randint(0, row.shape[0], size=1)[0]] = np.random.random()
            yield row

    def save(self, persist_dir: str):
        """Save tfidf vectorizer to disc using pickle."""
        if not self.fitted_once:
            raise ValueError("can not persist tfidf vectorizer that was not fitted before")

        with open(os.path.join(persist_dir, self.PERSIST_FILENAME), "wb") as file:
            pickle.dump(self.vectorizer, file)

    def load(self, persist_dir: str):
        """Load tfidf vectorizer from disc using pickle."""
        if self.fitted_once:
            raise ValueError("can not load tfidf vectorizer that was previously fitted")

        vectorizer_path = os.path.join(persist_dir, self.PERSIST_FILENAME)

        if not os.path.exists(vectorizer_path):
            raise ValueError(f"vectorizer state does not exists at {vectorizer_path}")

        with open(vectorizer_path, "rb") as file:
            self.vectorizer = pickle.load(file)  # nosec

    def __str__(self):
        """Return representative string of vectorizer."""
        return f"<TfidfVectorizer max_features={self.max_features}>"


class TfidfStemmingVectorizer(TfidfVectorizer):
    """Apply nltk stemming and stopword removal before vectorizing with Scikit TfidfVectorizer."""

    def __init__(
        self,
        lang_code: str,
        remove_stopwords: bool = True,
        max_features=10000,
        stemming_cache_filepath: str = None,
        **kwargs
    ):
        """Initialize vectorizer.

        Parameters
        ----------
        lang_code: str
            Language code of the text (e.g. "en", "de")
        remove_stopwords: bool
            Whether to remove stopwords
        max_features: int = 10000
            The maximum number of unique tokens to extract from text during fit.
        stemming_cache_filepath: str = None
            Optional file path to a file that is used to persist stemmed text for caching
        kwargs: Any
            additional arguments that are passed to the scikit tfidf vectorizer
        """
        super().__init__(max_features=max_features, **kwargs)
        self.lang_code = lang_code
        self.remove_stopwords = remove_stopwords
        if stemming_cache_filepath is not None:
            self.stemming = persisted_nltk_snowball_text_stemming_function(
                stemming_cache_filepath,
                lang_code,
                remove_stopwords
            )
        else:
            self.stemming = nltk_snowball_text_stemming_function(lang_code, remove_stopwords)

    def fit(self, texts: Iterator[str]):
        """Fit vectorizer."""
        # logger.debug("do stemming for fitting of tfidf vectorizer")
        stemmed_texts = [self.stemming(t) for t in texts]
        super().fit(iter(stemmed_texts))

    def transform(self, texts: Iterator[str]) -> Iterator[np.ndarray]:
        """Return vectorized texts."""
        # logger.debug("do stemming for transforming with tfidf vectorizer")
        stemmed_texts = [self.stemming(t) for t in texts]
        yield from super().transform(iter(stemmed_texts))

    def __str__(self):
        """Return representative string of vectorizer."""
        return f"<TfidfStemmingVectorizer max_features={self.max_features} lang_code={self.lang_code} " + \
            f"remove_stopwords={self.remove_stopwords}>"


class RandomVectorizer(AbstractVectorizer):
    """A vectorizer returning random vectors."""

    def __init__(self, size: int = 3):
        """Initialize vectorizer.

        Parameters
        ----------
        size: int = 3
            The size of the returned random vectors
        """
        self.size = size

    def fit(self, texts: Iterator[str]):
        """Fit vectorizer."""

    def transform(self, texts: Iterator[str]) -> Iterator[np.ndarray]:
        """Return vectorized texts."""
        for row in np.random.random((len(list(texts)), self.size)):
            yield row

    def __str__(self):
        """Return representative string of vectorizer."""
        return "<RandomVectorizer>"


class CachedVectorizer(PersistableVectorizer):
    """Caches vectorizations for fast repeated transforms."""

    def __init__(self, vectorizer: AbstractVectorizer, batch_size: int = 1000, fit_only_once: bool = False):
        """Initialize vectorizer.

        Parameters
        ----------
        vectorizer: AbstractVectorizer
            The parent vectorizer used to vectorize texts in case texts can not be found in cache.
        batch_size: int
            The number of text to process in one batch
        fit_only_once: bool
            Skip additional fit calls after fitting once
        """
        self.vectorizer = vectorizer
        self.batch_size = batch_size
        self.fit_only_once = fit_only_once
        self.fitted_once = False
        self.store = {}

    def fit(self, texts: Iterator[str]):
        """Fit parent vectorizer."""
        if not self.fitted_once or not self.fit_only_once:
            self.vectorizer.fit(texts)
        self.fitted_once = True

    def transform(self, texts: Iterator[str]) -> Iterator[np.ndarray]:
        """Return vectorized texts from cache or by calling parent vectorizer."""
        total = 0

        while True:
            texts_chunk = list(islice(texts, self.batch_size))

            if not texts_chunk:
                break

            # check which texts needs vectorizing
            texts_chunk_hashes = [sha1_hash_from_text(t) for t in texts_chunk]
            uncached_texts = [t for i, t in enumerate(texts_chunk) if texts_chunk_hashes[i] not in self.store]

            # do vectorization for not yet known texts
            if len(uncached_texts) > 0:
                for i, uncached_features in enumerate(self.vectorizer.transform(iter(uncached_texts))):
                    uncached_hash = sha1_hash_from_text(uncached_texts[i])
                    self.store[uncached_hash] = uncached_features

            for text_hash in texts_chunk_hashes:
                yield self.store[text_hash]

            total += len(texts_chunk)
            logger.debug("loaded vectorizations or generated vectors for %d texts so far", total)

    def save(self, persist_dir):
        """Relay save call to vectorizer."""
        if not isinstance(self.vectorizer, PersistableVectorizer):
            raise ValueError("can not persist vectorizer that is not persistable")
        self.vectorizer.save(persist_dir)

    def load(self, persist_dir):
        """Relay load call to vectorizer."""
        if not isinstance(self.vectorizer, PersistableVectorizer):
            raise ValueError("can not load vectorizer that is not persistable")
        self.vectorizer.load(persist_dir)

    def __str__(self):
        """Return representative string of vectorizer."""
        return f"<CachedVectorizer of={str(self.vectorizer)}>"


class PersistedCachedVectorizer(AbstractVectorizer):
    """Stores vectorizations in persistent cache."""

    def __init__(self, filepath: str, vectorizer: AbstractVectorizer, batch_size: int = 100):
        """Initialize vectorizer.

        Parameters
        ----------
        filepath: str
            The file path of the database where vectorizations will be stored.
        vectorizer: AbstractVectorizer
            The parent vectorizer used to vectorize texts in case texts can not be found in cache.
        batch_size: int
            The number of text to process in one batch
        """
        self.filepath = filepath
        self.batch_size = batch_size
        self.store = SqliteDict(filepath, tablename="vectorizations", flag="c", autocommit=False)
        self.vectorizer = vectorizer

    def fit(self, texts: Iterator[str]):
        """Fit parent vectorizer."""
        self.vectorizer.fit(texts)

    def transform(self, texts: Iterator[str]) -> Iterator[np.ndarray]:
        """Return vectorized texts from cache or by calling parent vectorizer."""
        vectorizer_str = str(self.vectorizer)
        total = 0

        while True:
            texts_chunk = list(islice(texts, self.batch_size))

            if not texts_chunk:
                break

            # check which texts needs vectorizing
            texts_chunk_hashes = [sha1_hash_from_text(vectorizer_str + t) for t in texts_chunk]
            uncached_texts = [t for i, t in enumerate(texts_chunk) if texts_chunk_hashes[i] not in self.store]

            # do vectorization for not yet known texts
            if len(uncached_texts) > 0:
                for i, uncached_features in enumerate(self.vectorizer.transform(iter(uncached_texts))):
                    uncached_hash = sha1_hash_from_text(vectorizer_str + uncached_texts[i])
                    self.store[uncached_hash] = numpy_array_to_bytes(uncached_features)
                self.store.commit()

            for text_hash in texts_chunk_hashes:
                yield bytes_to_numpy_array(self.store[text_hash])

            total += len(texts_chunk)
            logger.debug("loaded vectorizations or generated vectors for %d texts so far", total)

    def __str__(self):
        """Return representative string of vectorizer."""
        return f"<PersistentCachedVectorizer of={str(self.vectorizer)} at={self.filepath}>"


def _extract_subtext_samples(text: str, samples: int) -> Iterator[str]:
    """Extract multiple texts from a longer text uniformily distributed over the whole entire text.

    Subtext starting positions are optimized by moving them back to the beginning of a current word.
    """
    offset = len(text) / samples
    for i in range(samples):
        start_idx = int(i * offset)

        # move to beginning of a word
        while start_idx > 0 and text[start_idx] != " ":
            start_idx -= 1
        if text[start_idx] == " ":
            start_idx += 1

        sub_text = text[start_idx:]
        # logger.debug("subtext is %s", sub_text[:100].strip())
        yield sub_text


class HuggingfaceBertVectorizer(AbstractVectorizer):
    """Evaluates a pre-trained bert model for text vectorization.

    Embeddings are extracted as the last hidden states of the first "[CLS]" token, see:
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html

    If `samples` is 1, only the first 512 sub-tokens from the text are used. If `samples` is larger than 1, then
    multiple text strings are uniformly extracted from the entire text (at positions `i/samples`), and embeddings are
    concatenated.

    If `samples? is larger than 1, the total batch size for each run of the Bert model will be `batch_size * samples`.
    """

    def __init__(
        self,
        model_identifier: str = "dbmdz/bert-base-german-uncased",
        batch_size: int = 4,
        subtext_samples: int = 1,
        hidden_states: int = 1,
        cache_dir: str = HUGGINGFACE_CACHE_DIR,
    ):
        """Initialize vectorizer.

        Parameters
        ----------
        model_identifier: str = "dbmdz/bert-base-german-uncased"
            The Huggingface model path of a pre-trained Bert model
        batch_size: int = 4
            The number of texts that are vectorized in one batch
        subtext_samples: int = 1
            The number of text samples to use from the text
        hidden_states: int = 1
            The number of hidden states that is extracted from the bert model and used as vectorizations.
            Multiple hidden state vectors are simply concatenated.
        cache_dir: str
            The directory storing pre-trained Huggingface models
        """
        self.model_identifier = model_identifier
        self.batch_size = batch_size
        self.subtext_samples = subtext_samples
        self.hidden_states = hidden_states
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer: Optional[Module] = None
        self.model: Optional[Module] = None

    def _load_model(self):
        if self.model is None:
            try:
                # try offline loading first
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_identifier,
                    cache_dir=self.cache_dir,
                    local_files_only=True
                )
                self.model = BertModel.from_pretrained(
                    self.model_identifier,
                    cache_dir=self.cache_dir,
                    local_files_only=True,
                )
            except OSError:
                # check online again
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_identifier, cache_dir=self.cache_dir)
                self.model = BertModel.from_pretrained(self.model_identifier, cache_dir=self.cache_dir)
            self.model.to(self.device)

    def fit(self, texts: Iterator[str]):
        """Not required for pre-trained Huggingface models."""

    def transform(self, texts: Iterator[str]) -> Iterator[np.ndarray]:
        """Return vectorized texts as a matrix with shape (len(texts), 768)."""
        # lazy load model only when it is actually needed
        self._load_model()

        if self.tokenizer is None or self.model is None:
            raise ValueError("cannot transform texts when tokenizer or model did not load correctly")

        features_chunks = []
        i = 0
        # total = len(texts)
        total_so_far = 0

        while True:
            texts_chunk = list(islice(texts, self.batch_size))

            if not texts_chunk:
                break

            # extract subtexts and remeber which subtext belongs to which text
            subtext_texts = [t for text in texts_chunk for t in _extract_subtext_samples(text, self.subtext_samples)]

            # tokenize texts
            encodings = self.tokenizer(
                subtext_texts,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            encodings.to(self.device)

            logger.debug(
                "evaluate huggingface model for vectorization of chunk %d, total %d",
                i, total_so_far
            )

            # evaluate model
            with torch.no_grad():
                output = self.model(**encodings)

            # remember model outputs
            hidden_states_list = list(range(self.hidden_states))
            features_chunk = output.last_hidden_state[:, hidden_states_list, :].cpu().detach().numpy()

            features_chunk = features_chunk.reshape((len(texts_chunk), -1))
            # logger.info("features chunk shape is %s", features_chunk.shape)
            features_chunks.append(features_chunk)

            for features in features_chunk:
                yield cast(np.ndarray, features)

            total_so_far += len(texts_chunk)
            i += 1

    def __str__(self):
        """Return representative string of vectorizer."""
        return f"<HFaceBertVectorizer model=\"{self.model_identifier}\" batch_size={self.batch_size} " \
            + f"subtext_samples={self.subtext_samples} hidden_states={self.hidden_states}>"
