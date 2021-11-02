"""Methods to vectorize text."""

import logging
import os

from typing import Optional, Union, Sequence, cast
from itertools import islice

import torch
import numpy as np

from sqlitedict import SqliteDict
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer as ScikitTfidfVectorizer
from torch.nn.modules.module import Module
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel

from slub_docsa.common.paths import CACHE_DIR
from slub_docsa.data.store.array import bytes_to_numpy_array, numpy_array_to_bytes
from slub_docsa.data.store.document import sha1_hash_from_text

logger = logging.getLogger(__name__)

HUGGINGFACE_CACHE_DIR = os.path.join(CACHE_DIR, "huggingface")


class AbstractVectorizer:
    """Represents a vectorizer model that can be fitted to a corpus."""

    def fit(self, texts: Sequence[str]):
        """Optinally train a vectorizer model using the provided texts."""
        raise NotImplementedError()

    def transform(self, texts: Sequence[str]) -> Union[np.ndarray, csr_matrix]:
        """Return vector representation of texts as array with shape (len(texts), embedding_size)."""
        raise NotImplementedError()


class TfidfVectorizer(AbstractVectorizer):
    """Vectorizer using Scikit TfidfVectorizer."""

    def __init__(self, max_features=10000, **kwargs):
        """Initialize vectorizer.

        Parameters
        ----------
        max_features: int = 10000
            The maximum number of unique tokens to extract from text during fit.
        """
        self.max_features = max_features
        self.vectorizer = ScikitTfidfVectorizer(max_features=max_features, **kwargs)

    def fit(self, texts: Sequence[str]):
        """Fit vectorizer."""
        self.vectorizer.fit(texts)

    def transform(self, texts: Sequence[str]) -> csr_matrix:
        """Return vectorized texts."""
        return self.vectorizer.transform(texts)

    def __str__(self):
        """Return representative string of vectorizer."""
        return f"<TfidfVectorizer max_features={self.max_features}>"


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

    def fit(self, texts: Sequence[str]):
        """Fit vectorizer."""

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Return vectorized texts."""
        return np.random.random((len(texts), self.size))

    def __str__(self):
        """Return representative string of vectorizer."""
        return "<RandomVectorizer>"


class PersistedCachedVectorizer(AbstractVectorizer):
    """Stores vectorizations in persistent cache."""

    def __init__(self, filepath: str, vectorizer: AbstractVectorizer):
        """Initialize vectorizer.

        Parameters
        ----------
        filepath: str
            The file path of the database where vectorizations will be stored.
        vectorizer: AbstractVectorizer
            The parent vectorizer used to vectorize texts in case texts can not be found in cache.
        """
        self.filepath = filepath
        self.store = SqliteDict(filepath)
        self.vectorizer = vectorizer

    def fit(self, texts: Sequence[str]):
        """Fit parent vectorizer."""
        self.vectorizer.fit(texts)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Return vectorized texts from cache or by calling parent vectorizer."""
        # check which texts needs vectorizing
        text_hashes = [sha1_hash_from_text(t) for t in texts]
        uncached_texts = [t for i, t in enumerate(texts) if text_hashes[i] not in self.store]

        # do vectorization for not yet known texts
        if len(uncached_texts) > 0:
            uncached_features = self.vectorizer.transform(uncached_texts)
            for i, text in enumerate((uncached_texts)):
                self.store[sha1_hash_from_text(text)] = numpy_array_to_bytes(uncached_features[i])
            self.store.commit()

        # read all vectors from cache
        return cast(np.ndarray, np.vstack([bytes_to_numpy_array(self.store[h]) for h in text_hashes]))

    def __str__(self):
        """Return representative string of vectorizer."""
        return f"<PersistentCachedVectorizer of={str(self.vectorizer)} at={self.filepath}>"


class HuggingfaceBertVectorizer(AbstractVectorizer):
    """Evaluates a pre-trained bert model for text vectorization.

    Embeddings are extracted as the last hidden states of the first "[CLS]" token, see:
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
    """

    def __init__(
        self,
        model_identifier: str = "dbmdz/bert-base-german-uncased",
        batch_size: int = 32,
        cache_dir: str = HUGGINGFACE_CACHE_DIR,
    ):
        """Initialize vectorizer.

        Parameters
        ----------
        model_identifier: str
            The Huggingface model path of a pre-trained Bert model
        batch_size: int
            The number of texts that are vectorized in one batch
        cache_dir: str
            The directory storing pre-trained Huggingface models
        """
        self.model_identifier = model_identifier
        self.batch_size = batch_size
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

    def fit(self, texts: Sequence[str]):
        """Not required for pre-trained Huggingface models."""

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Return vectorized texts as a matrix with shape (len(texts), 768)."""
        # lazy load model only when it is actually needed
        self._load_model()

        if self.tokenizer is None or self.model is None:
            raise ValueError("cannot transform texts when tokenizer or model did not load correctly")

        text_iterator = iter(texts)
        features_chunks = []
        i = 0
        total = len(texts)
        total_so_far = 0

        while True:
            texts_chunk = list(islice(text_iterator, self.batch_size))

            if not texts_chunk:
                break

            # tokenize texts
            encodings = self.tokenizer(
                texts_chunk,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            encodings.to(self.device)

            logger.debug(
                "evaluate huggingface model for vectorization of chunk %d, total %d / %d",
                i, total_so_far, total
            )

            # evaluate model
            with torch.no_grad():
                output = self.model(**encodings)

            # remember model outputs
            features_chunk = output.last_hidden_state[:, 0, :].cpu().detach().numpy()
            features_chunks.append(features_chunk)

            total_so_far += len(texts_chunk)
            i += 1

        return cast(np.ndarray, np.vstack(features_chunks))

    def __str__(self):
        """Return representative string of vectorizer."""
        return f"<HFaceBertVectorizer path=\"{self.model_identifier}\" batch_size={self.batch_size}>"
