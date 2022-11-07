"""Methods that process documents."""

# pylint: disable=too-many-arguments, import-outside-toplevel

import logging
import os
import functools

from typing import Callable, Iterator, List, Optional

from sqlitedict import SqliteDict

from slub_docsa.common.document import Document
from slub_docsa.common.sample import Sample
from slub_docsa.data.load.nltk import download_nltk
from slub_docsa.data.store.document import sha1_hash_from_text

logger = logging.getLogger(__name__)

NLTK_LANGUAGE_CODES_MAP = {
    "de": "german",
    "en": "english"
}
"""A map between two-letter language codes and nltk language names (only de/en)."""


def document_as_concatenated_string(
    doc: Document,
    skip_title: bool = False,
    skip_authors: bool = False,
    skip_abstract: bool = False,
    skip_fulltext: bool = False,
    max_length: Optional[int] = None,
) -> str:
    """Convert a document to a string by simple concatenation of all meta data.

    The resulting string is a concatenation of the document's title, authors, abstract and fulltext.

    Parameters
    ----------
    doc: Document
        Document that is being converted to a simple text string
    skip_authors: bool = False
        Whether to skip including the list of authors in the output string
    skip_abstract: bool = False
        Whether to skip including the abstract in the output string
    skip_fulltext: bool = False
        Whether to skip including the fulltext in the output string

    Returns
    -------
    str
        The concatenated text of a document as a simple string
    """
    text = ""
    if not skip_title and doc.title is not None:
        text += doc.title
    if not skip_authors and doc.authors is not None:
        text += "\n" + ", ".join(doc.authors)
    if not skip_abstract and doc.abstract is not None:
        text += "\n" + doc.abstract
    if not skip_fulltext and doc.fulltext is not None:
        text += "\n" + doc.fulltext
    if max_length is not None:
        return text[:max_length]
    return text


def nltk_word_tokenize_text_function(lang_code: str) -> Callable[[str], List[str]]:
    """Return a function that tokenizes text using the nltk word tokenizer.

    Parameters
    ----------
    lang_code: str
        two-letter language code (de/en)

    Returns
    -------
    Callable[[str], List[str]]
        a function that tokenizes a text into a sequence of tokens using the nltk word tokenizer
    """
    from nltk.tokenize import word_tokenize
    download_nltk("punkt")
    nltk_language = NLTK_LANGUAGE_CODES_MAP[lang_code]

    def tokenize(text: str) -> List[str]:
        return word_tokenize(text, language=nltk_language)

    return tokenize


def nltk_snowball_text_stemming_function(
    lang_code: str,
    remove_stopwords: bool = True
) -> Callable[[str], str]:
    """Return a function that applies the nltk snowball stemmer to a text.

    Parameters
    ---------
    lang_code: str
        a two-letter language code (de/en)
    remove_stopwords: bool = True
        whether to also remove stopwords as provided by the nltk library

    Returns
    -------
    Callable[[str], str]
        a function that applies the nltk snowball stemmer to a text
    """
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords

    if lang_code not in NLTK_LANGUAGE_CODES_MAP:
        raise ValueError(f"language code '{lang_code}' not supported for stemming")

    if remove_stopwords:
        download_nltk("stopwords")

    nltk_language = NLTK_LANGUAGE_CODES_MAP[lang_code]
    stemmer = SnowballStemmer(nltk_language)
    tokenize = nltk_word_tokenize_text_function(lang_code)
    stopword_set = set(stopwords.words(nltk_language))

    @functools.lru_cache(maxsize=1000000)
    def stem_func(word):
        return stemmer.stem(word)

    def is_stopword(token: str) -> bool:
        return token in stopword_set

    def stem_text(text: str) -> str:
        logger.debug("stemming text of size %d", len(text))
        filtered_tokens = [token for token in tokenize(text) if not remove_stopwords or not is_stopword(token)]
        return " ".join([stem_func(token) for token in filtered_tokens])

    return stem_text


def persisted_nltk_snowball_text_stemming_function(
    filepath: str,
    lang_code: str,
    remove_stopwords: bool = True
) -> Callable[[str], str]:
    """Stem text via the nltk snowball stemmer and store the stemmed text in a sqlite database for caching.

    Parameters
    ----------
    filepath: str
        The path to the sqlite file used as database for caching
    lang_code: str
        The two-letter language code required in case a text is not yet stemmed and needs stemming via nltk
    remove_stopwords: bool = True
        Whether to also remove stopwords during stemming

    Returns
    -------
    Callable[[str], str]
        a function that stems text and at the same time caches the stemmed text by saving it to a sqlite file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    store = SqliteDict(filepath, tablename="stemmed_text", flag="c", autocommit=True)
    stem_function = nltk_snowball_text_stemming_function(lang_code, remove_stopwords)

    def stem_text(text: str) -> str:
        text_hash = sha1_hash_from_text(lang_code + str(remove_stopwords) + text)

        # if stemmed text is known, return from cache
        if text_hash in store:
            return store[text_hash]

        # actually do stemming, and save for later
        stemmed_text = stem_function(text)
        store[text_hash] = stemmed_text
        return stemmed_text

    return stem_text


def nltk_snowball_document_stemming_function(
    lang_code: str,
    remove_stopwords: bool = True
) -> Callable[[Document], Document]:
    """Return a function that applies the nltk snowball stemmer to both the title, abstract and fulltext of a document.

    Parameters
    ----------
    lang_code: str
        a two-letter language code (de/en)
    remove_stopwords: bool = True
        whether to also remove stopwords as provided by the nltk library

    Returns
    -------
    Callable[[Document], Document]
        a function that applies the nltk snowball stemmer to a document
    """
    stem_text = nltk_snowball_text_stemming_function(lang_code, remove_stopwords)

    def apply_stemming_to_document(doc: Document) -> Document:
        stemmed_title = stem_text(doc.title) if doc.title is not None else None
        stemmed_abstract = stem_text(doc.abstract) if doc.abstract is not None else None
        stemmed_fulltext = stem_text(doc.fulltext) if doc.fulltext is not None else None
        logger.debug("stemmed title is %s", stemmed_title)
        return Document(
            uri=doc.uri,
            title=stemmed_title,
            authors=doc.authors,
            abstract=stemmed_abstract,
            fulltext=stemmed_fulltext
        )

    return apply_stemming_to_document


def apply_nltk_snowball_stemming_to_document_samples_iterator(
    samples_iterator: Iterator[Sample],
    lang_code: str
) -> Iterator[Sample]:
    """Apply the nltk snowball stemming to each document of a samples iterator.

    See `nltk_snowball_document_stemming_function`.

    Parameters
    ----------
    samples_iterator: Iterator[Sample]
        the iterator over samples whose documents are supposed to be stemmed
    lang_code: str
        a two-letter language code (de/en)

    Returns
    -------
    Iterator[Sample]
        an iterator over samples whose documents were stemmed
    """
    stemming_function = nltk_snowball_document_stemming_function(lang_code)
    for sample in samples_iterator:
        yield Sample(stemming_function(sample.document), sample.subjects)
