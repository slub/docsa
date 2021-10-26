"""Methods that process documents."""

import logging

from typing import Callable, List

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from slub_docsa.common.document import Document
from slub_docsa.common.sample import SampleIterator
from slub_docsa.data.load.nltk import download_nltk

logger = logging.getLogger(__name__)

SNOWBALL_LANGUAGE_CODES = {
    "de": "german",
    "en": "english"
}


def document_as_concatenated_string(doc: Document, max_length: int = 2000) -> str:
    """Convert a document to a string by simple concatenation."""
    text = doc.title
    if doc.abstract is not None:
        text += "\n" + doc.abstract
    if doc.fulltext is not None:
        text += "\n" + doc.fulltext
    return text[:max_length]


def tokenize_text_function() -> Callable[[str], List[str]]:
    """Return a function that can be used to tokenize text."""
    download_nltk("punkt")

    def tokenize(text: str) -> List[str]:
        return word_tokenize(text)

    return tokenize


def snowball_document_stemming_function(lang_code: str) -> Callable[[Document], Document]:
    """Return a function that applies the snowball stemmer to both the document title, abstract and fulltext."""
    if lang_code not in SNOWBALL_LANGUAGE_CODES:
        raise ValueError("language code '%s' not supported for stemming" % lang_code)

    snowball_language = SNOWBALL_LANGUAGE_CODES[lang_code]

    stemmer = SnowballStemmer(snowball_language)
    tokenize = tokenize_text_function()

    def stem_text(text: str) -> str:
        return " ".join([stemmer.stem(token) for token in tokenize(text)])

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


def apply_snowball_stemming_to_document_samples_iterator(
    samples_iterator: SampleIterator,
    lang_code: str
) -> SampleIterator:
    """Apply Snowball stemming to each document of a samples iterator."""
    stemming_function = snowball_document_stemming_function(lang_code)
    for document, subjects in samples_iterator:
        yield stemming_function(document), subjects
