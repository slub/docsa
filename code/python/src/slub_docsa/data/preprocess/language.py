"""Language related pre-processing methods."""

import logging
from typing import Iterator

import langid

from slub_docsa.common.sample import Sample
from slub_docsa.data.preprocess.dataset import filter_samples_by_condition
from slub_docsa.data.preprocess.document import document_as_concatenated_string

logger = logging.getLogger(__name__)


def detect_language_from_text_via_langid(text: str) -> str:
    """Return language code for language detected by langid."""
    return langid.classify(text)[0]


def filter_samples_by_detected_language_via_langid(
    samples_iterator: Iterator[Sample],
    lang_code: str,
) -> Iterator[Sample]:
    """Return sample documents whose language detected by langid matches the expected language."""
    def condition(sample: Sample) -> bool:
        text = document_as_concatenated_string(sample.document)
        if text is not None:
            detected_lang_code = detect_language_from_text_via_langid(text)
            if detected_lang_code != lang_code:
                logger.debug(
                    "document '%s' with unexpected detected language of '%s'",
                    sample.document.uri,
                    detected_lang_code
                )
                logger.debug("document text begins with: %s", text[:100])
                return False
            return True
        return False

    return filter_samples_by_condition(samples_iterator, condition)
