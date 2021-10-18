"""Language related pre-processing methods."""

import logging

import langid

from slub_docsa.common.dataset import Dataset
from slub_docsa.common.document import Document
from slub_docsa.common.subject import SubjectUriList
from slub_docsa.data.preprocess.dataset import filter_samples_from_dataset_by_condition

logger = logging.getLogger(__name__)


def detect_language_from_text_via_langid(text: str) -> str:
    """Return language code for language detected by langid."""
    return langid.classify(text)[0]


def filter_dataset_by_detected_fulltext_language_via_langid(dataset: Dataset, lang_code: str) -> Dataset:
    """Return dataset filtered for samples whose fulltext language detected by langid matches the expected language."""
    def condition(document: Document, _: SubjectUriList) -> bool:
        if document.fulltext is not None:
            detected_lang_code = detect_language_from_text_via_langid(document.fulltext)
            if detected_lang_code != lang_code:
                logger.debug(
                    "document '%s' has fulltext with unexpected detected language of '%s'",
                    document.uri,
                    detected_lang_code
                )
                return False
            return True
        return False

    return filter_samples_from_dataset_by_condition(dataset, condition)
