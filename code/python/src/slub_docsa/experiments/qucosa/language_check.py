"""Compares qucosa language annotations with from fulltext detected languages."""

# pylint: disable=invalid-name

import logging

from slub_docsa.data.load.qucosa import read_qucosa_fulltext_rvk_training_dataset, read_qucosa_documents_from_directory
from slub_docsa.data.preprocess.language import detect_language_from_text_via_langid

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    lang_code = "de"

    dataset = read_qucosa_fulltext_rvk_training_dataset(
        read_qucosa_documents_from_directory(), lang_code
    )

    count = 0
    for doc in dataset.documents:
        if doc.fulltext is not None:
            detected_lang_code = detect_language_from_text_via_langid(doc.fulltext)
            logger.debug("detected language code %s for doc %s", detected_lang_code, doc.uri)
            if detected_lang_code != lang_code:
                count += 1
                logger.info("qucosa document %s is detected with wrong language but %s", doc.uri, detected_lang_code)

    logger.info("%d of in total %d documents are not detected as correct language", count, len(dataset.documents))
