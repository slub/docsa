"""Compares qucosa language annotations with automatically detected fulltext languages."""

# pylint: disable=invalid-name

import logging

from slub_docsa.data.load.qucosa import read_qucosa_samples, read_qucosa_documents_from_directory
from slub_docsa.data.preprocess.language import detect_language_from_text_via_langid

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    lang_code = "de"

    sample_iterator = read_qucosa_samples(read_qucosa_documents_from_directory(), "fulltexts", "rvk", lang_code)

    total = 0
    count = 0
    for doc, _ in sample_iterator:
        total += 1
        if doc.fulltext is not None:
            detected_lang_code = detect_language_from_text_via_langid(doc.fulltext)
            logger.debug("detected language code %s for doc %s", detected_lang_code, doc.uri)
            if detected_lang_code != lang_code:
                count += 1
                logger.info("qucosa document %s is detected with wrong language but %s", doc.uri, detected_lang_code)

    logger.info("%d of in total %d documents are not detected as correct language", count, total)
