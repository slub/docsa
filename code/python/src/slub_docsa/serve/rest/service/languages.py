"""REST service implementation for langauge detection."""

from typing import Sequence

from slub_docsa.common.document import Document
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.serve.common import LanguagesRestService
from slub_docsa.data.preprocess.language import detect_language_from_text_via_langid


class LangidLanguagesRestService(LanguagesRestService):
    """Language detection via langid."""

    def find_languages(self):
        """Return the list of available languages."""
        return ["de", "en"]

    def detect(self, documents: Sequence[Document]) -> Sequence[str]:
        """Detect the language of each document."""
        return [detect_language_from_text_via_langid(document_as_concatenated_string(d)) for d in documents]
