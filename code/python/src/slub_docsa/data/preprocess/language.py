"""Language related pre-processing methods."""

import logging
import os

from typing import Iterator, Optional, Any, cast

import langid
import fasttext

from slub_docsa.common.paths import get_resources_dir
from slub_docsa.common.sample import Sample

from slub_docsa.data.load.common import download_file
from slub_docsa.data.preprocess.dataset import filter_samples_by_condition
from slub_docsa.data.preprocess.document import document_as_concatenated_string


FASTTEXT_LANGUAGE_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
"""the default download URL for the fasttext language model"""

logger = logging.getLogger(__name__)


def download_fasttext_language_model(
    url: str = FASTTEXT_LANGUAGE_MODEL_URL,
    filepath: str = None,
):
    """Download fasttext language model.

    Parameters
    ----------
    url : str, optional
        the url to the fasttext language model, by default `FASTTEXT_LANGUAGE_MODEL_URL`
    filepath : str, optional
        the filepath where the language model is stored, if None, a default path of
        `SLUB_DOCSA_RESOURCES_DIR/fasttext/lid.176.bin` is used

    Returns
    -------
    None
        as soon as the file is downloaded, or immediately if the file already exists
    """
    if filepath is None:
        filepath = os.path.join(get_resources_dir(), "fasttext/lid.176.bin")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if not os.path.exists(filepath):
        logger.info("download fasttext language model to '%s'", filepath)
        download_file(url, filepath)


def load_fasttext_language_detector(
    min_probability: float = 0.5,
    model_url: str = FASTTEXT_LANGUAGE_MODEL_URL,
    model_filepath: str = None,
    download: bool = True
):
    """Load method that detects language code for text using fasttext.

    Parameters
    ----------
    min_probability : float, optional
        the minimum score to consider a language a correct detection, by default 0.5
    model_url : str, optional
        the URL that is used to download the required fasttext language model; default is `FASTTEXT_LANGUAGE_MODEL_URL`
    model_filepath : str, optional
        the filepath to the downloaded fasttext language model; if None, the filepath
        `$SLUB_DOCSA_RESOURCES_DIR/fasttext/lid.176.bin` is used
    download : bool, optional
        whether to do the download if the mode file is missing, by default True

    Returns
    -------
    _type_
        _description_
    """
    if model_filepath is None:
        model_filepath = os.path.join(get_resources_dir(), "fasttext/lid.176.bin")

    if download:
        download_fasttext_language_model(model_url, model_filepath)

    # disable fasttext log messages
    fasttext.FastText.eprint = lambda *args, **kwargs: None
    model = fasttext.load_model(model_filepath)

    def detect(text: str) -> Optional[str]:
        text = text.replace("\n", " ")
        result = cast(Any, model.predict([text], k=1))
        language = result[0][0][0].split("__label__")[1]
        probability = result[1][0][0]
        if probability >= min_probability:
            return language
        return None

    return detect


def detect_language_from_text_via_langid(text: str) -> str:
    """Return language code for language detected by langid.

    Parameters
    ----------
    text: str
        The text that is analyzed for its language

    Returns
    -------
    str
        The two-letter language code detected by langid
    """
    return langid.classify(text)[0]


def filter_samples_by_detected_language_via_langid(
    samples_iterator: Iterator[Sample],
    lang_code: str,
) -> Iterator[Sample]:
    """Return sample documents whose language detected by langid matches the expected language.

    Documents are converted to a simple text via the method
    `slub_docsa.data.preprocess.document.document_as_concatenated_string`.

    Parameters
    ----------
    samples_iterator: Iterator[Sample]
        an iterator over samples that is being filtered
    lang_code: str
        the expected language

    Returns
    -------
    Iterator[Sample]
        an iterator over samples only including samples that match the expected language
    """
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
