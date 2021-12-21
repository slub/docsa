"""Reads qucosa documents from jsonl files and the SLUB ElasticSearch server."""

# pylint: disable=unexpected-keyword-arg, too-many-arguments, broad-except, too-many-branches
# pylint: disable=too-many-return-statements, unnecessary-lambda

import os
import gzip
import json
import logging
import time
import pickle  # nosec

from typing import Callable, Iterable, Iterator, Mapping, List, Any, Optional, Tuple, Union
from typing_extensions import Literal

from elasticsearch import Elasticsearch
from lxml import etree  # nosec

from slub_docsa.common.document import Document
from slub_docsa.common.sample import Sample
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.load.ddc import ddc_correct_short_keys, ddc_key_to_uri, get_ddc_subject_store, is_valid_ddc_uri
from slub_docsa.data.load.ddc import extend_ddc_subject_list_with_ancestors
from slub_docsa.data.load.rvk import get_rvk_subject_store, rvk_notation_to_uri
from slub_docsa.common.paths import get_cache_dir, get_resources_dir

logger = logging.getLogger(__name__)

SLUB_ELASTICSEARCH_SERVER_URL = os.environ.get("SLUB_ELASTICSEARCH_SERVER_URL", "es.data.slub-dresden.de")
"""Default server URL (`es.data.slub-dresden.de`) for SLUB elasticsearch server.
Can be overwritten using the environment variable `SLUB_ELASTICSEARCH_SERVER_URL`.
"""

SLUB_ELASTICSEARCH_SERVER_USER = os.environ.get("SLUB_ELASTICSEARCH_SERVER_USER", None)
"""Default user name (`None`) for the SLUB elasticsearch server.
Can be overwritten using the environment variable `SLUB_ELASTICSEARCH_SERVER_USER`.
"""

SLUB_ELASTICSEARCH_SERVER_PASSWORD = os.environ.get("SLUB_ELASTICSEARCH_SERVER_PASSWORD", None)
"""Default password (`None`) for the SLUB elasticsearch server.
Can be overwritten using the environment variable `SLUB_ELASTICSEARCH_SERVER_PASSWORD`.
"""

# helper types for processing qucosa json documents
QucosaJsonClassificationElementType = Mapping[str, Any]
QucosaJsonClassifcationType = List[QucosaJsonClassificationElementType]

QucosaJsonDocument = Mapping[str, Any]
"""Represents a qucosa json document. Type is not fully expressed though."""

QUCOSA_LANGUAGE_CODE_MAP = {
    "de": "ger",
    "en": "eng",
}
"""Mapping between two-letter language codes and 3-letter language abbrevations used in qucosa json documents."""


def _initialize_elasticsearch_connection(
    host: str = SLUB_ELASTICSEARCH_SERVER_URL,
    http_auth: Optional[Tuple[str, str]] = None,
):
    if http_auth is None and \
            SLUB_ELASTICSEARCH_SERVER_USER is not None and \
            SLUB_ELASTICSEARCH_SERVER_PASSWORD is not None:
        http_auth = (SLUB_ELASTICSEARCH_SERVER_USER, SLUB_ELASTICSEARCH_SERVER_PASSWORD)

    return Elasticsearch([
        {"host": host, "use_ssl": True, "port": 443}
    ], http_auth=http_auth)


def read_qucosa_metadata_from_elasticsearch(
    host: str = SLUB_ELASTICSEARCH_SERVER_URL,
    http_auth: Optional[Tuple[str, str]] = None,
    index_name: str = "fulltext_qucosa",
    time_between_requests: float = 10.0,
    page_size: int = 100,
    scroll_timeout: str = "1m",
    query: Any = None,
) -> Iterable[QucosaJsonDocument]:
    """Read Qucosa documents and its meta data from the SLUB ElasticSearch server.

    Parameters
    ----------
    host: str = SLUB_ELASTICSEARCH_SERVER_URL
        The hostname of the ElasticSearch server
    http_auth: Optional[Tuple[str, str]]
        Http basic auth parameters as tuple of username and password. If http_auth is None, but environment variables
        `SLUB_ELASTICSEARCH_SERVER_USER` and `SLUB_ELASTICSEARCH_SERVER_PASSWORD` are set, then these are used as
        username and password.
    index_name: str = "fulltext_qucosa"
        The name of the ElasticSearch index to be queried.
    time_between_requests: float = 10.0
        The minimum time between subsequent requests to the ElasticSearch server
    page_size: int = 100
        The maximum number of documents to be retrieved in one request.
    scroll_timeout: str = "1m"
        The time a scroll operation is keep alive by the ElasticSearch server (which guranantees that documents are
        returned in the correct order over multiple subsequent requests).

    Returns
    -------
    Iterable[QucosaJsonDocument]
        The documents retrieved from the ElasticSearch index.
    """
    es_server = _initialize_elasticsearch_connection(host, http_auth)

    if query is None:
        query = {"match_all": {}}

    last_request = time.time()
    results = es_server.search(index=index_name, query=query, scroll=scroll_timeout, size=page_size)

    if "_scroll_id" not in results:
        raise RuntimeError("no scroll id returned by elasticsearch")
    scroll_id = results["_scroll_id"]

    while results and "hits" in results and "hits" in results["hits"] and results["hits"]["hits"]:

        hits = results["hits"]["hits"]
        for hit in hits:
            yield hit["_source"]

        # wait until X seconds after last request
        time.sleep(max(time_between_requests - (time.time() - last_request), 0.0))

        # do next request
        last_request = time.time()
        results = es_server.scroll(scroll_id=scroll_id, scroll=scroll_timeout)


def number_of_qucosa_metadata_in_elasticsearch(
    host: str = SLUB_ELASTICSEARCH_SERVER_URL,
    http_auth: Optional[Tuple[str, str]] = None,
    index_name: str = "fulltext_qucosa",
) -> int:
    """Return the number of qucosa documents currently available at the SLUB elastic search server.

    Parameters
    ----------
    host: str = SLUB_ELASTICSEARCH_SERVER_URL
        The hostname of the ElasticSearch server
    http_auth: Optional[Tuple[str, str]]
        Http basic auth parameters as tuple of username and password. If http_auth is None, but environment variables
        `SLUB_ELASTICSEARCH_SERVER_USER` and `SLUB_ELASTICSEARCH_SERVER_PASSWORD` are set, then these are used as
        username and password.
    index_name: str = "fulltext_qucosa"
        The name of the ElasticSearch index to be queried.

    Returns
    -------
    int
        the number of qucosa documents
    """
    es_server = _initialize_elasticsearch_connection(host, http_auth)
    return es_server.count(index=index_name, body={"query": {"match_all": {}}})["count"]


def save_qucosa_documents_to_directory(
    qucosa_documents: Iterable[QucosaJsonDocument],
    directory: str = None,
    max_documents_per_file: int = 100
):
    """Save qucosa documents to a directory as gzip compressed jsonl files.

    Qucosa documents can be read again by calling `read_qucosa_documents_from_directory`.

    Parameters
    ----------
    qucosa_documents: Iterable[QucosaDocument]
        The qucosa documents that are supposed to be saved to the directory.
    directory: str = "<resources_dir>/qucosa/fulltext_mapping/"
        The directory where qucosa documents are stored as jsonl files.
    max_documents_per_file: int = 100
        The maximum number of documents to store in one single gzip compressed jsonl file.

    Returns
    -------
    None
    """
    if directory is None:
        directory = os.path.join(get_resources_dir(), "qucosa/fulltext_mapping/")
    os.makedirs(directory, exist_ok=True)

    if len(os.listdir(directory)) > 0:
        raise ValueError("directory must be empty in order to save qucosa documents")

    file_count = 0
    doc_count = 0
    qucosa_buffer = []

    def write_buffer_to_file(buffer, i):
        filepath = os.path.join(directory, f"qucosa-{i}.jsonl.gz")
        with gzip.open(filepath, "wt", encoding="utf-8") as f_jsonl:
            for buffer_entry in buffer:
                f_jsonl.write(json.dumps(buffer_entry))
                f_jsonl.write("\n")

    for qucosa_document in qucosa_documents:

        qucosa_buffer.append(qucosa_document)
        doc_count += 1

        if doc_count % max_documents_per_file == 0:
            logger.info("saved %d qucosa documents so far", doc_count)

        if len(qucosa_buffer) >= max_documents_per_file:
            write_buffer_to_file(qucosa_buffer, file_count)
            file_count += 1
            qucosa_buffer = []

    if len(qucosa_buffer) > 0:
        write_buffer_to_file(qucosa_buffer, file_count)


def read_qucosa_documents_from_directory(
    directory: str = None,
    fallback_retrieve_from_elasticsearch: bool = True,
    check_elasticsearch_document_count: bool = False,
) -> Iterable[QucosaJsonDocument]:
    """Read qucosa documents from a directory of gzip compressed jsonl files.

    Parameters
    ----------
    directory: str = "<resources_dir>/qucosa/fulltext_mapping/"
        The directory which is scanned for gzip compressed jsonl files that are read as Qucosa documents.
    fallback_retrieve_from_elasticsearch: bool = True
        If true will try to download and save qucosa documents from SLUB elasticsearch server unless directory
        already exists.
    check_elasticsearch_document_count: bool = False
        If true, will query the SLUB elastic search server for the current amount of qucos documents and compare
        that to the number of documents stored on disc; it they do not match, an error is thrown

    Returns
    -------
    Iterable[QucosaJsonDocument]
        The Qucosa documents that were read from the provided directory.
    """
    if directory is None:
        directory = os.path.join(get_resources_dir(), "qucosa/fulltext_mapping/")
    if not os.path.exists(directory) and fallback_retrieve_from_elasticsearch:
        logger.info("qucosa directory does not exist, try to read and save documents from SLUB elasticsearch")
        qucosa_document_iterator = read_qucosa_metadata_from_elasticsearch()
        save_qucosa_documents_to_directory(qucosa_document_iterator, directory)
    if os.path.exists(directory) and check_elasticsearch_document_count:
        es_count = number_of_qucosa_metadata_in_elasticsearch()
        logger.debug("SLUB elastic search index contains %d qucosa documents", es_count)
        dir_count = 0
        for entry in os.scandir(directory):
            if entry.is_file():
                with gzip.open(entry.path, "r") as f_jsonl:
                    dir_count += sum(1 for line in f_jsonl)
        logger.debug("downloaded qucosa json files contain %d qucosa documents", dir_count)
        if dir_count != es_count:
            raise RuntimeError(
                f"number of downloaded qucosa documents ({dir_count}) "
                + f"does not match SLUB elastic search server ({es_count})"
            )

    if not os.path.exists(directory):
        raise ValueError(f"directory '{directory}' does not exist" % directory)

    for entry in os.scandir(directory):
        if entry.is_file():
            with gzip.open(entry.path, "r") as f_jsonl:
                for line in f_jsonl.readlines():
                    yield json.loads(line)


def _get_rvk_notations_from_qucosa_metadata(doc: QucosaJsonDocument) -> List[str]:
    """Return the list of RVK annotations from a qucosa json document."""
    classification: QucosaJsonClassifcationType = doc["metadata"]["classification"]
    rvk_dict = list(filter(lambda d: d["type"] == "rvk", classification))[0]
    keys = rvk_dict["keys"]
    if isinstance(keys, str):
        return list(map(lambda s: s.strip(), keys.split(",")))
    if isinstance(keys, list):
        return keys
    return []


def _get_rvk_subjects_from_qucosa_metadata(doc: QucosaJsonDocument) -> List[str]:
    """Return the list of RVK annotations from a qucosa json document."""
    return [rvk_notation_to_uri(n) for n in _get_rvk_notations_from_qucosa_metadata(doc)]


def _get_ddc_keys_from_qucosa_metadata(doc: QucosaJsonDocument) -> List[str]:
    """Return the list of DDC annotations from a qucosa json document."""
    classification: QucosaJsonClassifcationType = doc["metadata"]["classification"]
    ddc_dict = list(filter(lambda d: d["type"] == "ddc", classification))[0]
    keys = ddc_dict["keys"]
    if isinstance(keys, str):
        return list(map(lambda s: ddc_correct_short_keys(s.strip()), keys.split(",")))
    if isinstance(keys, list):
        return list(map(ddc_correct_short_keys, keys))
    return []


def _get_ddc_subjects_from_qucosa_metadata(doc: QucosaJsonDocument) -> List[str]:
    return [ddc_key_to_uri(k) for k in _get_ddc_keys_from_qucosa_metadata(doc)]


def _get_document_title_from_qucosa_metadata(
    doc: QucosaJsonDocument,
    lang_code: Optional[str] = "de",
    title_attribute: str = "title",
) -> Optional[str]:
    """Return the document title from a qucosa json document."""
    if title_attribute not in doc:
        logger.debug("document does not have a %s attribute", title_attribute)
        return None

    if "text" not in doc[title_attribute]:
        logger.debug("document does not have a %s text attribute", title_attribute)
        return None

    text_entry = doc[title_attribute]["text"]

    if lang_code is not None:
        # language filtering is required
        if "language" not in doc[title_attribute]:
            logger.debug("document has no %s language attribute", title_attribute)
            return None

        language_entry = doc[title_attribute]["language"]

        if not isinstance(text_entry, type(language_entry)):
            logger.debug("document %s text and language attribute have different type", title_attribute)
            return None

        if isinstance(language_entry, str):
            if language_entry == QUCOSA_LANGUAGE_CODE_MAP[lang_code]:
                return text_entry
            logger.debug("found document title with wrong language code %s", language_entry)
            return None

        if isinstance(language_entry, list):

            if not isinstance(text_entry, list):
                logger.debug("document %s text attribute is not a list, but language attribute is", title_attribute)

            if len(language_entry) != len(text_entry):
                logger.debug("document %s list not of the same length as language list", title_attribute)

            qucosa_lang_code = QUCOSA_LANGUAGE_CODE_MAP[lang_code]
            if qucosa_lang_code in language_entry:
                language_match_idx = language_entry.index(qucosa_lang_code)
                return text_entry[language_match_idx]

            logger.debug("found document title with other language codes %s", str(language_entry))
            return None

    # language doesn't matter, just return first title text
    if isinstance(text_entry, list):
        return text_entry[0]
    return text_entry


def _get_document_abstract_from_qucosa_metadata(
    doc: QucosaJsonDocument,
    lang_code: Optional[str] = None
) -> Optional[str]:
    """Return the document abstract from a qucosa json document."""
    for fulltext in doc["fulltext"]:
        if fulltext["type"] == "abstract" and "text" in fulltext and "language" in fulltext:
            language_entry = fulltext["language"]
            text_entry = fulltext["text"]

            # if language does not matter
            if lang_code is None:
                if isinstance(text_entry, list):
                    return text_entry[0]
                return text_entry

            # convert lang code to qucosa lang code
            qucosa_lang_code = QUCOSA_LANGUAGE_CODE_MAP[lang_code]

            # if there are multiple abstracts, find the abstract matching the current language
            if isinstance(language_entry, list) and isinstance(text_entry, list) and qucosa_lang_code in language_entry:
                language_match_idx = language_entry.index(qucosa_lang_code)
                return text_entry[language_match_idx]

            # if there is only one abstract, check that it is the correct language
            if isinstance(language_entry, str) and isinstance(text_entry, str) and language_entry == qucosa_lang_code:
                return text_entry
    return None


def _extract_text_from_qucosa_document_fulltext_xml(fulltext_xml: str) -> Optional[str]:
    """Extract raw text from the xml fulltext documents contained in a qucosa json document."""
    try:
        xml_root = etree.fromstring(fulltext_xml, parser=None)
        extracted_text = "\n".join([t for t in etree.XPath("//text()")(xml_root) if t and t.strip()])
        return extracted_text
    except Exception as xml_e:
        logger.error("error parsing fulltext xml document %s", xml_e)
    return None


def _get_document_fulltext_from_qucosa_metadata(doc: QucosaJsonDocument, lang_code: Optional[str]) -> Optional[str]:
    """Return the fulltext raw text from a qucosa json document."""
    for fulltext in doc["fulltext"]:
        if fulltext["type"] == "fulltext" and "text" in fulltext and "language" in fulltext:
            language_entry = fulltext["language"]
            text_entry = fulltext["text"]

            if text_entry is None:
                logger.debug(
                    "qucosa document '%s' has empty text field for fulltext",
                    _get_document_id_from_qucosa_metadate(doc)
                )
                continue

            if text_entry == "NOT EXTRACTED":
                logger.debug(
                    "qucosa document '%s' has NOT EXTRACTED as fulltext text",
                    _get_document_id_from_qucosa_metadate(doc)
                )
                continue

            extraction_source = None

            # language doesnt matter
            if lang_code is None:
                if isinstance(text_entry, list):
                    extraction_source = text_entry[0]
                else:
                    extraction_source = text_entry

            # check that language and text entries are of the same type (either both string or both list)
            if lang_code is not None and not isinstance(language_entry, type(text_entry)):
                logger.debug(
                    "qucosa document '%s' incorrect format for language entry %s with %s and text entry %s",
                    _get_document_id_from_qucosa_metadate(doc),
                    repr(type(language_entry)),
                    repr(language_entry),
                    repr(type(text_entry))
                )
                continue

            # if there are multiple fulltexts, find the one matching the requested language
            if lang_code is not None and isinstance(language_entry, list) and isinstance(text_entry, list):
                qucosa_lang_code = QUCOSA_LANGUAGE_CODE_MAP[lang_code]
                if qucosa_lang_code in language_entry:
                    language_match_idx = language_entry.index(qucosa_lang_code)
                    extraction_source = text_entry[language_match_idx]

                    if extraction_source is None or len(extraction_source) < 3:
                        logger.debug(
                            "qucosa document '%s' fulltext text is too short: %s",
                            _get_document_id_from_qucosa_metadate(doc),
                            repr(extraction_source)
                        )
                        continue

            # if there is only one fulltext, check that it is the correct language
            if lang_code is not None and isinstance(language_entry, str) and isinstance(text_entry, str):
                qucosa_lang_code = QUCOSA_LANGUAGE_CODE_MAP[lang_code]
                if language_entry == qucosa_lang_code:
                    extraction_source = text_entry

                    if extraction_source is None or len(extraction_source) < 3:
                        logger.debug(
                            "qucosa document '%s' fulltext text is too short: %s",
                            _get_document_id_from_qucosa_metadate(doc),
                            repr(extraction_source)
                        )
                        continue

            if extraction_source is None:
                logger.debug(
                    "qucosa document '%s' does not have requested fulltext language code, but %s",
                    _get_document_id_from_qucosa_metadate(doc),
                    repr(language_entry),
                )
                continue

            extracted_text = _extract_text_from_qucosa_document_fulltext_xml(extraction_source)

            if extracted_text is None or len(extracted_text) < 3:
                logger.debug(
                    "qucosa document '%s' has fulltext, but extracted text is too short",
                    _get_document_id_from_qucosa_metadate(doc),
                )
                continue

            return extracted_text
    return None


def _get_document_id_from_qucosa_metadate(doc: QucosaJsonDocument) -> str:
    """Return the document id from a qucosa metadata object."""
    return doc["id"]


def _read_qucosa_generic_samples(
    qucosa_iterator: Iterable[QucosaJsonDocument],
    create_document_from_qucosa: Callable[[QucosaJsonDocument, Optional[str]], Optional[Document]],
    read_subjects_from_doc: Callable[[QucosaJsonDocument], List[str]],
    subject_hierarchy: SubjectHierarchy,
    lang_code: Optional[str] = None,
) -> Iterator[Sample]:
    """Read qucosa data and extract documents and subjects."""
    logger.debug("read qucosa meta data json")
    for doc in qucosa_iterator:
        subjects = read_subjects_from_doc(doc)
        subjects = list(filter(lambda s_uri: s_uri in subject_hierarchy, subjects))

        if len(subjects) < 1:
            logger.debug("qucosa document with no known subjects: %s", _get_document_id_from_qucosa_metadate(doc))
            continue

        document = create_document_from_qucosa(doc, lang_code)
        if document is None:
            continue

        yield Sample(document, subjects)


def _make_title_only_doc(doc: QucosaJsonDocument, lang_code: Optional[str]) -> Optional[Document]:
    """Only uses title with at least 10 characters as document text."""
    doc_uri = "uri://" + _get_document_id_from_qucosa_metadate(doc)
    doc_title = _get_document_title_from_qucosa_metadata(doc, lang_code, "title")
    doc_subtitle = _get_document_title_from_qucosa_metadata(doc, lang_code, "subtitle")

    if doc_title is None:
        logger.debug("qucosa document '%s' with no title or of wrong language", doc_uri)
        return None

    full_title = doc_title
    if doc_subtitle is not None:
        full_title = doc_title + " " + doc_subtitle

    if len(full_title) < 10:
        logger.debug("qucosa document with short full title '%s': %s", full_title, doc_uri)
        return None

    return Document(uri=doc_uri, title=full_title)


def _make_title_and_abstract_doc(doc: QucosaJsonDocument, lang_code: Optional[str]) -> Optional[Document]:
    """Only uses title with at least 10 characters as document text."""
    doc_uri = "uri://" + _get_document_id_from_qucosa_metadate(doc)
    doc_title = _get_document_title_from_qucosa_metadata(doc, lang_code)
    doc_abstract = _get_document_abstract_from_qucosa_metadata(doc, lang_code)

    if doc_title is None or len(doc_title) < 1:
        logger.debug("qucosa document with no title or of wrong language: %s", doc_uri)
        return None

    if doc_abstract is None:
        logger.debug("qucosa document with no abstract: %s", doc_uri)
        return None

    if len(doc_abstract) < 20:
        logger.debug("qucosa document with too short abstract '%s': %s", doc_abstract, doc_uri)
        return None

    return Document(uri=doc_uri, title=doc_title, abstract=doc_abstract)


def _make_title_and_fulltext_doc(doc: QucosaJsonDocument, lang_code: Optional[str]) -> Optional[Document]:
    """Only uses title with at least 10 characters as document text."""
    doc_uri = "uri://" + _get_document_id_from_qucosa_metadate(doc)
    doc_title = _get_document_title_from_qucosa_metadata(doc, lang_code)
    doc_fulltext = _get_document_fulltext_from_qucosa_metadata(doc, lang_code)

    if doc_title is None or len(doc_title) < 1:
        logger.debug("qucosa document with no title or of wrong language: %s", doc_uri)
        return None

    if doc_fulltext is None:
        logger.debug("qucosa document with no fulltext: %s", doc_uri)
        return None

    return Document(uri=doc_uri, title=doc_title, fulltext=doc_fulltext)


def _make_filtered_complete_doc_generator(make_func: Any):
    """Return a function that filters documents based on whether they contain both title, abstract and fulltext.

    However, only extract text based on another function, e.g., that only extracts the title from the documents.

    The idea is to generate dataset variants that can be fairly compared with each other, since every dataset variant
    is based on the exact same document set.
    """

    def _make_doc(doc: QucosaJsonDocument, lang_code: Optional[str]) -> Optional[Document]:
        doc_uri = "uri://" + _get_document_id_from_qucosa_metadate(doc)
        doc_title = _get_document_title_from_qucosa_metadata(doc, lang_code, "title")
        doc_subtitle = _get_document_title_from_qucosa_metadata(doc, lang_code, "subtitle")
        doc_abstract = _get_document_abstract_from_qucosa_metadata(doc, lang_code)
        doc_fulltext = _get_document_fulltext_from_qucosa_metadata(doc, lang_code)

        if doc_title is None or len(doc_title) < 1:
            logger.debug("qucosa document with no title or of wrong language: %s", doc_uri)
            return None

        full_title = doc_title
        if doc_subtitle is not None:
            full_title = doc_title + " " + doc_subtitle

        if len(full_title) < 10:
            logger.debug("qucosa document with short full title '%s': %s", full_title, doc_uri)
            return None

        if doc_abstract is None:
            logger.debug("qucosa document with no abstract: %s", doc_uri)
            return None

        if len(doc_abstract) < 20:
            logger.debug("qucosa document with too short abstract '%s': %s", doc_abstract, doc_uri)
            return None

        if doc_fulltext is None:
            logger.debug("qucosa document with no fulltext: %s", doc_uri)
            return None

        return make_func(doc, lang_code)

    return _make_doc


def get_qucosa_ddc_subject_store(
    ddc_list_filepath: str = None,
) -> SubjectHierarchy:
    """Return the ddc subject hierarchy which contains only subjects present in the qucosa dataset."""
    if ddc_list_filepath is None:
        ddc_list_filepath = os.path.join(get_cache_dir(), "qucosa/ddc_list.pickle.gz")
    if not os.path.exists(ddc_list_filepath):
        logger.debug("create and fill ddc subject store (may take some time)")
        qucosa_iterator = read_qucosa_documents_from_directory()
        subjects = []

        for doc in qucosa_iterator:
            subjects += _get_ddc_subjects_from_qucosa_metadata(doc)

        subjects = list(filter(is_valid_ddc_uri, subjects))
        subjects = extend_ddc_subject_list_with_ancestors(subjects)
        with gzip.open(ddc_list_filepath, "wb") as file:
            pickle.dump(subjects, file)

    with gzip.open(ddc_list_filepath, "rb") as file:
        subjects = pickle.load(file)  # nosec
        return get_ddc_subject_store(subject_uris=subjects)


def qucosa_subject_hierarchy_by_subject_schema(
    subject_schema: Union[Literal["rvk"], Literal["ddc"]],
) -> SubjectHierarchy:
    """Return either rvk or ddc subject hierarchy depending on requested subject schema."""
    return {
        "rvk": lambda: get_rvk_subject_store(),
        "ddc": lambda: get_qucosa_ddc_subject_store(),
    }[subject_schema]()


def read_qucosa_samples(
    qucosa_iterator: Iterable[QucosaJsonDocument] = None,
    metadata_variant: str = "titles",
    subject_schema: Union[Literal["rvk"], Literal["ddc"]] = "rvk",
    lang_code: Optional[str] = "de",
) -> Iterator[Sample]:
    """Read qucosa documents and use only document titles as training data.

    Fields abstracts and fulltext are not loaded.

    Parameters
    ----------
    qucosa_iterator: Iterable[QucosaJsonDocument] = None
        An iterator over qucosa json documents. If None, tries to load them from default directory
        (and SLUB elasticsearch if not available).
    lang_code: Optional[str] = "de"
        The language code which decides which text is extracted. If None, all documents are returned independent of
        language. Otherwise, documents that do not contain text of the requested language are skipped.

    Returns
    -------
    Iterator[Sample]
        An iterator over pairs of documents and their respective RVK subject annotations.
    """
    if qucosa_iterator is None:
        qucosa_iterator = read_qucosa_documents_from_directory()

    _make_doc_func_map = {
        "titles": _make_title_only_doc,
        "abstracts": _make_title_and_abstract_doc,
        "fulltexts": _make_title_and_fulltext_doc,
        "complete_but_only_titles": _make_filtered_complete_doc_generator(_make_title_only_doc),
        "complete_but_only_abstracts": _make_filtered_complete_doc_generator(_make_title_and_abstract_doc),
        "complete_but_only_fulltexts": _make_filtered_complete_doc_generator(_make_title_and_fulltext_doc),
    }

    _subject_getter_map = {
        "rvk": _get_rvk_subjects_from_qucosa_metadata,
        "ddc": _get_ddc_subjects_from_qucosa_metadata,
    }

    _subject_store_func_map = {
        "rvk": lambda: get_rvk_subject_store(),
        "ddc": lambda: get_qucosa_ddc_subject_store(),
    }

    return _read_qucosa_generic_samples(
        qucosa_iterator,
        _make_doc_func_map[metadata_variant],
        _subject_getter_map[subject_schema],
        _subject_store_func_map[subject_schema](),
        lang_code,
    )


def save_qucosa_simple_rvk_training_data_as_annif_tsv(
    qucosa_iterator: Iterable[QucosaJsonDocument] = None,
    filepath: str = None,
):
    """Save all qucosa documents as annif tsv file using only the title as text and RVK as subject annotations.

    Parameters
    ----------
    qucosa_iterator: Iterable[QucosaJsonDocument] = None
        An iterator over qucosa json documents. If None, tries to load them from default directory
        (and SLUB elasticsearch if not available).
    filepath: str = "<cache_dir>/qucosa/simple_training_data.tsv"
        The path to the TSV file that is being created

    Returns
    -------
    None
    """
    if filepath is None:
        filepath = os.path.join(get_cache_dir(), "qucosa/simple_training_data.tsv")
    if qucosa_iterator is None:
        qucosa_iterator = read_qucosa_documents_from_directory()

    # make sure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf8") as f_tsv:
            for doc, subjects in read_qucosa_samples(qucosa_iterator):
                text = doc.title
                labels_str = " ".join(map(lambda l: f"<{l}>", subjects))
                f_tsv.write(f"{text}\t{labels_str}\n")
