"""Load k10plus data as an iterator over samples."""

# pylint: disable=too-many-arguments, too-many-locals

import time
import logging

from typing import Any, Iterable, Iterator, Mapping, Optional, Set

from slub_docsa.common.document import Document
from slub_docsa.common.sample import Sample
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.load.k10plus.cache import k10plus_read_from_json_cache
from slub_docsa.data.load.languages import load_language_codes, convert_language_code_to_l3
from slub_docsa.data.load.subjects.common import subject_hierarchy_by_subject_schema

logger = logging.getLogger(__name__)


def k10plus_json_combined_language(json_document: Mapping[str, Any]) -> str:
    """Return provided language or detected language if not language is provided.

    Parameters
    ----------
    json_document :
        the json document as dictionary

    Returns
    -------
    the combined language code
    """
    if json_document["language"]["provided"]:
        return json_document["language"]["provided"]
    return json_document["language"]["detected"]


def k10plus_json_document_as_sample(
    json_document: Mapping[str, Any],
    languages: Set[str],
    schemas: Set[str],
    subject_hierarchies: Mapping[str, SubjectHierarchy],
    filter_unknown_subjects: bool = True,
) -> Optional[Sample]:
    """Parse json document and extract title and subjects.

    Parameters
    ----------
    json_document : Mapping[str, Any]
        the json document as dictionary
    languages : Set[str]
        the set of acceptable languages; if a document has a different language, None is returned
    schemas : Set[str]
        the set of acceptable subject schema; if a document has no subjects from these schemas, None is returned
    subject_hierarchies : Mapping[str, SubjectHierarchy]
        the subject hierarchies for all schema, which are used to check whether a subject URI refers to a known subject
        in the subject hierarchy if `filter_unknown_subjects` is True
    filter_unknown_subjects : bool, optional
        whether to filter subjects that are not known in the provided subject hierarchies, by default True

    Returns
    -------
    Optional[Sample]
        the sample or None, if the document is not acceptable regarding the provided language and schema restrictions
    """
    # check language
    language = k10plus_json_combined_language(json_document)
    if languages and language not in languages:
        return None

    # extract subjects
    subjects = []
    for schema in schemas:
        schema_subjects = json_document["subjects"][schema]
        if filter_unknown_subjects:
            schema_subjects = [s_uri for s_uri in schema_subjects if s_uri in subject_hierarchies[schema]]
        subjects.extend(schema_subjects)

    if schemas and not subjects:
        return None

    # extract title
    title = json_document["title"]
    if title is None:
        return None

    subtitle = json_document["subtitle"]
    if subtitle is not None:
        title += " - " + subtitle

    document = Document(
        uri="ppn:" + json_document["ppn"],
        language=language,
        title=title,
    )

    return Sample(document, subjects)


def k10plus_public_samples_generator(
    xml_directory: str = None,
    json_directory: str = None,
    languages: Optional[Iterable[str]] = None,
    schemas: Optional[Iterable[str]] = None,
    download: bool = True,
    limit: Optional[int] = None,
    line_batch_size: int = 1000,
    workers: Optional[int] = None,
    filter_unknown_subjects: bool = True
) -> Iterator[Sample]:
    """Return iterator over k10plus samples.

    Parameters
    ----------
    xml_directory : str, optional
        the directory where k10plus marc21 xml dump files are stored; if None, the directory
        `SLUB_RESOURCE_DIR/k10plus/marc21` is used
    json_directory : str, optional
        the directory where generated json cache files are stored at; if None, the directory
        `SLUB_CACHE_DIR/k10plus/json` is used
    languages : Optional[Iterable[str]], optional
        the set of acceptable languages or None, if all languages are acceptable
    schemas : Optional[Iterable[str]], optional
        the set of acceptable subject schema or None, if no subjects are required
    download : bool, optional
        whether to download k10plus marc21 xml dump files if they do not exist yet, by default True
    limit : Optional[int], optional
        the maximum amount of samples to return or None, if all possible samples are required, by default None
    line_batch_size : int, optional
        the number of lines that are read in one batch to improve performance, by default 1000
    workers : Optional[int], optional
        the number of worker threads to process marc21 xml files; if None, the number of available cpu cores is used
    filter_unknown_subjects : bool, optional
        whether to filter subjects that are not known in the provided subject hierarchies, by default True

    Yields
    ------
    Sample
        each k10plus document matching the above specified criteria for language, schema, etc., as a sample

    Raises
    ------
    ValueError
        if a schema is provided that is not supported
    """
    doc_count = 0
    last_log_time = time.time()

    languages = set(languages) if languages is not None else set()
    schemas = set(schemas) if schemas is not None else set()

    # check language codes are valid ISO 639
    language_code_table = load_language_codes()
    languages = {convert_language_code_to_l3(code, language_code_table) for code in languages}

    # check schemas are supported
    for schema in schemas:
        if schema not in ["rvk", "bk", "ddc", "gnd"]:
            raise ValueError(f"schema {schema} not supported")

    subject_hierarchies = {schema: subject_hierarchy_by_subject_schema(schema) for schema in schemas}

    # iterate over all cached json documents
    json_generator = k10plus_read_from_json_cache(xml_directory, json_directory, download, workers, line_batch_size)
    for json_document in json_generator:
        sample = k10plus_json_document_as_sample(
            json_document, languages, schemas, subject_hierarchies, filter_unknown_subjects
        )
        if sample:
            if limit is not None and doc_count >= limit:
                logger.debug("stop because of limit=%d", limit)
                return

            now_time = time.time()
            if now_time - last_log_time > 5:
                logger.info("read %d k10plus samples so far", doc_count)
                last_log_time = now_time

            doc_count += 1
            yield sample


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # print(sum(1 for _ in k10plus_samples_generator(languages=["de"])))

    count_by_language: Mapping[str, int] = {}
    for d, s in k10plus_public_samples_generator(schemas=["gnd"]):
        count_by_language[d.language] = count_by_language.setdefault(d.language, 0) + 1

    print("by language", count_by_language)
    print("total", sum(count_by_language.values()))
