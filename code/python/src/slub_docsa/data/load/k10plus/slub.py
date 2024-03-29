"""Parse and load k10plus data provided by SLUB Dresden."""

# pylint: disable=too-many-arguments, too-many-locals

import json
import logging
import os
import tarfile
import codecs
import re
import time

from typing import Callable, Iterable, Mapping, NamedTuple, Iterator, Optional

from sqlitedict import SqliteDict
from slub_docsa.common.document import Document

from slub_docsa.common.paths import get_cache_dir, get_resources_dir
from slub_docsa.common.sample import Sample
from slub_docsa.data.load.languages import LanguageCodeTable, convert_language_code_to_l3, load_language_codes
from slub_docsa.data.load.subjects.common import subject_hierarchy_by_subject_schema
from slub_docsa.data.load.subjects.rvk import rvk_notation_to_uri
from slub_docsa.data.load.subjects.ddc import ddc_notation_to_uri, ddc_reduce_notation_to_short_notation
from slub_docsa.data.load.subjects.bk import bk_notation_to_uri
from slub_docsa.data.load.k10plus.samples import k10plus_public_samples_generator
from slub_docsa.data.preprocess.language import load_fasttext_language_detector

logger = logging.getLogger(__name__)


class K10plusSlubJsonTitles(NamedTuple):
    """Relevant title information for each document from the k10plus data dump provided by SLUB."""

    full: str
    main: str
    sub: str
    part: str
    responsibility: str


class K10plusSlubJsonObject(NamedTuple):
    """Relevant information for each document from the k10plus data dump provided by SLUB."""

    ppn: str
    titles: K10plusSlubJsonTitles
    classifications: Mapping[str, Iterable[str]]
    texts: Mapping[str, str]
    language: str


def get_k10plus_slub_json_zip_filepath():
    """Return default filepath to k10plus data dump provided by SLUB."""
    return os.path.join(get_resources_dir(), "k10plus/slub/2022-11-21_slub_kxp_volltexte.ldj.zip")


def get_k10plus_slub_json_tar_gz_filepath():
    """Return default filepath to k10plus data dump provided by SLUB."""
    return os.path.join(get_resources_dir(), "k10plus/slub/swbplus_fulltexts_20221130.tar.gz")


def get_k10plus_slub_data_cache_filepath():
    """Return default filepath to index and cache k10plus data provided by SLUB."""
    return os.path.join(get_cache_dir(), "k10plus/slub/slub_k10plus_daten.sqlite")


def _convert_classifications_to_uris(classifications: Mapping[str, Iterable[str]]) -> Mapping[str, Iterable[str]]:
    """Convert subject notations to full subject URIs for all available subjects.

    Parameters
    ----------
    classifications : Mapping[str, Iterable[str]]
        the map of subjects indexed by schema

    Returns
    -------
    Mapping[str, Iterable[str]]
        the converted map of subjects such that each subject is represented by its subject URI
    """
    if "rvk" in classifications:
        classifications["rvk"] = [rvk_notation_to_uri(notation) for notation in classifications["rvk"]]
    if "ddc" in classifications:
        classifications["ddc"] = [
            ddc_notation_to_uri(ddc_reduce_notation_to_short_notation(notation)) for notation in classifications["ddc"]
        ]
    if "bkl" in classifications:
        classifications["bk"] = [bk_notation_to_uri(notation) for notation in classifications["bkl"]]
        del classifications["bkl"]
    return classifications


def read_k10plus_slub_json_from_file(filepath: Optional[str] = None) -> Iterator[K10plusSlubJsonObject]:
    """Read k10plus data dump file provided by SLUB.

    Parameters
    ----------
    filepath : Optional[str], optional
        the filepath to the k10plus data dump provided by SLUB

    Yields
    ------
    K10plusSlubJsonObject
        parsed json documents as `K10plusSlubJsonObject`
    """
    if filepath is None:
        filepath = get_k10plus_slub_json_tar_gz_filepath()

    with tarfile.open(filepath, "r|gz") as tar_file:
        for member in tar_file:
            json_file = codecs.getreader("utf-8")(tar_file.extractfile(member))
            for line in json_file:
                doc = json.loads(line)
                yield K10plusSlubJsonObject(
                    ppn=doc["id"],
                    titles=K10plusSlubJsonTitles(
                        full=doc["title"]["full"],
                        main=doc["title"]["main"],
                        sub=doc["title"]["sub"],
                        part=doc["title"]["part"],
                        responsibility=doc["title"]["responsibility"]
                    ),
                    classifications=_convert_classifications_to_uris(doc["classifications"]),
                    texts={text_type: text[0] for text_type, text in doc["texts"].items()},
                    language=None
                )


def _clean_text_for_language_detection(text):
    """Clean text for language detection by removing most non-letter characters from the text."""
    tokens = text.replace("\n", ". ").replace(r"\s+", " ").split(" ")
    return " ".join(token for token in tokens if re.match(r"\w+", token) and len(token) > 2)


def _detect_language(
    doc: K10plusSlubJsonObject,
    language_detector: Callable[[str], str],
    language_code_table: LanguageCodeTable
) -> Optional[str]:
    """Detect the language of a document of the k10plus dump provided by SLUB."""
    # first try to detect language from title
    combined_titles = doc.titles.main + " " + doc.titles.sub
    detected_language = convert_language_code_to_l3(
        language_detector(combined_titles),
        language_code_table,
        raise_invalid=False
    )
    # if language detection based on title didn't work, use other texts (e.g. the toc) of the document
    if not detected_language:
        cleaned_texts = _clean_text_for_language_detection(" ".join(doc.texts.values())[:500])
        detected_language = convert_language_code_to_l3(
            language_detector(cleaned_texts),
            language_code_table,
            raise_invalid=False
        )
    return detected_language


def cache_and_index_k10plus_slub_data(
    json_filepath: Optional[str] = None,
    cache_filepath: Optional[str] = None,
    language_detection_certainty: float = 0.5,
):
    """Store k10plus data provided by SLUB indexed by PPN in an sqlite file.

    Parameters
    ----------
    json_filepath : Optional[str], optional
        the filepath to the json file provided by SLUB; if None, uses the default path that is returned by
        `get_k10plus_slub_json_tar_gz_filepath`
    cache_filepath : Optional[str], optional
        the filepath to the generated cache sqlite database; if None, uses the default path that is returned by
        `get_k10plus_slub_data_cache_filepath`
    language_detection_certainty : float, optional
        the minimum certainy score that is used to assume that the language detection was successful, by default 0.5
    """
    if cache_filepath is None:
        cache_filepath = get_k10plus_slub_data_cache_filepath()
    os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)

    store = SqliteDict(cache_filepath + ".tmp", tablename="k10plus", flag="w", autocommit=False)
    language_detector = load_fasttext_language_detector(language_detection_certainty)
    language_code_table = load_language_codes()

    for doc in read_k10plus_slub_json_from_file(json_filepath):
        store[doc.ppn] = K10plusSlubJsonObject(
            ppn=doc.ppn,
            titles=doc.titles,
            classifications=doc.classifications,
            texts=doc.texts,
            language=_detect_language(doc, language_detector, language_code_table)
        )

    store.commit()
    store.close()

    os.rename(cache_filepath + ".tmp", cache_filepath)


def load_cached_k10plus_slub_data(
    json_filepath: Optional[str] = None,
    cache_filepath: Optional[str] = None,
    language_detection_certainty: float = 0.5,
) -> Mapping[str, K10plusSlubJsonObject]:
    """Load cached Sqlite database containing k10plus data provided by SLUB indexed by ppn.

    If the Sqlite database doesn't exist, it will be created from the k10plus data json dump provided by SLUB.

    Parameters
    ----------
    json_filepath : Optional[str], optional
        the filepath to the json file provided by SLUB; if None, uses the default path that is returned by
        `get_k10plus_slub_json_tar_gz_filepath`
    cache_filepath : Optional[str], optional
        the filepath to the generated cache sqlite database; if None, uses the default path that is returned by
        `get_k10plus_slub_data_cache_filepath`
    language_detection_certainty : float, optional
        the minimum certainy score that is used to assume that the language detection was successful, by default 0.5

    Returns
    -------
    Mapping[str, K10plusSlubJsonObject]
        the sqlite database as dictionary mapping ppn numbers to all information about a document
    """
    if cache_filepath is None:
        cache_filepath = get_k10plus_slub_data_cache_filepath()

    if not os.path.exists(cache_filepath):
        cache_and_index_k10plus_slub_data(json_filepath, cache_filepath, language_detection_certainty)

    return SqliteDict(cache_filepath, tablename="k10plus", flag="r")


def k10plus_slub_merged_with_public_samples_generator(
    xml_directory: str = None,
    json_directory: str = None,
    slub_json_filepath: Optional[str] = None,
    slub_cache_filepath: Optional[str] = None,
    languages: Optional[Iterable[str]] = None,
    schemas: Optional[Iterable[str]] = None,
    download: bool = True,
    limit: Optional[int] = None,
    line_batch_size: int = 1000,
    workers: Optional[int] = None,
    require_toc: bool = True,
    use_slub_subjects: bool = True,
) -> Iterator[Sample]:
    """Read k10plus public documents and combine them with fulltext information from SLUB data dump.

    Parameters
    ----------
    xml_directory : str, optional
        the directory where the public k10plus marc21 xml dump files are stored; if None, the directory
        `SLUB_RESOURCE_DIR/k10plus/marc21` is used
    json_directory : str, optional
        the directory where the public k10plus json cache files are stored at; if None, the directory
        `SLUB_CACHE_DIR/k10plus/json` is used
    slub_json_filepath : Optional[str], optional
        the filepath to the json file provided by SLUB; if None, uses the default path that is returned by
        `get_k10plus_slub_json_tar_gz_filepath`
    slub_cache_filepath : Optional[str], optional
        the filepath to the generated cache sqlite database containing k10plus documents provided by SLUB; if None,
        uses the default path that is returned by `get_k10plus_slub_data_cache_filepath`
    languages : Optional[Iterable[str]], optional
        the set of acceptable languages or None, if all languages are acceptable
    schemas : Optional[Iterable[str]], optional
        the set of acceptable subject schema or None, if no subjects are required
    download : bool, optional
        whether to download public k10plus marc21 xml dump files if they do not exist yet, by default True
    limit : Optional[int], optional
        the maximum amount of samples to return or None, if all possible samples are required, by default None
    line_batch_size : int, optional
        the number of lines that are read in one batch to improve performance, by default 1000
    workers : Optional[int], optional
        the number of worker threads to process marc21 xml files; if None, the number of available cpu cores is used
    require_toc : bool, optional
        whether to only return samples for documents that have a ToC provided from the SLUB data dump, by default True
    use_slub_subjects : bool, optional
        whether to return subjects extracted from the SLUB data dump, or from the public k10plus data, by default True

    Yields
    ------
    Sample
        each k10plus document matching the above specified criteria for language, schema, etc., as a sample
    """
    doc_count = 0
    slub_store = load_cached_k10plus_slub_data(slub_json_filepath, slub_cache_filepath)
    slub_ppns = set(slub_store.keys())
    samples_generator = k10plus_public_samples_generator(
        xml_directory, json_directory, None, schemas, download, None, line_batch_size, workers
    )
    languages = set(languages) if languages is not None else set()
    schemas = set(schemas) if schemas is not None else set()
    for sample in samples_generator:
        if limit is not None and doc_count >= limit:
            logger.debug("stop because of limit=%d", limit)
            return

        if sample.document.uri.startswith("ppn:"):
            ppn = sample.document.uri[4:]
            if ppn in slub_ppns:
                slub_doc = slub_store.get(ppn)

                # combine language
                language = sample.doc.language or slub_doc.language
                if languages and language not in languages:
                    continue

                # extract and check toc
                toc = slub_doc.texts.get("toc", "").strip() or None
                if require_toc and toc is None:
                    continue

                doc_count += 1
                document = Document(
                    uri=sample.document.uri,
                    title=sample.document.title,
                    language=language,
                    abstract=None,
                    toc=toc,
                    fulltext=None
                )

                # combine and check subjects
                if use_slub_subjects:
                    subjects = sum([slub_doc.classifications.get(schema, []) for schema in schemas], [])
                else:
                    subjects = sample.subjects

                if schemas and not subjects:
                    continue

                yield Sample(document, subjects)


def k10plus_slub_samples_generator(
    json_filepath: Optional[str] = None,
    cache_filepath: Optional[str] = None,
    languages: Optional[Iterable[str]] = None,
    schemas: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    require_toc: bool = True,
    filter_unknown_subjects: bool = True,
    clean_toc: bool = False,
    titles_only: bool = False,
) -> Iterator[Sample]:
    """Read k10plus documents including fulltext information as provided from the SLUB data dump.

    Parameters
    ----------
    json_filepath : Optional[str], optional
        the filepath to the json file provided by SLUB; if None, uses the default path that is returned by
        `get_k10plus_slub_json_tar_gz_filepath`
    cache_filepath : Optional[str], optional
        the filepath to the generated cache sqlite database containing k10plus documents provided by SLUB; if None,
        uses the default path that is returned by `get_k10plus_slub_data_cache_filepath`
    languages : Optional[Iterable[str]], optional
        the set of acceptable languages or None, if all languages are acceptable
    schemas : Optional[Iterable[str]], optional
        the set of acceptable subject schema or None, if no subjects are required
    limit : Optional[int], optional
        the maximum amount of samples to return or None, if all possible samples are required, by default None
    require_toc : bool, optional
        whether to only return samples for documents that have a ToC provided from the SLUB data dump, by default True
    filter_unknown_subjects : bool, optional
        whether to filter subjects that are not known in the provided subject hierarchies, by default True
    clean_toc: bool, optional
        whether to clean the table of contents text by removing any non-letter characters, by default False
    titles_only: bool, optional
        whether to only return the title of the document (and no toc), by default False

    Yields
    ------
    Iterator[Sample]
        each k10plus document matching the above specified criteria for language, schema, etc., as a sample
    """
    doc_count = 0
    last_log_time = time.time()
    slub_store = load_cached_k10plus_slub_data(json_filepath, cache_filepath)

    languages = set(languages) if languages is not None else set()
    schemas = set(schemas) if schemas is not None else set()
    subject_hierarchies = {schema: subject_hierarchy_by_subject_schema(schema) for schema in schemas}

    language_code_table = load_language_codes()
    languages = {convert_language_code_to_l3(code, language_code_table) for code in languages}
    for ppn in slub_store:
        if limit is not None and doc_count >= limit:
            logger.debug("stop because of limit=%d", limit)
            return

        doc = slub_store[ppn]
        toc = doc.texts.get("toc", "").strip() or None

        if clean_toc and toc is not None:
            toc = _clean_text_for_language_detection(toc)

        # skip document if it does not have a toc
        if require_toc and toc is None:
            continue

        # skip document if language was not requested
        if languages and doc.language not in languages:
            continue

        title = doc.titles.main
        if doc.titles.sub:
            title += " - " + doc.titles.sub

        # skip document if it does not have a title
        if not title:
            logger.warning("skip doc wit ppn=%s because it has no title", ppn)
            continue

        # compile sample document
        document = Document(
            uri="ppn:" + ppn,
            title=title,
            language=doc.language,
            abstract=None,
            toc=None if titles_only else toc,
            fulltext=None
        )

        # extract requested classification information
        subjects = []
        for schema in schemas:
            schema_subjects = doc.classifications.get(schema, [])
            if filter_unknown_subjects:
                schema_subjects = [s_uri for s_uri in schema_subjects if s_uri in subject_hierarchies[schema]]
            subjects.extend(schema_subjects)

        # skip documents that have not classification annotations for requested schemas
        if schemas and not subjects:
            continue

        now_time = time.time()
        if now_time - last_log_time > 5:
            logger.info("read %d k10plus slub samples so far", doc_count)
            last_log_time = now_time

        doc_count += 1
        yield Sample(document, subjects)


def _print_occurances_of_text_types():
    """Print the number of k10plus documents that contain a fulltext of some type (e.g. toc)."""
    slub_store = load_cached_k10plus_slub_data()
    count_by_type: Mapping[str, int] = {}
    for doc in slub_store.values():
        for text_type in doc.classifications.keys():
            if doc.classifications.get(text_type):
                count_by_type[text_type] = count_by_type.setdefault(text_type, 0) + 1

    for text_type, count in reversed(sorted(count_by_type.items(), key=lambda x: x[1])):
        print(text_type, ":", count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # print("total slub documents:", sum(1 for _ in load_cached_k10plus_slub_data().keys()))
    # _print_occurances_of_text_types()

    count_by_language: Mapping[str, int] = {}
    for d, s in k10plus_slub_samples_generator(require_toc=True, schemas=["bk"]):
        count_by_language[d.language] = count_by_language.setdefault(d.language, 0) + 1

    print("by language", count_by_language)

    print(
        sum(count_by_language.values()), "|",
        count_by_language["ger"], "|",
        count_by_language["eng"], "|",
        count_by_language["ita"], "|",
        count_by_language["fre"], "|",
        count_by_language["spa"], "|",
        count_by_language[None]
    )
