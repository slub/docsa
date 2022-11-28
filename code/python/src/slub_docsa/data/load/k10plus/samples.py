"""Load k10plus data as iterator over samples."""

# pylint: disable=too-many-arguments, too-many-locals

import time
import logging

from typing import Iterable, Mapping, Optional, Set

from slub_docsa.common.document import Document
from slub_docsa.common.sample import Sample
from slub_docsa.data.load.k10plus.cache import k10plus_read_from_json_cache
from slub_docsa.data.load.languages import load_language_codes, convert_language_code_to_l3

logger = logging.getLogger(__name__)


def k10plus_json_combined_language(json_document):
    """Return provided language or detected language if not language is provided."""
    if json_document["language"]["provided"]:
        return json_document["language"]["provided"]
    return json_document["language"]["detected"]


def k10plus_json_document_as_sample(
    json_document,
    languages: Set[str],
    schemas: Set[str],
) -> Sample:
    """Parse json document and extract title and rvk classes."""
    # check language
    language = k10plus_json_combined_language(json_document)
    if languages and language not in languages:
        return None

    # extract subjects
    subjects = []
    if "rvk" in schemas:
        subjects.extend(json_document["subjects"]["rvk"])
    if "gnd" in schemas:
        subjects.extend(json_document["subjects"]["gnd"])
    if "bk" in schemas:
        subjects.extend(json_document["subjects"]["bk"])
    if "ddc" in schemas:
        subjects.extend(json_document["subjects"]["ddc"])

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


def k10plus_samples_generator(
    xml_directory: str = None,
    json_directory: str = None,
    languages: Optional[Iterable[str]] = None,
    schemas: Optional[Iterable[str]] = None,
    download: bool = True,
    limit: Optional[int] = None,
    line_batch_size: int = 1000,
    workers: Optional[int] = None,
):
    """Return iterator over k10plus samples."""
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

    # iterate over all cached json documents
    json_generator = k10plus_read_from_json_cache(xml_directory, json_directory, download, workers, line_batch_size)
    for json_document in json_generator:
        sample = k10plus_json_document_as_sample(json_document, languages, schemas)
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
    for d, s in k10plus_samples_generator(schemas=["gnd"]):
        count_by_language[d.language] = count_by_language.setdefault(d.language, 0) + 1

    print("by language", count_by_language)
    print("total", sum(count_by_language.values()))
