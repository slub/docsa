"""Parse and index k10plus data dump provided by SLUB Dresden."""

# pylint: disable=too-many-arguments, too-many-locals

import zipfile
import json
import logging
import os

from typing import Any, Iterable, Mapping, NamedTuple, Iterator, Optional

from sqlitedict import SqliteDict
from slub_docsa.common.document import Document

from slub_docsa.common.paths import get_cache_dir, get_resources_dir
from slub_docsa.common.sample import Sample
from slub_docsa.data.load.k10plus.samples import k10plus_samples_generator

logger = logging.getLogger(__name__)


class K10plusSlubJsonObject(NamedTuple):
    """Relevant information for each document from the k10plus data dump provided by SLUB."""

    ppn: str
    text_type: Optional[str]
    text: Optional[str]


def get_k10plus_slub_json_zip_filepath():
    """Return default filepath to k10plus data dump provided by SLUB."""
    return os.path.join(get_resources_dir(), "k10plus/slub/2022-11-21_slub_kxp_volltexte.ldj.zip")


def get_k10plus_slub_data_cache_filepath():
    """Return default filepath to index and cache k10plus data provided by SLUB."""
    return os.path.join(get_cache_dir(), "k10plus/slub/2022-11-21_slub_kxp_volltexte.sqlite")


def read_k10plus_slub_json_from_file(filepath: Optional[str] = None) -> Iterator[K10plusSlubJsonObject]:
    """Read k10plus data dump provided by SLUB."""
    if filepath is None:
        filepath = get_k10plus_slub_json_zip_filepath()

    with zipfile.ZipFile(filepath, "r") as f_zip:
        for filename in f_zip.namelist():
            with f_zip.open(filename, "r") as data:
                for line in data:
                    doc = json.loads(line)
                    yield K10plusSlubJsonObject(
                        ppn=doc["texts"]["id"],
                        text_type=doc["texts"].get("type"),
                        text=doc["texts"].get("Text")
                    )


def cache_and_index_k10plus_slub_data(
    json_filepath: Optional[str] = None,
    cache_filepath: Optional[str] = None,
):
    """Store k10plus data provided by SLUB indexed by PPN in Sqlite file."""
    if cache_filepath is None:
        cache_filepath = get_k10plus_slub_data_cache_filepath()
    os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)

    store = SqliteDict(cache_filepath + ".tmp", tablename="texts", flag="w", autocommit=False)

    for doc in read_k10plus_slub_json_from_file(json_filepath):
        if doc.text_type is not None and doc.text is not None:
            text_mapping = store.get(doc.ppn, {})
            text_mapping.update({
                doc.text_type: doc.text
            })
            store[doc.ppn] = text_mapping

    store.commit()
    store.close()

    os.rename(cache_filepath + ".tmp", cache_filepath)


def load_cached_k10plus_slub_data(
    json_filepath: Optional[str] = None,
    cache_filepath: Optional[str] = None,
) -> Mapping[str, Any]:
    """Load k10plus data provided by SLUB indexed by ppn."""
    if cache_filepath is None:
        cache_filepath = get_k10plus_slub_data_cache_filepath()

    if not os.path.exists(cache_filepath):
        cache_and_index_k10plus_slub_data(json_filepath, cache_filepath)

    return SqliteDict(cache_filepath, tablename="texts", flag="r")


def k10plus_slub_samples_generator(
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
    require_abstract: bool = False,
    require_toc: bool = False,
):
    """Read k10plus documents and combine them with fulltext information from SLUB data."""
    doc_count = 0
    slub_store = load_cached_k10plus_slub_data(slub_json_filepath, slub_cache_filepath)
    slub_ppns = set(slub_store.keys())
    samples_generator = k10plus_samples_generator(
        xml_directory, json_directory, languages, schemas, download, None, line_batch_size, workers
    )
    for sample in samples_generator:
        if limit is not None and doc_count >= limit:
            logger.debug("stop because of limit=%d", limit)
            return

        if sample.document.uri.startswith("ppn:"):
            ppn = sample.document.uri[4:]
            if ppn in slub_ppns:
                texts = slub_store.get(ppn)
                abstract = texts.get("Abstract", "").strip() or None
                toc = texts.get("Inhaltsverzeichnis", "").strip() or None

                if require_abstract and abstract is None:
                    continue

                if require_toc and toc is None:
                    continue

                doc_count += 1
                document = Document(
                    uri=sample.document.uri,
                    title=sample.document.title,
                    language=sample.document.language,
                    abstract=abstract,
                    toc=toc,
                    fulltext=None
                )
                yield Sample(document, sample.subjects)


def _print_occurances_of_text_types():
    slub_store = load_cached_k10plus_slub_data()
    count_by_type: Mapping[str, int] = {}
    for texts in slub_store.values():
        for text_type in texts.keys():
            count_by_type[text_type] = count_by_type.setdefault(text_type, 0) + 1
            if text_type == "Klappentext":
                print("----")
                print(texts[text_type].replace("\n", " ").strip())
                print("----")

    for text_type, count in reversed(sorted(count_by_type.items(), key=lambda x: x[1])):
        print(text_type, ":", count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("total slub documents:", sum(1 for _ in load_cached_k10plus_slub_data().keys()))

    count_by_language: Mapping[str, int] = {}
    for d, s in k10plus_slub_samples_generator(require_toc=True, schemas=["bk"]):
        count_by_language[d.language] = count_by_language.setdefault(d.language, 0) + 1

    print("by language", count_by_language)
    print("total", sum(count_by_language.values()))
