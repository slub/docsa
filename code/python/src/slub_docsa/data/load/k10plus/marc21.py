"""Loads k10plus data from various sources."""

import logging
import time
import re
import xml.etree.ElementTree as ET  # nosec

from itertools import islice
from typing import Any, Callable, Mapping, Optional, Sequence
from slub_docsa.data.load.subjects.bk import bk_notation_to_uri
from slub_docsa.data.load.subjects.ddc import ddc_notation_to_uri, ddc_reduce_notation_to_short_notation
from slub_docsa.data.load.subjects.rvk import rvk_notation_to_uri
from slub_docsa.data.load.subjects.gnd import gnd_notation_to_uri
from slub_docsa.data.load.languages import LanguageCodeTable
from slub_docsa.data.load.common import read_gzip_text_lines

logger = logging.getLogger(__name__)

NAMESPACE = {"marc": "http://www.loc.gov/MARC21/slim"}

UNKNOWN_LANGUAGE_CODES = set([
    "qry", "qmo", "scr", "qmw", "qoj", "qev", "qqa", "qnv",
    "fer", "qte", "qkj", "qce", "qlm", "scc", "qkc", "qqg",
    "qnn", "snh", "qju", "abs", "tag", "jap", "iri", "law",
    "frz", "ned", "fra", "gae", "neg", "deu", "gag", "enk",
    "max", "xxx", "aar", "hi0", "---", "esp", "bes", "lan",
    "tar", "sra", "miq", "gal", "pio", "sro", "tpc", "smu",
    "lko", "toc", "lar", "enh", "mzt"
])

IGNORED_LANGUAGE_CODES = [
    "zxx",  # no linguistic content
    "und",  # undetermined
    "mul",  # multiple
]


def _extract_rvk_subjects(root_xml: ET.Element) -> Sequence[str]:
    """Extract rvk labels from MARC21 xml documents of k10plus files."""
    labels = []
    for annotation in root_xml.findall("./marc:datafield[@tag='084']", NAMESPACE):
        annotation_type = annotation.find("marc:subfield[@code='2']", NAMESPACE)
        if annotation_type is not None and str(annotation_type.text) == "rvk":
            rvk_annotation = annotation.find("marc:subfield[@code='a']", NAMESPACE)
            if rvk_annotation is not None and rvk_annotation.text is not None:
                labels.append(rvk_notation_to_uri(rvk_annotation.text))

    return labels


def _extract_gnd_subjects(root_xml: ET.Element) -> Sequence[str]:
    """Extract gnd labels from MARC21 xml documents of k10plus files."""
    labels = []
    for annotation in root_xml.findall("./marc:datafield[@tag='689']", NAMESPACE):
        annotation_type = annotation.find("marc:subfield[@code='2']", NAMESPACE)
        if annotation_type is not None and str(annotation_type.text) == "gnd":
            for code in annotation.findall("marc:subfield[@code='0']", NAMESPACE):
                if code.text is not None and code.text.startswith("(DE-588)"):
                    gnd_code = code.text.replace("(DE-588)", "")
                    if gnd_code:
                        labels.append(gnd_notation_to_uri(gnd_code))

    return labels


def _extract_bk_subjects(root_xml: ET.Element) -> Sequence[str]:
    """Extract base classification subjects from MARC21 xml documents of k10plus files."""
    labels = []
    for annotation in root_xml.findall("./marc:datafield[@tag='936']", NAMESPACE):
        code = annotation.find("marc:subfield[@code='a']", NAMESPACE)
        if code is not None and code.text is not None and re.match(r"^\d\d\.\d\d", code.text):
            labels.append(bk_notation_to_uri(code.text))
    return labels


def _extract_ddc_subjects(root_xml: ET.Element) -> Sequence[str]:
    """Extract DDC subjects from MARC21 xml document of k10plus files."""
    labels = set()
    for annotation in root_xml.findall("./marc:datafield[@tag='082']", NAMESPACE):
        code = annotation.find("marc:subfield[@code='a']", NAMESPACE)
        if code is not None and code.text is not None:
            labels.add(ddc_notation_to_uri(ddc_reduce_notation_to_short_notation(code.text)))
    return list(labels)


def _extract_k10plus_marc_xml_title(root: ET.Element) -> Optional[str]:
    """Extract title from k10plus marc xml file."""
    # extract title
    title_element = root.find("./marc:datafield[@tag='245']/marc:subfield[@code='a']", NAMESPACE)
    if title_element is None or not str(title_element.text):
        return None
    return str(title_element.text)


def _extract_k10plus_marc_xml_subtitle(root: ET.Element) -> Optional[str]:
    """Extract subtitle from k10plus marc xml file."""
    subtitle_element = root.find("./marc:datafield[@tag='245']/marc:subfield[@code='b']", NAMESPACE)
    if subtitle_element is not None and subtitle_element.text is not None:
        return str(subtitle_element.text)
    return None


def _extract_k10plus_marc_xml_language(root: ET.Element, language_code_table: LanguageCodeTable) -> Optional[str]:
    """Extract language from k10plus marc xml file."""
    # extract language from tag 008
    control_element = root.find("./marc:controlfield[@tag='008']", NAMESPACE)
    control_language = None
    if control_element is not None and control_element.text is not None:
        control_language = control_element.text[35:38].lower()

    # extract language from tag 041
    datafield_elements = root.findall("./marc:datafield[@tag='041']/marc:subfield[@code='a']", NAMESPACE)
    datafield_language = None
    if len(datafield_elements) == 1 and datafield_elements[0].text is not None:
        datafield_language = datafield_elements[0].text.lower()

    # check both languages match
    if control_language == datafield_language:
        if datafield_language in language_code_table.by_l3:
            return datafield_language
        if datafield_language not in IGNORED_LANGUAGE_CODES and datafield_language not in UNKNOWN_LANGUAGE_CODES:
            logger.debug("k10plus language '%s' not available in language map", datafield_language)
    return None


def _extract_k10plus_marc_xml_pica_ppn(root: ET.Element) -> Optional[str]:
    control_element = root.find("./marc:controlfield[@tag='001']", NAMESPACE)
    if control_element is not None and control_element.text is not None:
        return control_element.text
    return None


def parse_single_k10plus_marc21_xml_file(
    filepath: str,
    line_batch_size: int = 1000,
) -> str:
    """Read single marc21 xml documents (as strings) from a k10plus xml dump file.

    Parameters
    ----------
    filepath : str
        the filepath to the k10plus marc21 xml file
    line_batch_size : int, optional
        the number of lines that are read in one batch to improve performance

    Yields
    ------
    str
        a str containing the xml of a single marc21 document
    """
    document_lines = []
    line_count = 0
    last_log_time = time.time()
    last_log_line_count = 0

    logger.debug("iterate over k10plus documents of file %s", filepath)
    line_generator = read_gzip_text_lines(filepath)
    while True:
        chunk = list(islice(line_generator, line_batch_size))

        if not chunk:
            break

        for line in chunk:
            line_count += 1

            if line.startswith("<record"):
                document_lines = []

            document_lines.append(line)

            if line.startswith("</record"):
                # detected end of document
                document_xml = "".join(document_lines)

                now_time = time.time()
                if now_time - last_log_time > 5:
                    lines_per_second = (line_count - last_log_line_count) / (now_time - last_log_time)
                    logger.info("k10plus xml %f lines per sec", lines_per_second)
                    last_log_time = now_time
                    last_log_line_count = line_count

                yield document_xml


def parse_k10plus_marc_xml_to_json(
    document_xml: str,
    language_detector: Callable[[str], Optional[str]],
    language_code_table: LanguageCodeTable,
) -> Mapping[str, Any]:
    """Parse relevant marc21 xml information and construct dictionary object to be saved as json.

    Parameters
    ----------
    document_xml : str
        the marc21 xml document as string
    language_detector : Callable[[str], Optional[str]]
        a language detection method that is used in case the marc21 document doesn't contain
        any language information about the document

    Returns
    -------
    Mapping[str, any]
        A dictionary of relevant information extracted from the marc21 xml document
    """
    root = ET.fromstring(document_xml)

    title = _extract_k10plus_marc_xml_title(root)
    subtitle = _extract_k10plus_marc_xml_subtitle(root)

    language_detection_text = None
    if title:
        language_detection_text = title

    if title and subtitle:
        language_detection_text = title + " - " + subtitle

    rvk_subjects = _extract_rvk_subjects(root)
    gnd_subjects = _extract_gnd_subjects(root)
    bk_subjects = _extract_bk_subjects(root)
    ddc_subjects = _extract_ddc_subjects(root)

    provided_language = _extract_k10plus_marc_xml_language(root, language_code_table)
    ppn = _extract_k10plus_marc_xml_pica_ppn(root)

    detected_language = None
    if language_detection_text is not None and provided_language is None:
        detected_language = language_detector(language_detection_text)
        if detected_language in language_code_table.by_l2:
            detected_language = language_code_table.by_l2[detected_language].l3
        if detected_language is not None and detected_language not in language_code_table.by_l3:
            logger.debug("detected language %s unkown", detected_language)

    return {
        "ppn": ppn,
        "title": title,
        "subtitle": subtitle,
        "language": {
            "provided": provided_language,
            "detected": detected_language,
        },
        "subjects": {
            "rvk": rvk_subjects,
            "gnd": gnd_subjects,
            "bk": bk_subjects,
            "ddc": ddc_subjects,
        },
    }
