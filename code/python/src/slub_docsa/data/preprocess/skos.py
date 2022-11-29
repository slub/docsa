"""Convert between subject hierarchy and skos using rdflib."""

import logging
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import rdflib
from rdflib.namespace import SKOS, RDF

from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.load.subjects.rvk import generate_rvk_custom_skos_triples
from slub_docsa.data.preprocess.subject import subject_ancestors_list

logger = logging.getLogger(__name__)


def subject_labels_to_skos_graph(
    subject_labels: Mapping[str, str],
    lang_code: str,
) -> rdflib.Graph:
    """Convert subject labels without hierarchical relationships to an rdflib SKOS graph.

    Parameters
    ----------
    subject_labels: Mapping[str, str]
        a map containing labels for each subject
    lang_code: str
        a two-letter language code representing the language of all labels

    Returns
    -------
    rdflib.Graph
        an rdflib graph containing all subjects as SKOS concepts and their labels as `prefLabel`
    """
    graph = rdflib.Graph()
    graph.namespace_manager.bind('skos', SKOS)

    for uri, label in subject_labels.items():
        subject_uri_ref = rdflib.URIRef(uri)
        graph.add((subject_uri_ref, RDF.type, SKOS.Concept))
        graph.add((subject_uri_ref, SKOS.prefLabel, rdflib.Literal(label, lang_code)))

    return graph


def subject_hierarchy_to_skos_graph(
    subject_hierarchy: SubjectHierarchy,
    lang_code: str,
    generate_custom_triples: Optional[Callable[[Any], List[Tuple[Any, Any, Any]]]] = None,
    mandatory_subject_list: Optional[Sequence[str]] = None,
) -> rdflib.Graph:
    """Convert a subject hierarchy to an rdflib graph using SKOS triples.

    Parameters
    ----------
    subject_hierarchy: SubjectHierarchy
        the subject hierarchy that is being converted to a SKOS rdflib graph
    lang_code: str
        a two-letter language code represeting the language of labels in the subject hierarchy
    generate_custom_triples: Optional[Callable[[str, SubjectHierarchy], List[Tuple[Any, Any, Any]]]] = None
        an optional function that returns additional triples given a specific subject URI and the subject hierarchy
    mandatory_subject_list: Sequence[str] = None
        an optional list of subjects; if provided, only these subjects and their ancestors are converted to SKOS,
        otherwise all subjects of the hierarchy are added

    Returns
    -------
    rdflib.Graph
        an rdflib graph that represents the subject hieararchy with SKOS triples
    """
    logger.debug("convert subject hierarchy to rdflib skos graph")
    graph = rdflib.Graph()
    graph.namespace_manager.bind('skos', SKOS)

    # decide whether to iterate over full hierarchy or just mandatory subjects + ancestors
    subject_iterator = iter(subject_hierarchy)
    if mandatory_subject_list is not None:
        subject_set = set()
        for subject_uri in mandatory_subject_list:
            if subject_uri in subject_hierarchy:
                subject_set.update([n.uri for n in subject_ancestors_list(subject_uri, subject_hierarchy)])
            else:
                logger.warning("subject %s is mandatory, but can not be found in subject hierarchy", subject_uri)
        subject_iterator = iter(subject_set)

    i = 0
    for subject_uri in subject_iterator:
        subject_uri_ref = rdflib.URIRef(subject_uri)
        label = subject_hierarchy.subject_labels(subject_uri).get(lang_code)
        parent_uri = subject_hierarchy.subject_parent(subject_uri)

        graph.add((subject_uri_ref, RDF.type, SKOS.Concept))
        graph.add((subject_uri_ref, SKOS.prefLabel, rdflib.Literal(label, lang_code)))

        if generate_custom_triples is not None:
            for custom_triple in generate_custom_triples(subject_uri, subject_hierarchy):
                graph.add(custom_triple)

        if parent_uri is not None:
            graph.add((subject_uri_ref, SKOS.broader, rdflib.URIRef(parent_uri)))

        i += 1

        if i % 10000 == 0:
            logger.debug("added %d subjects to graph so far", i)

    return graph


if __name__ == "__main__":
    import os
    from slub_docsa.data.load.subjects.rvk import load_rvk_subject_hierarchy_from_sqlite
    from slub_docsa.common.paths import get_cache_dir

    logging.basicConfig(level=logging.DEBUG)

    depth = 2  # pylint: disable=invalid-name
    rvk_subject_hierarchy = load_rvk_subject_hierarchy_from_sqlite(depth=depth)
    rvk_skos_graph = subject_hierarchy_to_skos_graph(rvk_subject_hierarchy, "de", generate_rvk_custom_skos_triples)

    rvk_skos_graph.serialize(
        destination=os.path.join(get_cache_dir(), f"rvk/rvk_skos_depth={depth}.ttl"),
        format="turtle"
    )
