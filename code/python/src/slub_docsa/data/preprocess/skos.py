"""Convert between subject hierarchy and skos using rdflib."""

import logging
from typing import Any, Callable, List, Optional

import rdflib
from rdflib.namespace import SKOS, RDF

from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.data.load.rvk import generate_rvk_custom_skos_triples

logger = logging.getLogger(__name__)


def subject_hierarchy_to_skos_graph(
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
    language: str,
    generate_custom_triples: Optional[Callable[[SubjectNodeType], List[Any]]] = None
):
    """Convert subject hierarchy to an rdflib graph using SKOS triples."""
    logger.debug("convert subject hierarchy to rdflib skos graph")
    graph = rdflib.Graph()
    graph.namespace_manager.bind('skos', SKOS)
    i = 0
    for subject_uri in subject_hierarchy:
        subject_uri_ref = rdflib.URIRef(subject_hierarchy[subject_uri].uri)
        label = subject_hierarchy[subject_uri].label
        parent_uri = subject_hierarchy[subject_uri].parent_uri

        graph.add((subject_uri_ref, RDF.type, SKOS.Concept))
        graph.add((subject_uri_ref, SKOS.prefLabel, rdflib.Literal(label, language)))

        if generate_custom_triples is not None:
            for custom_triple in generate_custom_triples(subject_hierarchy[subject_uri]):
                graph.add(custom_triple)

        if parent_uri is not None:
            graph.add((subject_uri_ref, SKOS.broader, rdflib.URIRef(parent_uri)))

        i += 1

        if i % 10000 == 0:
            logger.debug("added %d subjects to graph so far", i)

    return graph


if __name__ == "__main__":
    import os
    from slub_docsa.data.load.rvk import get_rvk_subject_store
    from slub_docsa.common.paths import CACHE_DIR

    logging.basicConfig(level=logging.DEBUG)

    rvk_subject_hierarchy = get_rvk_subject_store()
    rvk_skos_graph = subject_hierarchy_to_skos_graph(rvk_subject_hierarchy, "de", generate_rvk_custom_skos_triples)

    rvk_skos_graph.serialize(destination=os.path.join(CACHE_DIR, "rvk/rvk_skos.ttl"), format="turtle")
