"""Convert between subject hierarchy and skos using rdflib."""

import logging
from typing import Any, Callable, List, Optional, Sequence

import rdflib
from rdflib.namespace import SKOS, RDF

from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType
from slub_docsa.data.load.rvk import generate_rvk_custom_skos_triples
from slub_docsa.data.preprocess.subject import subject_ancestors_list

logger = logging.getLogger(__name__)


def subject_hierarchy_to_skos_graph(
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
    language: str,
    generate_custom_triples: Optional[Callable[[SubjectNodeType], List[Any]]] = None,
    mandatory_subject_list: Sequence[str] = None,
):
    """Convert subject hierarchy to an rdflib graph using SKOS triples."""
    logger.debug("convert subject hierarchy to rdflib skos graph")
    graph = rdflib.Graph()
    graph.namespace_manager.bind('skos', SKOS)

    # decide whether to iterate over full hierarchy or just mandatory subjects + ancestors
    subject_iterator = iter(subject_hierarchy)
    if mandatory_subject_list is not None:
        subject_set = set()
        for subject_uri in mandatory_subject_list:
            if subject_uri in subject_hierarchy:
                subject_node = subject_hierarchy[subject_uri]
                subject_set.update([n.uri for n in subject_ancestors_list(subject_node, subject_hierarchy)])
            else:
                logger.warning("subject %s is mandatory, but can not be found in subject hierarchy", subject_uri)
        subject_iterator = iter(subject_set)

    i = 0
    for subject_uri in subject_iterator:
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
