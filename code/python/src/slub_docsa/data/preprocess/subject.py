"""Methods for pre-processing subjects, e.g., pruning."""

import logging

from typing import List, Mapping, Set

from slub_docsa.common.subject import SubjectHierarchy, SubjectNode, SubjectTargets, SubjectUriList

logger = logging.getLogger(__name__)


def subject_ancestors_list(
    subject: SubjectNode,
    subject_hierarchy: SubjectHierarchy,
) -> List[SubjectNode]:
    """Return the list of ancestors for a subject node in a subject hierarchy (including the subject itself).

    Parameters
    ----------
    subject: SubjectNode
        the subjet whose ancestors are being returned
    subject_hierarchy: SubjectHierarchy
        the subject hierarchy which is being searched for ancestors of the specified subject

    Returns
    -------
    List[SubjectNode]
        the list of ancestors (starting with the root ancestors, ending with and including the subject itself)
    """
    ancestors: List[SubjectNode] = []
    next_subject = subject

    while next_subject is not None:
        ancestors.insert(0, next_subject)
        if next_subject.parent_uri is not None:
            next_subject = subject_hierarchy.get(next_subject.parent_uri, None)
        else:
            next_subject = None

    return ancestors


def subject_siblings_list(
    subject_node: SubjectNode,
    subject_hierarchy: SubjectHierarchy,
) -> List[SubjectNode]:
    """Return the list of siblings for a subject node in a subject hierarchy.

    Parameters
    ----------
    subject_node: SubjectNode
        the subject whose siblings are being returned
    subject_hierarchy: SubjectHierarchy
        the subject hierarchy that is searched for siblings of the specified subject

    Returns
    -------
    List[SubjectNode]
        the list of siblings in arbitrary order, including the subject itself
    """
    siblings: List[SubjectNode] = []

    for siblings_node in subject_hierarchy.values():
        if siblings_node.parent_uri == subject_node.parent_uri:
            siblings.append(siblings_node)

    return siblings


def subject_label_breadcrumb(
    subject: SubjectNode,
    subject_hierarchy: SubjectHierarchy,
) -> str:
    """Return a label breadcrumb as a string describing the subject hierarchy path from root to this subject.

    Parameters
    ----------
    subject: SubjectNode
        the subject whose ancestor path (breadcrumb) is supposed to be returned
    subject_hierarchy: SubjectHierarchy
        the subject hierarchy that is searched for ancestors of the specified subject

    Returns
    -------
    str
        a simple breadcrumb string (e.g. subject1 | subject2 | subject3) describing the ancestor path of the subject
    """
    subject_path = subject_ancestors_list(subject, subject_hierarchy)
    return " | ".join(map(lambda s: s.label, subject_path))


def prune_subject_uri_to_level(
    level: int,
    subject_uri: str,
    subject_hierarchy: SubjectHierarchy
) -> str:
    """Return the ancestor subject uri at a specific hierarchy level.

    Parameters
    ----------
    level: int
        The level at which to return a ancestor subject uri. Level 1 is the root level.
    subject_uri: str
        The uri of the subject that is supposed to be pruned, i.e., replaced by a ancestor subject
    subject_hierarchy: SubjectHierarchy
        The subject hierarchy used for pruning

    Returns
    -------
    str
        Either the ancestor subject uri of an ancestor at the specified hierarchy level, or the original subject uri
        if the requested subject is already lower (near the root) in the hierarchy than the requested level.
    """
    if level < 1:
        raise ValueError("level must be 1 or larger than 1")
    if subject_uri not in subject_hierarchy:
        raise ValueError(f"subject uri {subject_uri} not in hiearchy")

    ancestors = subject_ancestors_list(subject_hierarchy[subject_uri], subject_hierarchy)
    if len(ancestors) <= level:
        # subject is below (closer to root) or exactly at the requested level
        return subject_uri

    # subject must be above (closer to leaf) than level
    return ancestors[level - 1].uri


def prune_subject_uris_to_level(
    level: int,
    subjects: SubjectUriList,
    subject_hierarchy: SubjectHierarchy,
) -> SubjectUriList:
    """Prune a list of subjects to a specified hierarchy level.

    Duplicate subjects are removed. Applies `prune_subject_uri_to_level` to every subject uri of the provided list.

    Parameters
    ----------
    level: int
        The level at which to return a ancestor subject uri. Level 1 is the root level.
    subjects: SubjectUriList
        The list of subjects to be pruned to the specified hierarchy level
    subject_hierarchy: SubjectHierarchy
        The subject hierarchy that contains the required subject relationships

    Returns
    -------
    SubjectUriList
        a list of subject uris, each pruned according to `prune_subject_uri_to_level`
    """
    return list({prune_subject_uri_to_level(level, s, subject_hierarchy) for s in subjects})


def prune_subject_targets_to_level(
    level: int,
    subject_target_list: SubjectTargets,
    subject_hierarchy: SubjectHierarchy,
) -> SubjectTargets:
    """Prune all subjects of a subject list to the specified hierarchy level.

    Duplicate subjects are removed. Applies `prune_subject_uris_to_level` to every subject uri of the provided list of
    subject lists.

    Parameters
    ----------
    level: int
        The level at which to return a ancestor subject uri. Level 1 is the root level.
    subject_target_list: SubjectTargets
        The list of subject lists to be pruned to the specified hierarchy level
    subject_hierarchy: SubjectHierarchy
        The subject hierarchy that contains the required subject relationships

    Returns
    -------
    SubjectTargets
        A list of subject uri lists, each pruned according to `prune_subject_uris_to_level`
    """
    return list(map(lambda l: prune_subject_uris_to_level(level, l, subject_hierarchy), subject_target_list))


def count_number_of_samples_by_subjects(subject_targets: SubjectTargets) -> Mapping[str, int]:
    """Count the number of occurances of subjects annotations.

    Parameters
    ----------
    subject_targets: SubjectTargets
        A list of subject lists as provided by a dataset

    Returns
    -------
    Mapping[str, int]
        Occurance counts for each subject
    """
    counts = {}

    for subject_list in subject_targets:
        for subject_uri in subject_list:
            counts[subject_uri] = counts.get(subject_uri, 0) + 1

    return counts


def children_map_from_subject_hierarchy(
    subject_hierarchy: SubjectHierarchy,
) -> Mapping[str, Set[str]]:
    """Return a set of children subjects for each parent subject in a subject hierarchy.

    Parameters
    ----------
    subject_hierarchy: SubjectHierarchy
        The subject hierarchy to be scanned for parent-child relationships

    Returns
    -------
    Mapping[str, Set[str]]
        A mapping from parent subjects to their respective set of child subjects
    """
    subject_children: Mapping[str, Set[str]] = {}
    for subject_uri in subject_hierarchy:
        # print(repr(subject_uri), "vs", repr(subject_hierarchy[subject_uri].uri))
        parent_uri = subject_hierarchy[subject_uri].parent_uri
        if parent_uri is not None:
            if parent_uri in subject_children:
                subject_children[parent_uri].add(subject_uri)
            else:
                subject_children[parent_uri] = {subject_uri}
    return subject_children


def prune_subject_uris_to_parent(
    subject_uris: SubjectUriList,
    to_be_pruned_subjects: Set[str],
    subject_hierarchy: SubjectHierarchy,
) -> SubjectUriList:
    """Replace subject uris with uri of parent subject for those that are supposed to be pruned.

    Is used by `prune_subject_targets_to_minimum_samples` to incrementally replace subjects with their parent subjects
    in case there would be insufficient training examples for that subject.

    Parameters
    ----------
    subject_uris: SubjectUriList
        the list of subjects that is scanned for subjects that are supposed to be pruned
    to_be_pruned_subjects: Set[str]
        the list of subjects that is supposed to be replaced with their parent subjects
    subject_hierarchy: SubjectHierarchy
        The subject hierarchy that contains the required subject relationships

    Returns
    -------
    SubjectUriList
        the same subject list as `subject_uris`,  where some subjects are replaced with their parent subject
    """
    new_subject_uris = []
    for subject_uri in subject_uris:
        if subject_uri in to_be_pruned_subjects:
            parent_uri = subject_hierarchy[subject_uri].parent_uri
            if parent_uri is not None:
                new_subject_uris.append(parent_uri)
            else:
                new_subject_uris.append(subject_uri)
        else:
            new_subject_uris.append(subject_uri)
    return list(set(new_subject_uris))


def prune_subject_targets_to_parent(
    subject_targets: SubjectTargets,
    to_be_pruned_subjects: Set[str],
    subject_hierarchy: SubjectHierarchy,
) -> SubjectTargets:
    """Replace subject uris in target list with uri of parent subject for those that are supposed to be pruned.

    Is used by `prune_subject_targets_to_minimum_samples` to incrementally replace subjects with their parent subjects
    in case there would be insufficient training examples for that subject.

    Parameters
    ----------
    subject_targets: SubjectTargets
        the list of subject lists that is scanned for subjects that are supposed to be pruned
    to_be_pruned_subjects: Set[str]
        the list of subjects that is supposed to be replaced with their parent subjects
    subject_hierarchy: SubjectHierarchy
        The subject hierarchy that contains the required subject relationships

    Returns
    -------
    SubjectTargets
        the same subject target list as `subject_targets`,  where some subjects are replaced with their parent subject
    """
    return list(
        map(lambda t: prune_subject_uris_to_parent(t, to_be_pruned_subjects, subject_hierarchy), subject_targets)
    )


def prune_subject_targets_to_minimum_samples(
    minimum_samples: int,
    subject_targets: SubjectTargets,
    subject_hierarchy: SubjectHierarchy
) -> SubjectTargets:
    """Prune subject targets such that subjects with insufficient samples are replaced with their parents.

    Subjects with no parents are left as is. Therefore, it is not guaranteed that all subjects will have a minimum
    number of samples after pruning.

    Parameters
    ----------
    minimum_samples: int
        minimum number of required samples for subjects to be considered of acceptable size
    subject_targets: SubjectTargets
        subject targets that are checked for their number of samples, and modified if necessary (not in-place)
    subject_hierarchy: SubjectHierarchy
        The subject hierarchy that contains the required subject relationships

    Returns
    -------
    SubjectTargets
        the same subject target list as `subject_targets`,  where some subjects are replaced with their parent subject
        in case they do not have a sufficient number of samples
    """
    pruned_subject_targets = subject_targets
    subject_counts = count_number_of_samples_by_subjects(subject_targets)
    subject_children = dict(children_map_from_subject_hierarchy(subject_hierarchy))
    subjects_to_be_checked = set(subject_hierarchy.keys())

    i = 0
    while len(subjects_to_be_checked) > 0:
        # only consider childless subjects, whose counts can not increase further
        childless_subjects = [s for s in subjects_to_be_checked if s not in subject_children]
        logger.debug("prune to parent iteration %d considerung %d childless subjects", i + 1, len(childless_subjects))

        # find subjects that need pruning
        childless_subjects_to_be_pruned = {
            s for s in childless_subjects if s in subject_counts and subject_counts[s] < minimum_samples
        }

        # do pruning
        logger.debug("prune %d subjects because of insufficient samples", len(childless_subjects_to_be_pruned))
        pruned_subject_targets = prune_subject_targets_to_parent(
            pruned_subject_targets,
            childless_subjects_to_be_pruned,
            subject_hierarchy
        )

        # check each childless subject one by one
        for s_uri in childless_subjects:
            # remove every childless subject from child set of parent,
            # such that parent can become childless in the next iterations,
            # and eventually all subjects are checked
            parent_uri = subject_hierarchy[s_uri].parent_uri
            if parent_uri:
                subject_children[parent_uri].remove(s_uri)
                if len(subject_children[parent_uri]) == 0:
                    del subject_children[parent_uri]

        # update counts and check again
        subject_counts = count_number_of_samples_by_subjects(pruned_subject_targets)
        subjects_to_be_checked.difference_update(childless_subjects)
        i += 1

    return pruned_subject_targets
