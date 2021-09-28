"""Methods for pre-processing subjects, e.g., pruning."""

import logging

from typing import List, Mapping, Set

from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType, SubjectTargets, SubjectUriList

logger = logging.getLogger(__name__)


def subject_ancestors_list(
    subject: SubjectNodeType,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
) -> List[SubjectNodeType]:
    """Return the list of ancestors for a subject node in a subject hierarchy."""
    ancestors: List[SubjectNodeType] = []
    next_subject = subject

    while next_subject is not None:
        ancestors.insert(0, next_subject)
        if next_subject.parent_uri is not None:
            next_subject = subject_hierarchy.get(next_subject.parent_uri, None)
        else:
            next_subject = None

    return ancestors


def subject_label_breadcrumb(
    subject: SubjectNodeType,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
) -> str:
    """Return a label breadcrumb as a string describing the subject hierarchy path from root to this subject."""
    subject_path = subject_ancestors_list(subject, subject_hierarchy)
    return " | ".join(map(lambda s: s.label, subject_path))


def prune_subject_uri_to_level(
    level: int,
    subject_uri: str,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType]
) -> str:
    """Return the ancestor subject uri at a specific hierarchy level.

    Parameters
    ----------
    level: int
        The level at which to return a ancestor subject uri. Level 1 is the root level.
    subject_uri: str
        The uri of the subject that is supposed to be pruned, i.e., replaced by a ancestor subject
    subject_hierarchy: SubjectHierarchyType
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
        raise ValueError("subject uri %s not in hiearchy" % subject_uri)

    ancestors = subject_ancestors_list(subject_hierarchy[subject_uri], subject_hierarchy)
    if len(ancestors) <= level:
        # subject is below (closer to root) or exactly at the requested level
        return subject_uri

    # subject must be above (closer to leaf) than level
    return ancestors[level-1].uri


def prune_subject_uris_to_level(
    level: int,
    subjects: SubjectUriList,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType]
) -> SubjectUriList:
    """Prune a list of subjects to a specified hierarchy level.

    Duplicate subjects are removed.
    """
    return list({prune_subject_uri_to_level(level, s, subject_hierarchy) for s in subjects})


def prune_subject_targets_to_level(
    level: int,
    subject_target_list: SubjectTargets,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType]
) -> SubjectTargets:
    """Prune all subjects of a subject list to the specified hiearchy level.

    Duplicate subjects are removed.
    """
    return list(map(lambda l: prune_subject_uris_to_level(level, l, subject_hierarchy), subject_target_list))


def count_number_of_samples_by_subjects(subject_targets: SubjectTargets) -> Mapping[str, int]:
    """Count the number of occurances of subjects annotations."""
    counts = {}

    for subject_list in subject_targets:
        for subject_uri in subject_list:
            counts[subject_uri] = counts.get(subject_uri, 0) + 1

    return counts


def children_map_from_subject_hierarchy(
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType],
) -> Mapping[str, Set[str]]:
    """Return a set of children subjects for each parent subject in a subject hierarchy."""
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
    subject_hierarchy:  SubjectHierarchyType[SubjectNodeType],
) -> SubjectUriList:
    """Replace subject uris with uri of parent subject for those that are supposed to be pruned."""
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
    subject_hierarchy:  SubjectHierarchyType[SubjectNodeType],
) -> SubjectTargets:
    """Replace subject uris in target list with uri of parent subject for those that are supposed to be pruned."""
    return list(
        map(lambda t: prune_subject_uris_to_parent(t, to_be_pruned_subjects, subject_hierarchy), subject_targets)
    )


def prune_subject_targets_to_minimum_samples(
    minimum_samples: int,
    subject_targets: SubjectTargets,
    subject_hierarchy: SubjectHierarchyType[SubjectNodeType]
) -> SubjectTargets:
    """Prune subject targets such that subjects with insufficient samples are replaced with their parents.

    Subjects with no parents are left as is. Therefore, it is not guaranteed that all subjects will have a minimum
    number of samples after pruning.
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
