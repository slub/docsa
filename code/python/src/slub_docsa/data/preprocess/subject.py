"""Methods for pre-processing subjects, e.g., pruning."""

from typing import List

from slub_docsa.common.subject import SubjectHierarchyType, SubjectNodeType, SubjectTargets, SubjectUriList


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
