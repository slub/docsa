"""Schema rest service implementations."""

from typing import Sequence, Mapping
from slub_docsa.common.subject import SubjectHierarchy
from slub_docsa.data.preprocess.subject import subject_ancestors_list
from slub_docsa.serve.common import PublishedSubjectInfo, PublishedSubjectShortInfo, SchemaNotFoundException
from slub_docsa.serve.common import SubjectNotFoundException, SchemasRestService


class SimpleSchemaRestService(SchemasRestService):
    """Schema REST service that provides access to a list of subject hierarchies."""

    def __init__(self, subject_hierarchies: Mapping[str, SubjectHierarchy]):
        """Init schema REST service."""
        self.subject_hierarchies = subject_hierarchies

    def find_schemas(self) -> Sequence[str]:
        """Return the list of avaialble schemas."""
        return list(self.subject_hierarchies.keys())

    def schema_info(self, schema_id: str):
        """Return information about a specific schema."""
        raise NotImplementedError()

    def find_subjects(self, schema_id: str, root_only: bool = True) -> Sequence[str]:
        """Return a list of available subjects for a specific schema."""
        if schema_id not in self.subject_hierarchies:
            raise SchemaNotFoundException(schema_id)
        if root_only:
            return list(self.subject_hierarchies[schema_id].root_subjects())
        return list(self.subject_hierarchies[schema_id])

    def subject_info(self, schema_id: str, subject_uri: str) -> PublishedSubjectInfo:
        """Return information about a subject of a schema."""
        if schema_id not in self.subject_hierarchies:
            raise SchemaNotFoundException(schema_id)
        if subject_uri not in self.subject_hierarchies[schema_id]:
            raise SubjectNotFoundException(schema_id, subject_uri)
        subject_hierarchy = self.subject_hierarchies[schema_id]
        return PublishedSubjectInfo(
            subject_uri=subject_uri,
            labels=subject_hierarchy.subject_labels(subject_uri),
            ancestors=[
                PublishedSubjectShortInfo(
                    subject_uri=ancestor_uri,
                    labels=subject_hierarchy.subject_labels(ancestor_uri),
                ) for ancestor_uri in subject_ancestors_list(subject_uri, subject_hierarchy)[:-1]
            ],
            children=[
                PublishedSubjectShortInfo(
                    subject_uri=child_uri,
                    labels=subject_hierarchy.subject_labels(child_uri),
                ) for child_uri in subject_hierarchy.subject_children(subject_uri)
            ],
        )
