# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Note data model."""

from typing import Optional

from pydantic import Field

from automationbench.schema.salesforce.base import SalesforceRecord


class Note(SalesforceRecord):
    """Represents a Salesforce Note record."""

    # Required: parent_id (the record this note is attached to); custom_type selects the object type
    parent_id: str = Field(..., description="ID of the record this note is attached to (mandatory)")
    custom_type: Optional[str] = Field(
        None,
        description="Specifies which object type to target (e.g., account, contact, lead, opportunity)",
    )
    title: Optional[str] = Field(None, description="Heading or subject of the note")
    body: Optional[str] = Field(None, description="Main text content of the note")
    is_private: bool = Field(
        default=False, description="When True, only the owner can view this note"
    )

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        return {
            "Id": self.id,
            "ParentId": self.parent_id,
            "Title": self.title,
            "Body": self.body,
        }
