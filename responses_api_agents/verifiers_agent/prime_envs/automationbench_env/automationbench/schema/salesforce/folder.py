# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Folder data model."""

from typing import Optional

from pydantic import Field

from automationbench.schema.salesforce.base import SalesforceRecord


class Folder(SalesforceRecord):
    """Represents a Salesforce Folder record used to organize documents and reports."""

    # Used in search_or_write operations via search_field / search_value
    # Standard Folder fields
    Name: str = Field(..., description="Display name of the folder (mandatory)")
    Type: Optional[str] = Field(
        None, description="Category of content stored (e.g., Document, Email, Report)"
    )
    AccessType: Optional[str] = Field(
        "Public", description="Visibility setting (e.g., Public, Hidden, Shared)"
    )
    DeveloperName: Optional[str] = Field(None, description="Internal API name for the folder")

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        return {
            "Id": self.id,
            "Name": self.Name,
            "Type": self.Type,
            "AccessType": self.AccessType,
        }
