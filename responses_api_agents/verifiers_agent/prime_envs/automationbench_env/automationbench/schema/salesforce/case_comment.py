# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce CaseComment data model."""

from pydantic import Field

from automationbench.schema.salesforce.base import SalesforceRecord


class CaseComment(SalesforceRecord):
    """Represents a Salesforce CaseComment record."""

    # Required inputs: parent_id (the case), CommentBody (the comment text)
    parent_id: str = Field(..., description="ID of the case this comment belongs to (mandatory)")
    CommentBody: str = Field(..., description="Text content of the comment (mandatory)")
    IsPublished: bool = Field(
        default=False, description="When True, the comment is visible on the customer portal"
    )

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        return {
            "Id": self.id,
            "ParentId": self.parent_id,
            "CommentBody": self.CommentBody,
            "IsPublished": self.IsPublished,
        }
