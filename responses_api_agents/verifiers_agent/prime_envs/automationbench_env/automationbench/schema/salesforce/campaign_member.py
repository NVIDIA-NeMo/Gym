# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce CampaignMember data model."""

from typing import Optional

from pydantic import Field

from automationbench.schema.salesforce.base import SalesforceRecord


class CampaignMember(SalesforceRecord):
    """Represents a Salesforce CampaignMember record that associates Contacts or Leads with a Campaign."""

    CampaignId: str = Field(..., description="ID of the related campaign (mandatory)")
    ContactId: Optional[str] = Field(
        None, description="ID of the associated contact, if applicable"
    )
    LeadId: Optional[str] = Field(None, description="ID of the associated lead, if applicable")
    Status: str = Field(default="Sent", description="Participation status of this member")

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        result = {
            "Id": self.id,
            "CampaignId": self.CampaignId,
            "Status": self.Status,
        }
        if self.ContactId:
            result["ContactId"] = self.ContactId
        if self.LeadId:
            result["LeadId"] = self.LeadId
        return result
