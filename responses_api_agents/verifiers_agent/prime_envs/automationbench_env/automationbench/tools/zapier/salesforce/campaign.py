# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Campaign tools."""

import json
from datetime import datetime
from typing import Optional

from automationbench.schema.salesforce import Campaign, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_campaign_create(
    world: WorldState,
    campaign_name: str,
    campaign_type: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    description: Optional[str] = None,
    budgeted_cost: Optional[float] = None,
    expected_revenue: Optional[float] = None,
    is_active: Optional[bool] = None,
) -> str:
    """
    Create a new Campaign in Salesforce.

    Args:
        campaign_name: Campaign name (required)
        campaign_type: Campaign type (e.g., Email, Webinar, Conference)
        status: Campaign status (default: Planned)
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        description: Campaign description
        budgeted_cost: Budgeted cost for campaign
        expected_revenue: Expected revenue from campaign
        is_active: Whether campaign is active

    Returns:
        JSON string with created campaign or error.
    """
    campaign = Campaign(
        id=generate_salesforce_id(),
        campaign_name=campaign_name,
        campaign_type=campaign_type,
        status=status or "Planned",
        start_date=datetime.fromisoformat(start_date) if start_date else None,
        end_date=datetime.fromisoformat(end_date) if end_date else None,
        description=description,
        budgeted_cost=budgeted_cost,
        expected_revenue=expected_revenue,
        is_active=is_active if is_active is not None else True,
    )
    world.salesforce.campaigns.append(campaign)
    return json.dumps({"success": True, "campaign": campaign.to_display_dict()})


register_metadata(
    salesforce_campaign_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_record",
        "type": "write",
        "action_id": "core:3079403",
    },
)
