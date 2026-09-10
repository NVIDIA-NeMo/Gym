# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Facebook Lead Ads API.

These implementations align with Facebook Lead Ads API field naming conventions
and delegate WorldState mutations to the Zapier tool functions. The api_fetch
routing layer invokes them, forwarding parameters without modification.
"""

from typing import Any, Optional, cast

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.facebook_lead_ads.ads import (
    facebook_lead_ads_create_lead_ad,
    facebook_lead_ads_find_ad,
    facebook_lead_ads_get_leads,
)

# ---------------------------------------------------------------------------
# Lead Forms / Ads
# ---------------------------------------------------------------------------


def facebook_lead_ads_leadgen_forms_create(
    world: WorldState,
    pageId: str,
    ad_name: str,
    adset_id: str,
    creative_name: str,
    message: str,
    link: str,
    form: str,
    action_type: str = "LEARN_MORE",
    status: str = "ACTIVE",
    description: Optional[str] = None,
    campaign_name: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Set up a Lead Ad on a Facebook Page. Matches POST /facebook/lead_ads/v25/{pageId}/leadgen_forms."""
    return facebook_lead_ads_create_lead_ad(
        world,
        ad_name=ad_name,
        adset_id=adset_id,
        creative_name=creative_name,
        message=message,
        link=link,
        form=form,
        action_type=cast(Any, action_type),
        status=cast(Any, status),
        description=description,
        campaign_name=campaign_name,
    )


# ---------------------------------------------------------------------------
# Leads
# ---------------------------------------------------------------------------


def facebook_lead_ads_leads_list(
    world: WorldState,
    formId: str,
    **kwargs: Any,
) -> str:
    """Get leads from a Lead Ad form. Matches GET /facebook/lead_ads/v25/{formId}/leads."""
    return facebook_lead_ads_get_leads(
        world,
        ad_id=formId,
    )


# ---------------------------------------------------------------------------
# Ads (find)
# ---------------------------------------------------------------------------


def facebook_lead_ads_ads_find(
    world: WorldState,
    adAccountId: str,
    name: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Find a Lead Ad by name within an ad account. Matches GET /facebook/lead_ads/v25/act_{adAccountId}/ads."""
    return facebook_lead_ads_find_ad(
        world,
        name=name,
    )
