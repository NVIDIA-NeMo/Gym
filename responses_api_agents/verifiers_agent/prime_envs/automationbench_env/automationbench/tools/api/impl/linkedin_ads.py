# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Ads tool implementations using the native API.

Each function here aligns with LinkedIn Marketing API (REST) field naming
conventions and delegates to Zapier tool functions for WorldState mutations.
The api_fetch routing layer invokes these functions, forwarding parameters
without modification.
"""

from typing import Any, Optional, cast

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.linkedin_ads.audiences import (
    linkedin_ads_add_company_to_audience,
    linkedin_ads_add_contact_to_audience,
    linkedin_ads_create_audience,
    linkedin_ads_remove_email_from_audience,
)
from automationbench.tools.zapier.linkedin_ads.conversions import (
    linkedin_ads_create_conversion_event,
    linkedin_ads_create_report,
)

# ---------------------------------------------------------------------------
# Audiences (DMP Segments)
# ---------------------------------------------------------------------------


def linkedin_ads_dmp_segments_create(
    world: WorldState,
    accountId: str,
    name: str,
    type: str = "USER",
    **kwargs: Any,
) -> str:
    """Create a LinkedIn Matched Audience segment. Matches POST /linkedin/ads/rest/adAccounts/{accountId}/dmpSegments."""
    return linkedin_ads_create_audience(
        world,
        account=accountId,
        name=name,
        type=cast(Any, type),
    )


def linkedin_ads_dmp_segment_users_add(
    world: WorldState,
    segmentId: str,
    account: str,
    audience: Optional[str] = None,
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Add a contact to a LinkedIn Matched Audience. Matches POST /linkedin/ads/rest/dmpSegments/{segmentId}/users?action=ADD."""
    return linkedin_ads_add_contact_to_audience(
        world,
        account=account,
        audience=audience or segmentId,
        email=email,
        first_name=first_name,
        last_name=last_name,
    )


def linkedin_ads_dmp_segment_companies_add(
    world: WorldState,
    segmentId: str,
    account: str,
    audience: Optional[str] = None,
    company_name: Optional[str] = None,
    company_id: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Add a company to a LinkedIn Matched Audience. Matches POST /linkedin/ads/rest/dmpSegments/{segmentId}/companies?action=ADD."""
    return linkedin_ads_add_company_to_audience(
        world,
        account=account,
        audience=audience or segmentId,
        company_name=company_name,
        company_id=company_id,
    )


def linkedin_ads_dmp_segment_users_remove(
    world: WorldState,
    segmentId: str,
    account: str,
    audience: Optional[str] = None,
    email: str = "",
    **kwargs: Any,
) -> str:
    """Remove an email address from a LinkedIn Matched Audience. Matches DELETE /linkedin/ads/rest/dmpSegments/{segmentId}/users."""
    return linkedin_ads_remove_email_from_audience(
        world,
        account=account,
        audience=audience or segmentId,
        email=email,
    )


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def linkedin_ads_conversions_create(
    world: WorldState,
    account: str,
    conversion: str,
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    value: Optional[float] = None,
    currency: str = "USD",
    **kwargs: Any,
) -> str:
    """Submit a conversion event to LinkedIn Ads. Matches POST /linkedin/ads/rest/conversions."""
    return linkedin_ads_create_conversion_event(
        world,
        account=account,
        conversion=conversion,
        email=email,
        first_name=first_name,
        last_name=last_name,
        value=value,
        currency=currency,
    )


# ---------------------------------------------------------------------------
# Analytics / Reports
# ---------------------------------------------------------------------------


def linkedin_ads_analytics_create(
    world: WorldState,
    account: str,
    name: Optional[str] = None,
    report_type: str = "CAMPAIGN_PERFORMANCE",
    **kwargs: Any,
) -> str:
    """Generate a LinkedIn Ads analytics report. Matches POST /linkedin/ads/rest/adAnalytics."""
    return linkedin_ads_create_report(
        world,
        account=account,
        name=name,
        report_type=report_type,
    )
