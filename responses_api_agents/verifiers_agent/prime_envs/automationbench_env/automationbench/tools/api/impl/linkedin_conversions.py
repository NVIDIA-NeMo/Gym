# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Conversions API tool implementations using the native API.

Each function here aligns with LinkedIn Conversions API (CAPI) field naming
conventions and delegates to Zapier tool functions for WorldState mutations.
The api_fetch routing layer invokes these functions, forwarding parameters
without modification.
"""

from typing import Any, Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.linkedin_conversions.events import (
    linkedin_conversions_send_conversion_event,
)

# ---------------------------------------------------------------------------
# Conversion Events
# ---------------------------------------------------------------------------


def linkedin_conversions_events_create(
    world: WorldState,
    account: str,
    conversion: str,
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    title: Optional[str] = None,
    company: Optional[str] = None,
    country: Optional[str] = None,
    linkedin_first_party_ads_tracking_uuid: Optional[str] = None,
    lead_id: Optional[str] = None,
    amount: Optional[float] = None,
    currency_code: str = "USD",
    conversion_happened_at: Optional[str] = None,
    event_id: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Submit a server-side conversion event to LinkedIn. Matches POST /linkedin/conversions/rest/conversionEvents."""
    from datetime import datetime

    happened_at = None
    if conversion_happened_at:
        try:
            happened_at = datetime.fromisoformat(conversion_happened_at)
        except (ValueError, TypeError):
            happened_at = None

    return linkedin_conversions_send_conversion_event(
        world,
        account=account,
        conversion=conversion,
        email=email,
        first_name=first_name,
        last_name=last_name,
        title=title,
        company=company,
        country=country,
        linkedin_first_party_ads_tracking_uuid=linkedin_first_party_ads_tracking_uuid,
        lead_id=lead_id,
        amount=amount,
        currency_code=currency_code,
        conversion_happened_at=happened_at,
        event_id=event_id,
    )
