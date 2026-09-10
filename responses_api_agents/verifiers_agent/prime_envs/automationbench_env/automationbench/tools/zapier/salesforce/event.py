# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Event tools."""

import json
from datetime import datetime
from typing import Optional

from automationbench.schema.salesforce import Event, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_event_create(
    world: WorldState,
    subject: str,
    start_datetime: str,
    end_datetime: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    who_id: Optional[str] = None,
    what_id: Optional[str] = None,
    is_all_day_event: Optional[bool] = None,
) -> str:
    """
    Create a new Event in Salesforce.

    Args:
        subject: Event subject (required)
        start_datetime: Start date and time (ISO format, required)
        end_datetime: End date and time (ISO format, required)
        description: Event description
        location: Event location
        who_id: Related Contact or Lead ID
        what_id: Related Account, Opportunity, etc.
        is_all_day_event: Whether this is an all-day event

    Returns:
        JSON string with created event or error.
    """
    event = Event(
        id=generate_salesforce_id(),
        subject=subject,
        start_datetime=datetime.fromisoformat(start_datetime),
        end_datetime=datetime.fromisoformat(end_datetime),
        description=description,
        location=location,
        who_id=who_id,
        what_id=what_id,
        is_all_day_event=is_all_day_event if is_all_day_event is not None else False,
    )
    world.salesforce.events.append(event)
    return json.dumps({"success": True, "event": event.to_display_dict()})


register_metadata(
    salesforce_event_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_record",
        "type": "write",
        "action_id": "core:3079403",
    },
)
