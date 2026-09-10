# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Routing layer for the Google Calendar API.

Dispatches incoming REST-style requests to the appropriate API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.google_calendar import (
    google_calendar_calendar_list_list,
    google_calendar_calendars_create,
    google_calendar_calendars_get,
    google_calendar_events_create,
    google_calendar_events_delete,
    google_calendar_events_get,
    google_calendar_events_list,
    google_calendar_events_quick_add,
    google_calendar_events_update,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    # List all calendars for the current user
    ("GET", r"calendar/v3/users/me/calendarList$", "find_calendars"),
    # Calendar CRUD
    ("POST", r"calendar/v3/calendars$", "create_calendar"),
    ("GET", r"calendar/v3/calendars/([^/]+)$", "get_calendar"),
    # Event endpoints — more-specific action paths are matched before the generic ones
    ("POST", r"calendar/v3/calendars/([^/]+)/events/quickAdd$", "quick_add_event"),
    ("GET", r"calendar/v3/calendars/([^/]+)/events/([^/]+)$", "get_event"),
    ("PUT", r"calendar/v3/calendars/([^/]+)/events/([^/]+)$", "update_event"),
    ("PATCH", r"calendar/v3/calendars/([^/]+)/events/([^/]+)$", "update_event"),
    ("DELETE", r"calendar/v3/calendars/([^/]+)/events/([^/]+)$", "delete_event"),
    ("POST", r"calendar/v3/calendars/([^/]+)/events$", "create_event"),
    ("GET", r"calendar/v3/calendars/([^/]+)/events$", "find_events"),
]

_HANDLERS = {
    "create_calendar": lambda w, ids, p, b: google_calendar_calendars_create(w, **b),
    "get_calendar": lambda w, ids, p, b: google_calendar_calendars_get(w, ids[0]),
    "find_calendars": lambda w, ids, p, b: google_calendar_calendar_list_list(w, **p),
    "quick_add_event": lambda w, ids, p, b: google_calendar_events_quick_add(w, ids[0], **p),
    "create_event": lambda w, ids, p, b: google_calendar_events_create(w, ids[0], **b),
    "find_events": lambda w, ids, p, b: google_calendar_events_list(w, ids[0], **p),
    "get_event": lambda w, ids, p, b: google_calendar_events_get(w, ids[0], ids[1]),
    "update_event": lambda w, ids, p, b: google_calendar_events_update(w, ids[0], ids[1], **b),
    "delete_event": lambda w, ids, p, b: google_calendar_events_delete(w, ids[0], ids[1], **p),
}


route_google_calendar = make_router(_ROUTES, _HANDLERS)
