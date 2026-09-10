# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom registrant schema model."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.zoom.base import ZoomRecord, generate_zoom_uuid


class Registrant(ZoomRecord):
    """Zoom meeting/webinar registrant model.

    Based on ZoomCLIAPI create_meeting_registrant and create_registrant actions.
    """

    id: str = Field(default_factory=generate_zoom_uuid, description="Registrant ID")  # type: ignore
    meeting_id: str | int = Field(description="Meeting ID this registrant is for")
    email: str = Field(description="Registrant email address")
    first_name: Optional[str] = Field(default=None, description="Registrant first name")
    last_name: Optional[str] = Field(default=None, description="Registrant last name")
    status: Literal["pending", "approved", "denied"] = Field(
        default="pending",
        description="Registration status",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language preference (e.g., 'en-US', 'de-DE')",
    )
    timezone: Optional[str] = Field(
        default=None,
        description="Registrant timezone (e.g., 'America/New_York')",
    )
    # Attendance tracking
    attended: Optional[bool] = Field(default=None, description="Whether registrant attended")
    attendance_duration: Optional[int] = Field(
        default=None, description="Attendance duration in minutes"
    )
    questions_asked: Optional[int] = Field(default=None, description="Number of questions asked")
    polls_responded: Optional[int] = Field(default=None, description="Number of polls responded to")
    materials_downloaded: Optional[bool] = Field(
        default=None, description="Whether registrant downloaded materials"
    )

    # Company info
    company: Optional[str] = Field(default=None, description="Registrant's company name")
    notes: Optional[str] = Field(default=None, description="Admin notes about registrant")

    auto_approved: bool = Field(
        default=False,
        description="Whether this registrant was auto-approved",
    )
    registered_at: datetime = Field(
        default_factory=datetime.now,
        description="Registration timestamp",
    )
    join_url: Optional[str] = Field(
        default=None,
        description="Unique join URL for this registrant",
    )

    def to_display_dict(self) -> dict:
        """Return dict for agent display."""
        d = {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "status": self.status,
            "timezone": self.timezone,
            "attended": self.attended,
            "attendance_duration": self.attendance_duration,
            "questions_asked": self.questions_asked,
            "polls_responded": self.polls_responded,
            "materials_downloaded": self.materials_downloaded,
            "company": self.company,
            "registered_at": self.registered_at.isoformat(),
            "join_url": self.join_url,
        }
        return {k: v for k, v in d.items() if v is not None}
