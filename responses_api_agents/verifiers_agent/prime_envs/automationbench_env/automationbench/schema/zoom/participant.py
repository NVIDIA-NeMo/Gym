# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom participant schema model."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Participant(BaseModel):
    """Zoom meeting/webinar participant with attendance duration."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    meeting_id: str | int = Field(description="Meeting or webinar ID")
    user_id: Optional[str] = Field(default=None, description="Participant user ID")
    user_email: str = Field(description="Participant email")
    user_name: Optional[str] = Field(default=None, description="Participant name")
    duration: int = Field(default=0, description="Attendance duration in minutes")
    join_time: Optional[str] = Field(default=None, description="Join time (ISO 8601)")
    leave_time: Optional[str] = Field(default=None, description="Leave time (ISO 8601)")

    def to_display_dict(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "user_email": self.user_email,
            "user_name": self.user_name,
            "duration": self.duration,
        }
