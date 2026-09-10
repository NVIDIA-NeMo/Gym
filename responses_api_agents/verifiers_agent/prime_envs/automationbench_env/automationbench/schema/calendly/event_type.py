# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Calendly EventType model."""

from typing import Literal, Optional

from pydantic import Field, model_validator

from automationbench.schema.calendly.base import (
    CalendlyRecord,
    generate_calendly_uri,
    generate_calendly_uuid,
)


class EventType(CalendlyRecord):
    """A Calendly event type that defines a scheduling template."""

    @model_validator(mode="before")
    @classmethod
    def normalize_event_type_fields(cls, data: dict) -> dict:
        """Remap alternate field names to canonical ones for backwards compatibility."""
        if isinstance(data, dict):
            # active -> is_active
            if "active" in data and "is_active" not in data:
                data["is_active"] = data.pop("active")
            # duration_minutes -> duration
            if "duration_minutes" in data and "duration" not in data:
                data["duration"] = data.pop("duration_minutes")
        return data

    id: str = Field(default_factory=generate_calendly_uuid)
    uri: str = Field(default_factory=lambda: generate_calendly_uri("event_types"))
    name: str
    description: Optional[str] = None
    slug: Optional[str] = None

    # Duration
    duration: int = 30  # minutes

    # Location settings
    location_kind: Optional[
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
    ] = None
    location: Optional[str] = None

    # Scheduling settings
    scheduling_url: Optional[str] = None
    color: Optional[str] = None

    # Owner
    owner_id: str = ""  # User ID who owns this event type
    owner_email: Optional[str] = Field(None, description="Email address of the event type owner")

    # Status
    is_active: bool = True

    # Type
    kind: Literal["solo", "group", "collective"] = "solo"
