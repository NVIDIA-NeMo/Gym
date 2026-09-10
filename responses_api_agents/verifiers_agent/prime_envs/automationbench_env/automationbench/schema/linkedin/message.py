# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn message schema model."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import Field

from automationbench.schema.linkedin.base import LinkedInRecord, generate_linkedin_urn


class Message(LinkedInRecord):
    """Represents a LinkedIn direct message exchanged between two members."""

    id: str = Field(default_factory=lambda: generate_linkedin_urn("message"))
    sender_profile_id: str
    recipient_profile_id: str
    text: str
    sent_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    subject: Optional[str] = None

    def to_display_dict(self) -> dict:
        return {
            "id": self.id,
            "sender_profile_id": self.sender_profile_id,
            "recipient_profile_id": self.recipient_profile_id,
            "text": self.text,
            "sent_at": self.sent_at,
            "subject": self.subject,
        }
