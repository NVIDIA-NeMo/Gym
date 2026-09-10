# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Hiver tools for conversations (read-only)."""

import json

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

# ============================================================================
# Conversation Tools (Read-only)
# ============================================================================


def hiver_get_conversations(
    world: WorldState,
) -> str:
    """
    Get all conversations from Hiver.

    Returns:
        JSON string with all conversations.
    """
    conversations = [c.to_display_dict() for c in world.hiver.conversations]

    return json.dumps(
        {
            "success": True,
            "conversations": conversations,
            "count": len(conversations),
        }
    )


register_metadata(
    hiver_get_conversations,
    {
        "selected_api": "HiverCLIAPI@1.1.0",
        "action": "conversations",
        "type": "read",
        "action_id": "core:2993001",
    },
)


def hiver_get_conversation(
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
    conversation = world.hiver.get_conversation_by_id(conversation_id)
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
    hiver_get_conversation,
    {
        "selected_api": "HiverCLIAPI@1.1.0",
        "action": "conversation",
        "type": "read",
        "action_id": "core:2993002",
    },
)


# ============================================================================
# User Tools (Read-only)
# ============================================================================


def hiver_get_users(
    world: WorldState,
) -> str:
    """
    Get all users from Hiver.

    Returns:
        JSON string with all users.
    """
    users = [u.to_display_dict() for u in world.hiver.users]

    return json.dumps(
        {
            "success": True,
            "users": users,
            "count": len(users),
        }
    )


register_metadata(
    hiver_get_users,
    {
        "selected_api": "HiverCLIAPI@1.1.0",
        "action": "users",
        "type": "read",
        "action_id": "core:2993003",
    },
)
