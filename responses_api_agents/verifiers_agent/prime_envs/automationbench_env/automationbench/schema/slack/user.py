# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Model representing a Slack workspace user."""

from typing import Optional

from pydantic import Field, model_validator

from automationbench.schema.slack.base import SlackRecord, generate_slack_user_id


class User(SlackRecord):
    """A member of a Slack workspace."""

    @model_validator(mode="before")
    @classmethod
    def normalize_user_fields(cls, data: dict) -> dict:
        """Remap legacy field aliases so older state definitions remain compatible."""
        if isinstance(data, dict):
            data = dict(data)
            if "real_name" in data and "name" not in data:
                data["name"] = data.pop("real_name")
            elif "real_name" in data:
                data.pop("real_name")
        return data

    id: str = Field(default_factory=generate_slack_user_id)
    username: Optional[str] = None  # Short handle, e.g., "jsmith"
    name: str  # Full display name, e.g., "John Smith"
    email: Optional[str] = None
    # Profile information
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    phone: Optional[str] = None
    # Status
    status_text: Optional[str] = None
    status_emoji: Optional[str] = None
    # Flags
    is_admin: bool = False
    is_owner: bool = False
    is_bot: bool = False
    is_deleted: bool = False
    # Timezone
    tz: Optional[str] = None
    tz_label: Optional[str] = None

    def to_display_dict(self) -> dict:
        """Build the dictionary representation shown to agents."""
        profile: dict[str, str] = {
            "display_name": self.name,
        }
        if self.email:
            profile["email"] = self.email
        if self.first_name:
            profile["first_name"] = self.first_name
        if self.last_name:
            profile["last_name"] = self.last_name
        if self.title:
            profile["title"] = self.title
        if self.status_text:
            profile["status_text"] = self.status_text
        if self.status_emoji:
            profile["status_emoji"] = self.status_emoji

        result: dict[str, object] = {
            "id": self.id,
            "name": self.username or self.name,
            "real_name": self.name,
            "profile": profile,
        }
        if self.is_admin:
            result["is_admin"] = True
        if self.is_bot:
            result["is_bot"] = True
        return result
