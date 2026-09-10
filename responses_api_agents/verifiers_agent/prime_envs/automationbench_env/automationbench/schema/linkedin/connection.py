# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Connection model."""

from typing import Literal, Optional

from pydantic import Field, model_validator

from automationbench.schema.linkedin.base import LinkedInRecord, generate_linkedin_urn


class Connection(LinkedInRecord):
    """Represents a LinkedIn connection at the first or second degree."""

    id: str = Field(default_factory=lambda: generate_linkedin_urn("fs_connection"))
    # The user who owns this connection
    owner_id: Optional[str] = None
    # The connected person's profile ID
    connected_profile_id: Optional[str] = None
    # Email for looking up connections
    email: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_profile_id(cls, data: dict) -> dict:
        """Remap profile_id to connected_profile_id for backwards compatibility."""
        if isinstance(data, dict):
            if "profile_id" in data and "connected_profile_id" not in data:
                data["connected_profile_id"] = data.pop("profile_id")
        return data

    # Connection degree (1st = direct, 2nd = friend of friend)
    degree: Literal[1, 2] = 1
    # When the connection was made
    connected_at: Optional[str] = None  # ISO datetime
    # Shared connections count (for 2nd degree)
    shared_connections_count: int = 0

    # Denormalized profile info for convenience
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    headline: Optional[str] = None
    company: Optional[str] = None
    profile_picture_url: Optional[str] = None

    @property
    def full_name(self) -> Optional[str]:
        """Return the combined first and last name when both are available."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return None
