# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Calendly User model."""

from typing import Optional

from pydantic import Field, model_validator

from automationbench.schema.calendly.base import (
    CalendlyRecord,
    generate_calendly_uri,
    generate_calendly_uuid,
)


class User(CalendlyRecord):
    """A Calendly account holder who hosts scheduling events."""

    @model_validator(mode="before")
    @classmethod
    def normalize_user_fields(cls, data: dict) -> dict:
        """Remap alternate field names to canonical ones for backwards compatibility."""
        if isinstance(data, dict):
            data = dict(data)
            if "role" in data and "organization_role" not in data:
                data["organization_role"] = data.pop("role")
            elif "role" in data:
                data.pop("role")
        return data

    id: str = Field(default_factory=generate_calendly_uuid)
    uri: str = Field(default_factory=lambda: generate_calendly_uri("users"))
    name: str
    email: str
    slug: Optional[str] = None
    timezone: str = "America/New_York"
    avatar_url: Optional[str] = None
    scheduling_url: Optional[str] = None

    # Organization info
    organization_id: Optional[str] = None
    organization_role: Optional[str] = None  # "owner", "admin", "user"

    # Status
    is_active: bool = True
