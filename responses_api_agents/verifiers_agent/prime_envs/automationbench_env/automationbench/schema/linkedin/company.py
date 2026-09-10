# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Company model."""

from typing import Optional

from pydantic import Field

from automationbench.schema.linkedin.base import (
    LinkedInRecord,
    generate_linkedin_company_id,
)


class Company(LinkedInRecord):
    """A LinkedIn organization or company page."""

    id: str = Field(default_factory=generate_linkedin_company_id)
    name: str
    description: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    company_size: Optional[str] = Field(
        default=None, validation_alias="employee_count_range"
    )  # e.g., "1001-5000 employees"
    headquarters: Optional[str] = None
    founded_year: Optional[int] = None
    logo_url: Optional[str] = None
    cover_image_url: Optional[str] = None

    # Company type
    company_type: Optional[str] = None  # e.g., "Public Company", "Privately Held"

    # Specialties/focus areas
    specialties: list[str] = Field(default_factory=list)

    # Followers and stats
    follower_count: int = 0
    employee_count: Optional[int] = None  # Exact employee count (e.g., 850)
    previous_employee_count: Optional[int] = Field(
        None, description="Prior headcount used to calculate workforce growth"
    )

    # Admin user IDs (people who can post as company)
    admin_ids: list[str] = Field(default_factory=list)

    # Verification status
    is_verified: bool = False
