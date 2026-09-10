# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Job model."""

from typing import Literal, Optional

from pydantic import Field, model_validator

from automationbench.schema.linkedin.base import LinkedInRecord


def generate_linkedin_job_id() -> str:
    """Construct a plausible LinkedIn job listing identifier."""
    import random
    import string

    return "".join(random.choices(string.digits, k=10))


class Job(LinkedInRecord):
    """Represents a LinkedIn job listing."""

    @model_validator(mode="before")
    @classmethod
    def normalize_job_fields(cls, data: dict) -> dict:
        """Remap alternate field names to canonical ones for backwards compatibility."""
        if isinstance(data, dict):
            data = dict(data)
            if "job_id" in data and "id" not in data:
                data["id"] = data.pop("job_id")
            elif "job_id" in data:
                data.pop("job_id")
        return data

    id: str = Field(default_factory=generate_linkedin_job_id)
    title: str
    company_id: Optional[str] = None
    company_name: Optional[str] = None

    # Job details
    description: Optional[str] = None
    location: Optional[str] = None
    workplace_type: Optional[Literal["on-site", "remote", "hybrid"]] = None
    employment_type: Optional[
        Literal["full-time", "part-time", "contract", "internship", "temporary"]
    ] = None
    seniority_level: Optional[str] = None  # e.g., "Mid-Senior level"
    industry: Optional[str] = None
    job_functions: list[str] = Field(default_factory=list)

    # Requirements
    skills_required: list[str] = Field(default_factory=list)
    experience_years: Optional[int] = None

    # Compensation
    salary_range: Optional[str] = None  # e.g., "$100,000 - $150,000"

    # Application info
    applicant_count: int = 0
    apply_url: Optional[str] = None

    # Poster info
    poster_id: Optional[str] = None  # Profile ID of person who posted
    poster_name: Optional[str] = None

    # Timestamps
    posted_at: Optional[str] = None  # ISO datetime
    expires_at: Optional[str] = None  # ISO datetime

    # State
    is_active: bool = True
