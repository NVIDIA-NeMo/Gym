# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn job tools: get job postings."""

import json
from typing import Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def linkedin_get_job(
    world: WorldState,
    job_id: str,
    include_details: bool = True,
    include_applicant_count: bool = False,
    include_poster_info: bool = False,
) -> str:
    """
    Get details for a LinkedIn job posting.

    Args:
        job_id: The job posting ID.
        include_details: Whether to include full job details.
        include_applicant_count: Whether to include applicant count.
        include_poster_info: Whether to include poster information.

    Returns:
        JSON string with job details.
    """
    job = world.linkedin.get_job_by_id(job_id)
    if job is None:
        return json.dumps({"success": False, "error": f"Job '{job_id}' not found"})

    result = {
        "success": True,
        "id": job.id,
        "title": job.title,
        "company_id": job.company_id,
        "company_name": job.company_name,
        "location": job.location,
        "is_active": job.is_active,
    }

    if include_details:
        result.update(
            {
                "description": job.description,
                "workplace_type": job.workplace_type,
                "employment_type": job.employment_type,
                "seniority_level": job.seniority_level,
                "industry": job.industry,
                "job_functions": job.job_functions,
                "skills_required": job.skills_required,
                "experience_years": job.experience_years,
                "salary_range": job.salary_range,
                "apply_url": job.apply_url,
                "posted_at": job.posted_at,
                "expires_at": job.expires_at,
            }
        )

    if include_applicant_count:
        result["applicant_count"] = job.applicant_count

    if include_poster_info:
        result["poster_id"] = job.poster_id
        result["poster_name"] = job.poster_name

    return json.dumps(result)


register_metadata(
    linkedin_get_job,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "job",
        "type": "read",
        "action_id": "core:3123022",
    },
)


def linkedin_find_jobs(
    world: WorldState,
    company_id: Optional[str] = None,
    title_contains: Optional[str] = None,
    location: Optional[str] = None,
    active_only: bool = True,
    limit: int = 50,
) -> str:
    """
    Find LinkedIn job postings with filters.

    Args:
        company_id: Filter by company.
        title_contains: Filter by title keyword.
        location: Filter by location.
        active_only: Only return active job postings.
        limit: Maximum number of jobs to return.

    Returns:
        JSON string with matching jobs.
    """
    jobs = world.linkedin.jobs

    if active_only:
        jobs = [j for j in jobs if j.is_active]

    if company_id:
        jobs = [j for j in jobs if j.company_id == company_id]

    if title_contains:
        title_lower = title_contains.lower()
        jobs = [j for j in jobs if title_lower in j.title.lower()]

    if location:
        location_lower = location.lower()
        jobs = [j for j in jobs if j.location and location_lower in j.location.lower()]

    jobs = jobs[:limit]

    return json.dumps(
        {
            "success": True,
            "jobs": [
                {
                    "id": j.id,
                    "title": j.title,
                    "company_name": j.company_name,
                    "location": j.location,
                    "posted_at": j.posted_at,
                }
                for j in jobs
            ],
            "count": len(jobs),
        }
    )


register_metadata(
    linkedin_find_jobs,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "job",
        "type": "read",
        "action_id": "core:3123022",
    },
)
