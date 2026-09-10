# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Jira Cloud REST API request dispatchers for api_fetch.

Translates REST-style Jira API calls into calls to the matching impl tool functions.
Modeled after the Jira Cloud REST API v3 (https://developer.atlassian.com/cloud/jira/platform/rest/v3/).
"""

from automationbench.tools.api.impl.jira import (
    jira_issues_comment,
    jira_issues_create,
    jira_projects_search,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"jira/rest/api/3/project/search$", "projects_search"),
    ("POST", r"jira/rest/api/3/issue$", "issues_create"),
    ("POST", r"jira/rest/api/3/issue/([^/]+)/comment$", "issues_comment"),
]

_HANDLERS = {
    "projects_search": lambda w, ids, p, b: jira_projects_search(w, **p),
    "issues_create": lambda w, ids, p, b: jira_issues_create(w, **b),
    "issues_comment": lambda w, ids, p, b: jira_issues_comment(
        w,
        issueKey=ids[0],
        body=b.get("body", b.get("comment", b.get("text", ""))),
    ),
}


route_jira = make_router(_ROUTES, _HANDLERS)
