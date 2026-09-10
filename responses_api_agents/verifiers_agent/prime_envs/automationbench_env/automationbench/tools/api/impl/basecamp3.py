# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Basecamp3 REST API.

These implementations align with Basecamp3 API field naming conventions and
mutate Pydantic models through action recording. The api_fetch routing layer
invokes them, forwarding parameters without modification.
"""

import json
from typing import Any, Dict, Optional

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Todos
# ---------------------------------------------------------------------------


def basecamp3_todos_create(
    world: WorldState,
    account_id: str,
    project_id: str,
    todo_set_id: str,
    todolist_id: str,
    content: Optional[str] = None,
    due_on: Optional[str] = None,
    **kwargs,
) -> str:
    """Add a new todo in Basecamp3. Matches POST /basecamp3/{account_id}/buckets/{project_id}/todosets/{todo_set_id}/todolists/{todolist_id}/todos."""
    params: Dict[str, Any] = {
        "account": account_id,
        "project": project_id,
        "todo_set": todo_set_id,
        "todo_list": todolist_id,
        "content": content,
    }
    if due_on:
        params["due_on"] = due_on
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = world.basecamp3.record_action("todo", params)

    return json.dumps(
        {
            "id": record.id,
            "content": content,
            "status": "active",
            "type": "Todo",
        }
    )
