# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Trello REST API route handlers for api_fetch.

Dispatches REST-style Trello API requests to the appropriate API-native tool functions.
Arguments are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.trello import (
    trello_boards_cards_find,
    trello_boards_lists,
    trello_cards_add_label,
    trello_cards_comment,
    trello_cards_create,
    trello_cards_update,
    trello_members_boards,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"trello/1/members/([^/]+)/boards$", "members_boards"),
    ("GET", r"trello/1/boards/([^/]+)/lists$", "boards_list_find"),
    ("GET", r"trello/1/boards/([^/]+)/cards$", "boards_cards_find"),
    ("POST", r"trello/1/cards$", "cards_create"),
    ("PUT", r"trello/1/cards/([^/]+)$", "cards_update"),
    ("POST", r"trello/1/cards/([^/]+)/idLabels$", "cards_add_label"),
    ("POST", r"trello/1/cards/([^/]+)/actions/comments$", "cards_add_comment"),
]

_HANDLERS = {
    "members_boards": lambda w, ids, p, b: trello_members_boards(w, memberId=ids[0], **p),
    "boards_list_find": lambda w, ids, p, b: trello_boards_lists(w, boardId=ids[0], **p),
    "boards_cards_find": lambda w, ids, p, b: trello_boards_cards_find(w, boardId=ids[0], **p),
    "cards_create": lambda w, ids, p, b: trello_cards_create(w, **b),
    "cards_update": lambda w, ids, p, b: trello_cards_update(w, cardId=ids[0], **{**p, **b}),
    "cards_add_label": lambda w, ids, p, b: trello_cards_add_label(w, cardId=ids[0], **p, **b),
    "cards_add_comment": lambda w, ids, p, b: trello_cards_comment(w, cardId=ids[0], **{**p, **b}),
}


route_trello = make_router(_ROUTES, _HANDLERS)
