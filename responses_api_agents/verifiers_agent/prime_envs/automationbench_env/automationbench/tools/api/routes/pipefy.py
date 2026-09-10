# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Pipefy REST API route handlers for api_fetch.

Dispatches REST-style Pipefy API requests to the appropriate API-native tool functions.
Arguments are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.pipefy import (
    pipefy_cards_create,
    pipefy_cards_move,
    pipefy_cards_update_field,
    pipefy_database_find_records,
    pipefy_database_tables_list,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"pipefy/v1/tables$", "tables_list"),
    ("GET", r"pipefy/v1/tables/([^/:]+)/records:find$", "records_find"),
    ("POST", r"pipefy/v1/cards$", "cards_create"),
    ("POST", r"pipefy/v1/cards/([^/:]+)/move$", "cards_move"),
    ("POST", r"pipefy/v1/cards/([^/:]+)/fields:update$", "cards_update_field"),
]

_HANDLERS = {
    "tables_list": lambda w, ids, p, b: pipefy_database_tables_list(w),
    "records_find": lambda w, ids, p, b: pipefy_database_find_records(w, tableId=ids[0], **p),
    "cards_create": lambda w, ids, p, b: pipefy_cards_create(w, **b),
    "cards_move": lambda w, ids, p, b: pipefy_cards_move(w, cardId=ids[0], **b),
    "cards_update_field": lambda w, ids, p, b: pipefy_cards_update_field(w, cardId=ids[0], **b),
}


route_pipefy = make_router(_ROUTES, _HANDLERS)
