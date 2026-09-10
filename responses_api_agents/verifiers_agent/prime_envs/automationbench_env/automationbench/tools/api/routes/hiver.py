# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Hiver REST API request dispatchers for api_fetch.

Translates REST-style Hiver API calls into calls to the matching impl functions.
Modeled after the Hiver API v1 (https://developer.hiverhq.com/).
"""

from automationbench.tools.api.impl.hiver import (
    hiver_conversations_get,
    hiver_conversations_list,
    hiver_conversations_update,
    hiver_inboxes_get,
    hiver_inboxes_list,
    hiver_users_list,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"hiver/v1/inboxes/([^/]+)/conversations/([^/]+)$", "conversations_get"),
    ("PATCH", r"hiver/v1/inboxes/([^/]+)/conversations/([^/]+)$", "conversations_update"),
    ("GET", r"hiver/v1/inboxes/([^/]+)/conversations$", "conversations_list"),
    ("GET", r"hiver/v1/inboxes/([^/]+)/users$", "users_list"),
    ("GET", r"hiver/v1/inboxes/([^/]+)$", "inboxes_get"),
    ("GET", r"hiver/v1/inboxes$", "inboxes_list"),
]

_HANDLERS = {
    "conversations_list": lambda w, ids, p, b: hiver_conversations_list(w, inbox_id=ids[0], **p),
    "conversations_get": lambda w, ids, p, b: hiver_conversations_get(
        w, inbox_id=ids[0], conversation_id=ids[1]
    ),
    "conversations_update": lambda w, ids, p, b: hiver_conversations_update(
        w, inbox_id=ids[0], conversation_id=ids[1], **b
    ),
    "users_list": lambda w, ids, p, b: hiver_users_list(w, inbox_id=ids[0]),
    "inboxes_get": lambda w, ids, p, b: hiver_inboxes_get(w, inbox_id=ids[0]),
    "inboxes_list": lambda w, ids, p, b: hiver_inboxes_list(w),
}


route_hiver = make_router(_ROUTES, _HANDLERS)
