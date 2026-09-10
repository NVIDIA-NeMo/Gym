# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Routing layer for the Freshdesk API.

Dispatches incoming REST-style requests to the appropriate API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.freshdesk import (
    freshdesk_companies_create,
    freshdesk_companies_get,
    freshdesk_companies_list,
    freshdesk_contacts_create,
    freshdesk_contacts_get,
    freshdesk_contacts_list,
    freshdesk_contacts_search,
    freshdesk_ticket_notes_create,
    freshdesk_ticket_notes_list,
    freshdesk_tickets_create,
    freshdesk_tickets_get,
    freshdesk_tickets_list,
    freshdesk_tickets_search,
    freshdesk_tickets_update,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and handler map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # Support tickets
    ("GET", r"freshdesk/api/v2/search/tickets$", "tickets_search"),
    ("POST", r"freshdesk/api/v2/tickets$", "tickets_create"),
    ("GET", r"freshdesk/api/v2/tickets/([^/]+)/conversations$", "ticket_notes_list"),
    ("POST", r"freshdesk/api/v2/tickets/([^/]+)/notes$", "ticket_notes_create"),
    ("POST", r"freshdesk/api/v2/tickets/([^/]+)/conversations$", "ticket_notes_create"),
    ("GET", r"freshdesk/api/v2/tickets/([^/]+)$", "tickets_get"),
    ("PUT", r"freshdesk/api/v2/tickets/([^/]+)$", "tickets_update"),
    ("PATCH", r"freshdesk/api/v2/tickets/([^/]+)$", "tickets_update"),
    ("GET", r"freshdesk/api/v2/tickets$", "tickets_list"),
    # Customer contacts
    ("GET", r"freshdesk/api/v2/search/contacts$", "contacts_search"),
    ("POST", r"freshdesk/api/v2/contacts$", "contacts_create"),
    ("GET", r"freshdesk/api/v2/contacts/([^/]+)$", "contacts_get"),
    ("GET", r"freshdesk/api/v2/contacts$", "contacts_list"),
    # Company accounts
    ("GET", r"freshdesk/api/v2/companies/([^/]+)$", "companies_get"),
    ("GET", r"freshdesk/api/v2/companies$", "companies_list"),
    ("POST", r"freshdesk/api/v2/companies$", "companies_create"),
]

_HANDLERS = {
    "tickets_list": lambda w, ids, p, b: freshdesk_tickets_list(w),
    "tickets_get": lambda w, ids, p, b: freshdesk_tickets_get(w, ids[0]),
    "tickets_create": lambda w, ids, p, b: freshdesk_tickets_create(w, **b),
    "tickets_update": lambda w, ids, p, b: freshdesk_tickets_update(w, ids[0], **b),
    "tickets_search": lambda w, ids, p, b: freshdesk_tickets_search(w, **p),
    "ticket_notes_list": lambda w, ids, p, b: freshdesk_ticket_notes_list(w, ids[0]),
    "ticket_notes_create": lambda w, ids, p, b: freshdesk_ticket_notes_create(w, ids[0], **b),
    "contacts_list": lambda w, ids, p, b: freshdesk_contacts_list(w),
    "contacts_get": lambda w, ids, p, b: freshdesk_contacts_get(w, ids[0]),
    "contacts_create": lambda w, ids, p, b: freshdesk_contacts_create(w, **b),
    "contacts_search": lambda w, ids, p, b: freshdesk_contacts_search(w, **p),
    "companies_list": lambda w, ids, p, b: freshdesk_companies_list(w),
    "companies_get": lambda w, ids, p, b: freshdesk_companies_get(w, ids[0]),
    "companies_create": lambda w, ids, p, b: freshdesk_companies_create(w, **b),
}


route_freshdesk = make_router(_ROUTES, _HANDLERS)
