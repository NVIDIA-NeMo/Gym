# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Routing layer for the Google Ads API.

Dispatches incoming REST-style requests to the appropriate API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.google_ads import (
    google_ads_campaigns_get,
    google_ads_campaigns_mutate,
    google_ads_conversions_upload,
    google_ads_gaql_search,
    google_ads_offline_user_data_jobs_add_operations,
    google_ads_offline_user_data_jobs_combined,
    google_ads_offline_user_data_jobs_create,
    google_ads_offline_user_data_jobs_run,
    google_ads_user_lists_remove_members,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    # GAQL query endpoint (handles both list and search operations)
    ("POST", r"googleads/v19/customers/([^/]+)/googleAds:search$", "gaql_search"),
    # Offline user data jobs
    ("POST", r"googleads/v19/customers/([^/]+)/offlineUserDataJobs:create$", "offline_jobs_create"),
    (
        "POST",
        r"googleads/v19/customers/([^/]+)/offlineUserDataJobs/([^/]+):addOperations$",
        "offline_jobs_add_ops",
    ),
    (
        "POST",
        r"googleads/v19/customers/([^/]+)/offlineUserDataJobs/([^/]+):run$",
        "offline_jobs_run",
    ),
    # Single-call shorthand: creates a job and appends operations in one request (body supplies userListId + operations)
    ("POST", r"googleads/v19/customers/([^/]+)/offlineUserDataJobs$", "offline_jobs_combined"),
    # Campaign management
    ("POST", r"googleads/v19/customers/([^/]+)/campaigns:mutate$", "campaigns_mutate"),
    ("GET", r"googleads/v19/customers/([^/]+)/campaigns/([^/]+)$", "campaigns_get"),
    # Remove users from a user list
    (
        "POST",
        r"googleads/v19/customers/([^/]+)/userLists/([^/:]+):removeMembers$",
        "user_lists_remove_members",
    ),
    # Click conversion upload
    ("POST", r"googleads/v19/customers/([^/]+):uploadClickConversions$", "conversions_upload"),
]

_HANDLERS = {
    "gaql_search": lambda w, ids, p, b: google_ads_gaql_search(w, ids[0], **b),
    "campaigns_get": lambda w, ids, p, b: google_ads_campaigns_get(w, ids[1], mainAccountId=ids[0]),
    "campaigns_mutate": lambda w, ids, p, b: google_ads_campaigns_mutate(
        w, mainAccountId=ids[0], **b
    ),
    "offline_jobs_create": lambda w, ids, p, b: google_ads_offline_user_data_jobs_create(
        w, ids[0], **b
    ),
    "offline_jobs_add_ops": lambda w, ids, p, b: google_ads_offline_user_data_jobs_add_operations(
        w, ids[0], ids[1], **b
    ),
    "offline_jobs_run": lambda w, ids, p, b: google_ads_offline_user_data_jobs_run(
        w, ids[0], ids[1], **b
    ),
    "offline_jobs_combined": lambda w, ids, p, b: google_ads_offline_user_data_jobs_combined(
        w, ids[0], **b
    ),
    "user_lists_remove_members": lambda w, ids, p, b: google_ads_user_lists_remove_members(
        w, listId=ids[1], mainAccountId=ids[0], **b
    ),
    "conversions_upload": lambda w, ids, p, b: google_ads_conversions_upload(
        w, mainAccountId=ids[0], **b
    ),
}


route_google_ads = make_router(_ROUTES, _HANDLERS)
