# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated constants for the CUA-Gym-Hub deployment.

APP_DIRS mirrors `deploy-all.sh` in CUA-Gym-Hub commit b8207bfc: mock apps are the
sorted `websites/*_mock` directories, served on consecutive ports from the base
port (default 8000). PLACEHOLDER_MAP mirrors the dataset's url_variables.json.
Regenerate against a new hub commit rather than editing by hand.

Deployments that do not follow the base_port + APP_DIRS-index convention (e.g. a
partial hub serving a handful of apps on arbitrary host:port pairs) are described
by a YAML file named via $CUA_GYM_ENDPOINTS_YAML instead of by editing this module
— see endpoint_overrides().
"""

from __future__ import annotations

import os
from typing import Any


HUB_COMMIT = "b8207bfc1838f69b971986e9c489c8aa5d0460c1"  # pragma: allowlist secret

ENDPOINTS_YAML_ENV = "CUA_GYM_ENDPOINTS_YAML"

APP_DIRS: list[str] = [
    "12306_mock",
    "Canvas-LMS_mock",
    "Expensify_mock",
    "PACS-viewer_mock",
    "SAP_mock",
    "ServiceNow_mock",
    "TradingView_mock",
    "Zendesk_mock",
    "adp_mock",
    "airtable_mock",
    "aliyun_mock",
    "amazon_mock",
    "amazon_seller_mock",
    "amplitude_mock",
    "asana_mock",
    "aws_console_mock",
    "azure_mock",
    "bamboohr_mock",
    "booking_com_mock",
    "canva_mock",
    "canvas_mock",
    "circleci_mock",
    "clio_mock",
    "cloudflare_mock",
    "coinbase_mock",
    "confluence_mock",
    "contractbook_mock",
    "datadog_mock",
    "dingtalk_mock",
    "discord_mock",
    "docusign_mock",
    "ebay_mock",
    "epic-health_mock",
    "expedia_mock",
    "facebook_mock",
    "feishu_mock",
    "github_mock",
    "gitlab_mock",
    "gmail_mock",
    "google_ads_mock",
    "google_analytics_mock",
    "google_calendar_mock",
    "google_docs_mock",
    "google_drive_mock",
    "google_flights_mock",
    "google_sheets_mock",
    "greenhouse_mock",
    "gusto_mock",
    "hotjar_mock",
    "hubspot_marketing_mock",
    "hubspot_mock",
    "instacart_mock",
    "instagram_mock",
    "jira_mock",
    "klaviyo_mock",
    "lattice_mock",
    "linear_mock",
    "linkedin_mock",
    "looker_studio_mock",
    "lucidchart_mock",
    "mailchimp_mock",
    "meta_ads_mock",
    "microsoft_teams_mock",
    "miro_mock",
    "mixpanel_mock",
    "monday_mock",
    "notion_mock",
    "openreview_mock",
    "outlook_web_mock",
    "paypal_mock",
    "pinterest_mock",
    "postman_mock",
    "quickbooks_mock",
    "reddit_mock",
    "robinhood_mock",
    "salesforce_mock",
    "sentry_mock",
    "shopify_admin_mock",
    "slack_mock",
    "stripe_dashboard_mock",
    "tableau_mock",
    "taobao_seller_mock",
    "trello_mock",
    "tripadvisor_mock",
    "twitter_mock",
    "uber_eats_mock",
    "vercel_mock",
    "wandb_mock",
    "wechat_mock",
    "weibo_mock",
    "westlaw_mock",
    "woocommerce_mock",
    "workday_mock",
    "xiaohongshu_mock",
    "youtube_mock",
    "zhihu_mock",
    "zillow_mock",
    "zoom_web_mock",
]

# placeholder -> {app_dir, kind}; kind "url" gets http://host:port, "host" gets host:port
PLACEHOLDER_MAP: dict[str, dict[str, str]] = {
    "__CUA_GYM_ASANA_URL__": {"app_dir": "asana_mock", "kind": "url"},
    "__CUA_GYM_DISCORD_URL__": {"app_dir": "discord_mock", "kind": "url"},
    "__CUA_GYM_DOCUSIGN_URL__": {"app_dir": "docusign_mock", "kind": "url"},
    "__CUA_GYM_FACEBOOK_HOST__": {"app_dir": "facebook_mock", "kind": "host"},
    "__CUA_GYM_FACEBOOK_URL__": {"app_dir": "facebook_mock", "kind": "url"},
    "__CUA_GYM_GITHUB_HOST__": {"app_dir": "github_mock", "kind": "host"},
    "__CUA_GYM_GITHUB_URL__": {"app_dir": "github_mock", "kind": "url"},
    "__CUA_GYM_GITLAB_HOST__": {"app_dir": "gitlab_mock", "kind": "host"},
    "__CUA_GYM_GITLAB_URL__": {"app_dir": "gitlab_mock", "kind": "url"},
    "__CUA_GYM_GMAIL_HOST__": {"app_dir": "gmail_mock", "kind": "host"},
    "__CUA_GYM_GMAIL_URL__": {"app_dir": "gmail_mock", "kind": "url"},
    "__CUA_GYM_GOOGLE_CALENDAR_URL__": {"app_dir": "google_calendar_mock", "kind": "url"},
    "__CUA_GYM_GOOGLE_DOCS_HOST__": {"app_dir": "google_docs_mock", "kind": "host"},
    "__CUA_GYM_GOOGLE_DOCS_URL__": {"app_dir": "google_docs_mock", "kind": "url"},
    "__CUA_GYM_GOOGLE_DRIVE_HOST__": {"app_dir": "google_drive_mock", "kind": "host"},
    "__CUA_GYM_GOOGLE_DRIVE_URL__": {"app_dir": "google_drive_mock", "kind": "url"},
    "__CUA_GYM_GOOGLE_SHEETS_HOST__": {"app_dir": "google_sheets_mock", "kind": "host"},
    "__CUA_GYM_GOOGLE_SHEETS_URL__": {"app_dir": "google_sheets_mock", "kind": "url"},
    "__CUA_GYM_HUBSPOT_HOST__": {"app_dir": "hubspot_mock", "kind": "host"},
    "__CUA_GYM_HUBSPOT_URL__": {"app_dir": "hubspot_mock", "kind": "url"},
    "__CUA_GYM_INSTACART_HOST__": {"app_dir": "instacart_mock", "kind": "host"},
    "__CUA_GYM_INSTACART_URL__": {"app_dir": "instacart_mock", "kind": "url"},
    "__CUA_GYM_INSTAGRAM_HOST__": {"app_dir": "instagram_mock", "kind": "host"},
    "__CUA_GYM_INSTAGRAM_URL__": {"app_dir": "instagram_mock", "kind": "url"},
    "__CUA_GYM_JIRA_URL__": {"app_dir": "jira_mock", "kind": "url"},
    "__CUA_GYM_LINKEDIN_URL__": {"app_dir": "linkedin_mock", "kind": "url"},
    "__CUA_GYM_MICROSOFT_TEAMS_URL__": {"app_dir": "microsoft_teams_mock", "kind": "url"},
    "__CUA_GYM_MONDAY_URL__": {"app_dir": "monday_mock", "kind": "url"},
    "__CUA_GYM_NOTION_MOCK_URL__": {"app_dir": "notion_mock", "kind": "url"},
    "__CUA_GYM_NOTION_URL__": {"app_dir": "notion_mock", "kind": "url"},
    "__CUA_GYM_OUTLOOK_WEB_HOST__": {"app_dir": "outlook_web_mock", "kind": "host"},
    "__CUA_GYM_OUTLOOK_WEB_URL__": {"app_dir": "outlook_web_mock", "kind": "url"},
    "__CUA_GYM_PINTEREST_URL__": {"app_dir": "pinterest_mock", "kind": "url"},
    "__CUA_GYM_POSTMAN_URL__": {"app_dir": "postman_mock", "kind": "url"},
    "__CUA_GYM_REDDIT_URL__": {"app_dir": "reddit_mock", "kind": "url"},
    "__CUA_GYM_SALESFORCE_URL__": {"app_dir": "salesforce_mock", "kind": "url"},
    "__CUA_GYM_SHOPIFY_ADMIN_URL__": {"app_dir": "shopify_admin_mock", "kind": "url"},
    "__CUA_GYM_SLACK_URL__": {"app_dir": "slack_mock", "kind": "url"},
    "__CUA_GYM_STRIPE_DASHBOARD_URL__": {"app_dir": "stripe_dashboard_mock", "kind": "url"},
    "__CUA_GYM_TRELLO_HOST__": {"app_dir": "trello_mock", "kind": "host"},
    "__CUA_GYM_TRELLO_URL__": {"app_dir": "trello_mock", "kind": "url"},
    "__CUA_GYM_TWITTER_URL__": {"app_dir": "twitter_mock", "kind": "url"},
    "__CUA_GYM_UBER_EATS_URL__": {"app_dir": "uber_eats_mock", "kind": "url"},
    "__CUA_GYM_WECHAT_URL__": {"app_dir": "wechat_mock", "kind": "url"},
}

# Episode-isolation modes an app can declare in the YAML ``sessions`` block.
# "sid": hub convention — every episode gets ?sid=<uuid> on its URLs and a
#        best-effort ``/post?sid=`` state reset at close (the default).
# "none": stateless external site (e.g. an OpenStreetMap clone) that neither
#        understands nor needs a session id.
SESSION_MODE_SID = "sid"
SESSION_MODE_NONE = "none"
SESSION_MODES = frozenset({SESSION_MODE_SID, SESSION_MODE_NONE})

_OVERRIDES_CACHE: dict[str, Any] | None = None
_OVERRIDES_CACHE_PATH: str | None = None


def endpoint_overrides() -> dict[str, Any]:
    """Per-deployment endpoint config from the YAML file named by $CUA_GYM_ENDPOINTS_YAML.

    Schema::

        endpoints:                       # app_dir -> absolute base URL; wins over
          webarena_gitlab_mock: http://203.0.113.20:8001   # base_port + APP_DIRS index math
        placeholders:                    # extra tokens merged over PLACEHOLDER_MAP
          __CUA_GYM_WEBARENA_GITLAB_URL__: {app_dir: webarena_gitlab_mock, kind: url}
        sessions:                        # app_dir -> episode-isolation mode (default: sid)
          webarena_map_mock: none        # stateless site: no ?sid= on URLs, no /post reset

    Returns ``{"endpoints": {}, "placeholders": {}, "sessions": {}}`` when the env var is
    unset, so the builtin base-port convention applies unchanged. Validation is eager: a
    malformed file fails the first substitution loudly instead of sending an episode to a
    half-built URL.
    """
    global _OVERRIDES_CACHE, _OVERRIDES_CACHE_PATH
    path = os.environ.get(ENDPOINTS_YAML_ENV) or None
    if _OVERRIDES_CACHE is not None and path == _OVERRIDES_CACHE_PATH:
        return _OVERRIDES_CACHE
    overrides: dict[str, Any] = {"endpoints": {}, "placeholders": {}, "sessions": {}}
    if path is not None:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for app_dir, url in (data.get("endpoints") or {}).items():
            if not isinstance(url, str) or "://" not in url:
                raise ValueError(f"{path}: endpoints.{app_dir} must be an absolute URL, got {url!r}")
            overrides["endpoints"][app_dir] = url.rstrip("/")
        for token, spec in (data.get("placeholders") or {}).items():
            app_dir = (spec or {}).get("app_dir")
            kind = (spec or {}).get("kind", "url")
            if kind not in ("url", "host"):
                raise ValueError(f"{path}: placeholders.{token}.kind must be 'url' or 'host', got {kind!r}")
            if app_dir not in overrides["endpoints"] and app_dir not in APP_DIRS:
                raise ValueError(f"{path}: placeholders.{token} references unknown app_dir {app_dir!r}")
            overrides["placeholders"][token] = {"app_dir": app_dir, "kind": kind}
        for app_dir, mode in (data.get("sessions") or {}).items():
            if mode not in SESSION_MODES:
                raise ValueError(f"{path}: sessions.{app_dir} must be one of {sorted(SESSION_MODES)}, got {mode!r}")
            if app_dir not in overrides["endpoints"] and app_dir not in APP_DIRS:
                raise ValueError(f"{path}: sessions.{app_dir} references unknown app_dir {app_dir!r}")
            overrides["sessions"][app_dir] = mode
    _OVERRIDES_CACHE, _OVERRIDES_CACHE_PATH = overrides, path
    return overrides
