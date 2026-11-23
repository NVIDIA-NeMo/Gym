# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep every tab of an episode in the episode's hub session.

Hub mock apps find their session id in ``?sid=`` and fall back to a per-tab
``sessionStorage`` key (``gitlab_sid``, ``reddit_sid``, ...). The worker puts the
sid only on the start URL, so a tab the agent opens itself (``tabs_create``,
Ctrl+T plus a typed URL, a ``target=_blank`` link) has neither: the app runs on
the default session, and work done there is never scored (a Qwen screen did a
whole task in such a tab and scored 0).

``sid_init_script`` builds a browser-context init script, installed once per
episode before the first page opens, that runs in every document of every tab.
On hub origins only (never on ``sessions: none`` apps such as the map) it sets
each known sid key that is missing and, in top-level documents, adds the sid to
the URL when it is missing (some app code reads ``location.search`` directly).
A sid already present in the URL or in sessionStorage is never overwritten.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from resources_servers.cuagym.hub_apps import SESSION_MODE_NONE, SESSION_MODE_SID


# sessionStorage keys under which hub apps keep the sid. Collected from
# CUA-Gym-Hub 3cfb89093 websites/*/src (sessionStorage.getItem/setItem literals and
# SID_KEY constants); an app reads only its own key, so setting all of them on a
# hub origin is harmless.
HUB_SID_STORAGE_KEYS: tuple[str, ...] = (
    # WebArena mocks
    "gitlab_sid",
    "reddit_sid",
    "webarena_shopping_sid",
    "shopping_admin_sid",
    "classifieds_sid",
    # other hub apps
    "mock_sid",
    "12306_sid",
    "aliyun_sid",
    "amplitude_sid",
    "bamboohr_sid",
    "canvas_sid",
    "clio_sid",
    "cloudflare_sid",
    "coinbase_mock_sid",
    "contractbook_sid",
    "dingtalk_sid",
    "docs_mock_sid",
    "expedia_sid",
    "expensify_sid",
    "ga4_mock_sid",
    "gitlab_mock_sid",
    "greenhouse_sid",
    "gusto_sid",
    "hotjar_sid",
    "hs_sid",
    "klaviyo_sid",
    "lattice_sid",
    "linear_sid",
    "looker_studio_sid",
    "mailchimp_mock_sid",
    "meta_ads_sid",
    "mixpanel_sid",
    "openreview_sid",
    "sap_mock_sid",
    "stripe_mock_sid",
    "tableau_sid",
    "uber_eats_sid",
    "vercel_mock_sid",
    "wandb_sid",
    "wc_sid",
    "zendesk_sid",
)

_DEFAULT_PORTS = {"http": 80, "https": 443}


def origin_of(url: str) -> str | None:
    """``scheme://host[:port]`` as ``window.location.origin`` reports it, or None."""
    try:
        parsed = urlparse(str(url).strip())
        port = parsed.port
    except ValueError:
        return None
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


@dataclass(frozen=True)
class HubScope:
    """Where the episode's sid applies: exact origins, plus any port of the hub host."""

    origins: tuple[str, ...]
    hosts: tuple[str, ...]
    excluded_origins: tuple[str, ...]

    def contains(self, url: str) -> bool:
        origin = origin_of(url)
        if origin is None or origin in self.excluded_origins:
            return False
        return origin in self.origins or (urlparse(origin).hostname or "") in self.hosts


def hub_scope(
    *,
    base_url: str,
    base_session_mode: str,
    hub_base_url: str,
    endpoints: Mapping[str, str],
    sessions: Mapping[str, str],
) -> HubScope:
    """Origins of the sid-convention apps an episode can reach.

    ``endpoints``/``sessions`` are the $CUA_GYM_ENDPOINTS_YAML blocks; apps without
    an endpoint entry live on the hub host at base_port + index, hence ``hosts``.
    The episode's own app (``base_url``) counts unless it is a ``sessions: none`` app.
    """
    excluded = {origin_of(url) for app, url in endpoints.items() if sessions.get(app) == SESSION_MODE_NONE}
    origins = {
        origin_of(url) for app, url in endpoints.items() if sessions.get(app, SESSION_MODE_SID) == SESSION_MODE_SID
    }
    if base_session_mode == SESSION_MODE_NONE:
        excluded.add(origin_of(base_url))
    else:
        origins.add(origin_of(base_url))
    hub_host = (urlparse(hub_base_url).hostname or "").lower()
    excluded.discard(None)
    origins.discard(None)
    return HubScope(
        origins=tuple(sorted(origins - excluded)),
        hosts=(hub_host,) if hub_host else (),
        excluded_origins=tuple(sorted(excluded)),
    )


_SCRIPT_TEMPLATE = """(() => {
  const cfg = %s;
  try {
    const loc = window.location;
    if (cfg.excluded.indexOf(loc.origin) !== -1) return;
    if (cfg.origins.indexOf(loc.origin) === -1 && cfg.hosts.indexOf(loc.hostname) === -1) return;
    try {
      const store = window.sessionStorage;
      for (const key of cfg.keys) {
        if (!store.getItem(key)) store.setItem(key, cfg.sid);
      }
    } catch (e) {}
    if (window.top === window && (loc.protocol === "http:" || loc.protocol === "https:")) {
      const params = new URLSearchParams(loc.search);
      if (!params.get("sid")) {
        params.set("sid", cfg.sid);
        window.history.replaceState(window.history.state, "", loc.pathname + "?" + params.toString() + loc.hash);
      }
    }
  } catch (e) {}
})();
"""


def sid_init_script(sid: str, scope: HubScope, keys: Iterable[str] = HUB_SID_STORAGE_KEYS) -> str:
    """JavaScript for ``BrowserContext.add_init_script`` (see the module docstring)."""
    if not sid or not all(c.isalnum() or c in "-_" for c in sid):
        raise ValueError(f"unexpected episode sid: {sid!r}")
    cfg = {
        "sid": sid,
        "keys": list(dict.fromkeys(keys)),
        "origins": list(scope.origins),
        "hosts": list(scope.hosts),
        "excluded": list(scope.excluded_origins),
    }
    return _SCRIPT_TEMPLATE % json.dumps(cfg, sort_keys=True)
