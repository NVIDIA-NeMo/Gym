# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser egress allowlist for CUA-Gym episodes.

Chromium sends every host lookup through ``--host-resolver-rules``. The rule
``MAP * ~NOTFOUND`` makes each lookup fail with ``net::ERR_NAME_NOT_RESOLVED``;
``EXCLUDE <pattern>`` exempts a host (``*`` and ``?`` wildcards, port-agnostic).
IP-literal URLs go through the same rules, so every hub IP must be listed: the
mock hubs and the map deployments are all IP literals.

The list mirrors the real WebArena harness copy
(``projects/webarena_eval/osworld_internal/webarena/common/browser.py``,
``_chrome_args()`` with ``WA_EGRESS_ALLOWLIST=1``): the benchmark hosts, loopback,
and the map site's public dependencies (tile hosts, FOSSGIS routing, GraphHopper,
MeMoMaps). One addition: ``overpass-api.de``, the ``OVERPASS_URL`` of the mock
map's "Query features" tool. The mock 5-site apps load nothing from other hosts
(checked on 2026-10-17: every asset, font and image is same-origin).

The rule is browser-only. Setup and reward programs, and the judge proxy that
reward programs call, run as Python subprocesses of the browser worker and are
not affected. Endpoints that are not websites stay out of the list: the launcher
adds the judge proxy (``webarena_judge: http://<head ip>:<port>``) to the endpoints
YAML, and an EXCLUDE is per host, so allowing it would open every port of the
Ray head node to the browser.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from urllib.parse import urlparse


LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1")

# Endpoint entries that reward programs use but the browser never needs.
NON_BROWSER_ENDPOINT_APPS: tuple[str, ...] = ("webarena_judge",)

# Public hosts that the OpenStreetMap-based map site loads in the browser.
MAP_DEPENDENCY_HOSTS: tuple[str, ...] = (
    "tile.openstreetmap.org",  # "Standard" layer tiles
    "*.tile.openstreetmap.org",  # e.g. gps.tile.openstreetmap.org overlay
    "*.openstreetmap.de",  # FOSSGIS routing / Valhalla direction engines
    "graphhopper.com",  # GraphHopper direction engine
    "*.graphhopper.com",
    "*.memomaps.de",  # OPNVKarte layer
    "overpass-api.de",  # "Query features" (OVERPASS_URL of the mock map)
)

_HOST_PATTERN = re.compile(r"^[a-z0-9*?]([a-z0-9.*?-]*[a-z0-9*?])?$")


def normalize_host(entry: str) -> str:
    """Return the lowercase hostname pattern of an allowlist entry.

    Accepts a bare host or pattern (``*.example.org``), ``host:port``, or a URL
    (``http://203.0.113.10:8001/x``). Raises ``ValueError`` for anything that
    cannot be a Chromium host pattern (commas or spaces would break the rule).
    """
    text = str(entry).strip().lower()
    if "://" in text:
        host = urlparse(text).hostname or ""
    else:
        host = text.split("/", 1)[0]
        head, sep, port = host.rpartition(":")
        if sep and port.isdigit():
            host = head
    if not host or not _HOST_PATTERN.match(host):
        raise ValueError(f"not a valid egress allowlist host: {entry!r}")
    return host


def hub_hosts(
    hub_base_url: str,
    endpoints: Mapping[str, str],
    exclude_apps: Iterable[str] = NON_BROWSER_ENDPOINT_APPS,
) -> list[str]:
    """Hosts of the hub base URL and of every browser endpoint in $CUA_GYM_ENDPOINTS_YAML."""
    skip = set(exclude_apps)
    hosts: list[str] = []
    for url in [hub_base_url, *(url for app, url in endpoints.items() if app not in skip)]:
        if url:
            hosts.append(normalize_host(url))
    return list(dict.fromkeys(hosts))


def egress_allow_hosts(
    *,
    hub_base_url: str,
    endpoints: Mapping[str, str],
    extra: Iterable[str] = (),
    exclude_apps: Iterable[str] = NON_BROWSER_ENDPOINT_APPS,
) -> list[str]:
    """Ordered, de-duplicated allowlist: hub hosts, loopback, map dependencies, extras."""
    hosts = [
        *hub_hosts(hub_base_url, endpoints, exclude_apps),
        *LOOPBACK_HOSTS,
        *MAP_DEPENDENCY_HOSTS,
        *(normalize_host(item) for item in extra),
    ]
    return list(dict.fromkeys(hosts))


def host_resolver_rules(allow_hosts: Iterable[str]) -> str:
    """The Chromium rule string: fail every lookup except the allowed hosts."""
    return ", ".join(["MAP * ~NOTFOUND", *(f"EXCLUDE {normalize_host(host)}" for host in allow_hosts)])


def host_resolver_rules_arg(allow_hosts: Iterable[str]) -> str:
    return "--host-resolver-rules=" + host_resolver_rules(allow_hosts)


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    # Chromium's base::MatchPattern: '*' matches any run of characters, '?' one character.
    return re.compile("".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern) + r"\Z")


def host_allowed(host: str, allow_hosts: Iterable[str]) -> bool:
    """Whether Chromium would resolve ``host`` under the rule built from ``allow_hosts``."""
    name = (host or "").strip().lower()
    return bool(name) and any(_pattern_regex(normalize_host(p)).match(name) for p in allow_hosts)
