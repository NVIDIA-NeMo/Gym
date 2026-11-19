# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Egress rules that limit a sandbox to a few hosts, for agents and for resources servers that lend a sandbox."""

import ipaddress
from urllib.parse import urlsplit


def reachable_host(url: str) -> str:
    """Return the host of ``url``, or raise ``ValueError`` if a remote sandbox cannot reach it (loopback or none)."""
    target = (urlsplit(url).hostname or "").rstrip(".")
    try:
        address = ipaddress.ip_address(target) if target else None
    except ValueError:
        address = None
    if not target or target == "localhost" or (address and (address.is_loopback or address.is_unspecified)):
        raise ValueError(
            "Model/tool endpoint must advertise a sandbox-reachable host. Set use_absolute_ip=true "
            "or bind the server to an explicitly reachable host; rewriting a loopback URL is insufficient."
        )
    return target


def restricted_network_policy(provider_name: str, urls: list[str]) -> dict[str, object]:
    """Fail closed when a provider or endpoint cannot support network isolation."""
    if provider_name != "opensandbox":
        raise ValueError("Restricted network access requires the OpenSandbox network-policy provider")
    targets = set()
    for url in urls:
        targets.add(reachable_host(url))
    return {"defaultAction": "deny", "egress": [{"action": "allow", "target": t} for t in sorted(targets)]}
