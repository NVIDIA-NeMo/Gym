# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor phase's ``network_mode`` and ``allowed_hosts`` as an OpenSandbox egress policy.

Harbor's phase modes (``public``, ``no-network``, ``allowlist``) and the host spellings its
``normalize_allowed_hosts`` admits are Harbor semantics, so they live beside the server that applies
them (``sandboxes.py``) rather than under ``nemo_gym.sandbox``, which stays lifecycle-only. The module
imports nothing from the environment server, so every other module may import it.
"""

import ipaddress
import re
from collections.abc import Sequence


# A host name an egress rule can name: labels of letters, digits and hyphens joined by dots, optionally under a
# leading ``*.`` wildcard (the forms Harbor's ``allowed_hosts`` admits and OpenSandbox's egress sidecar matches).
_HOST_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_ALLOWED_HOST = re.compile(rf"^(?:\*\.)?{_HOST_LABEL}(?:\.{_HOST_LABEL})*\.?$")


def _egress_target(host: str, what: str = "allowed_hosts entry") -> str:
    """One ``allowed_hosts`` entry (or another host, named by ``what`` in errors) as an egress rule target: a host
    name, a ``*.`` wildcard or an IP literal.

    The spellings are Harbor's (``normalize_allowed_hosts``): an IPv6 literal is written bare (no brackets, no zone
    id) and a ``*.`` wildcard covers host names only, so a task Harbor refuses at parse is refused here too.
    """
    target = str(host).strip()
    if not target:
        raise ValueError(f"{what} must not be empty")
    if "[" in target or "]" in target:
        raise ValueError(f"{what} {target!r}: IPv6 literals are written without brackets")
    if "%" in target:
        raise ValueError(f"{what} {target!r}: IPv6 zone ids are not supported")
    try:
        return str(ipaddress.ip_address(target))
    except ValueError:
        pass
    if "/" in target:
        try:
            ipaddress.ip_network(target, strict=False)
        except ValueError:
            pass
        else:
            # CIDR ranges are not an egress target the sidecar documents (and were not verified live); fail closed.
            raise ValueError(f"{what} {target!r}: CIDR ranges are not supported by the OpenSandbox egress policy")
    if not _ALLOWED_HOST.match(target):
        raise ValueError(f"{what} {target!r} is not a host name, a *.domain wildcard or an IP address")
    normalized = target.rstrip(".").lower()
    if normalized.startswith("*."):
        try:
            ipaddress.ip_address(normalized[2:])
        except ValueError:
            pass
        else:
            raise ValueError(f"{what} {target!r}: wildcards must target host names, not IP addresses")
    return normalized


def phase_network_policy(
    provider_name: str | None, network_mode: str | None, allowed_hosts: Sequence[str] | None = None
) -> dict[str, object] | None:
    """The provider's create-time network policy for one Harbor phase; ``None`` when the phase is public.

    ``no-network`` denies every destination, ``allowlist`` denies all but ``allowed_hosts`` (host names,
    ``*.domain`` wildcards and bare IP literals as Harbor spells them, in order and without repeats). Like
    :func:`restricted_network_policy`, it fails closed: a provider other than OpenSandbox cannot enforce
    an egress policy, and an entry the policy cannot express is refused rather than dropped.
    """
    if network_mode in (None, "public"):
        if allowed_hosts:
            raise ValueError("allowed_hosts is only valid when network_mode='allowlist'")
        return None
    if network_mode not in ("no-network", "allowlist"):
        raise ValueError(f"Unknown network_mode {network_mode!r}; expected public, no-network or allowlist")
    if provider_name != "opensandbox":
        raise ValueError(
            f"network_mode={network_mode!r} requires the OpenSandbox network-policy provider "
            f"(the task runs on {provider_name or 'an unknown provider'!r})"
        )
    if network_mode == "no-network" and allowed_hosts:
        raise ValueError("allowed_hosts is only valid when network_mode='allowlist'")
    targets = list(dict.fromkeys(_egress_target(host) for host in (allowed_hosts or ())))
    return {"defaultAction": "deny", "egress": [{"action": "allow", "target": target} for target in targets]}


def with_harness_hosts(policy: dict[str, object], hosts: Sequence[str]) -> dict[str, object]:
    """``policy`` (a restricted phase's, from :func:`phase_network_policy`) with an ``allow`` rule for each of
    ``hosts`` after its own rules: the hosts the agent's sandbox must also reach for the harness to work there (the
    model endpoint it calls, where its seed downloads its runtime, the tool servers granted to it).

    Each must be a host name or an IP literal the sandbox can reach: a loopback or unspecified address names the
    sandbox itself, so it is refused rather than opened. Raises ``ValueError``.
    """
    targets = [rule["target"] for rule in policy["egress"]]
    for host in hosts:
        target = _egress_target(host, "host the harness needs")
        try:
            address = ipaddress.ip_address(target)
        except ValueError:
            address = None
        if target == "localhost" or (address is not None and (address.is_loopback or address.is_unspecified)):
            raise ValueError(
                f"host the harness needs {target!r} is a loopback or unspecified address, which names the task's "
                "sandbox itself; advertise a reachable address for the servers the sandbox calls (use_absolute_ip)"
            )
        targets.append(target)
    return {**policy, "egress": [{"action": "allow", "target": target} for target in dict.fromkeys(targets)]}
