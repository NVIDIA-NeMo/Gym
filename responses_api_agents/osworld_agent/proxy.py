# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict, credential-safe proxy configuration for OSWorld rollouts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_ALLOWED_HOST_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_:[]%"
)


@dataclass(frozen=True)
class ProxyConfigInfo:
    path: str
    sha256: str
    entry_count: int


def parse_env_bool(name: str, default: bool, *, environ: Optional[Mapping[str, str]] = None) -> bool:
    """Parse one opt-in environment flag without truthy-string ambiguity."""

    source = os.environ if environ is None else environ
    raw_value = source.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    value = raw_value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise ValueError(f"{name} must be one of: {accepted}; got {raw_value!r}")


def task_requires_proxy(task_config: Mapping[str, Any]) -> bool:
    """Return the task's strict boolean proxy requirement."""

    value = task_config.get("proxy", False)
    if not isinstance(value, bool):
        raise ValueError(f"OSWorld task field 'proxy' must be boolean, got {value!r}")
    return value


def inspect_proxy_config_file(path: Optional[str]) -> ProxyConfigInfo:
    """Validate a proxy file and return only non-secret provenance."""

    if not path or not path.strip():
        raise ValueError("PROXY_CONFIG_FILE is required when OSWorld proxy support is enabled")
    resolved = os.path.abspath(os.path.expanduser(path.strip()))
    if not os.path.isfile(resolved):
        raise ValueError(f"PROXY_CONFIG_FILE is not a readable file: {resolved}")
    try:
        with open(resolved, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ValueError(f"cannot read PROXY_CONFIG_FILE {resolved}: {exc}") from exc
    try:
        entries = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"PROXY_CONFIG_FILE is not valid UTF-8 JSON: {resolved}: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError("PROXY_CONFIG_FILE must contain a non-empty JSON list")

    for index, entry in enumerate(entries):
        prefix = f"PROXY_CONFIG_FILE entry {index}"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{prefix} must be an object")
        host = str(entry.get("host") or "").strip()
        if not host or any(character not in _ALLOWED_HOST_CHARACTERS for character in host):
            raise ValueError(f"{prefix} has an invalid host")
        try:
            port = int(entry.get("port"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix} port must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError(f"{prefix} port must be in 1..65535")
        protocol = str(entry.get("protocol") or "http").lower()
        if protocol != "http":
            raise ValueError(f"{prefix} protocol must be 'http'")
        username = str(entry.get("username") or "")
        password = str(entry.get("password") or "")
        if bool(username) != bool(password):
            raise ValueError(f"{prefix} must provide both username and password, or neither")

    return ProxyConfigInfo(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        entry_count=len(entries),
    )


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} PROXY_CONFIG_FILE", file=sys.stderr)
        return 2
    try:
        info = inspect_proxy_config_file(argv[1])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # Tab-separated, non-secret fields for the shell launcher.
    print(f"{info.sha256}\t{info.entry_count}\t{info.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
