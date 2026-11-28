# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution of inlined CUA-Gym episode code (setup and reward programs).

Task rows carry the bundle's setup and reward programs as code strings
(``cuagym.initial_setup`` / ``cuagym.eval_reward_code``), pre-transformed at
data-conversion time so that:

- the session id appears only as the ``__CUA_GYM_SID__`` placeholder (no
  ``/tmp`` sid handoff files);
- hub endpoints appear as ``__CUA_GYM_<APP>_URL__`` placeholders;
- no GUI browser is launched by the code.

At episode time the server substitutes the placeholders with plain
``str.replace`` and pipes the program to a Python subprocess over stdin —
nothing is unpacked or written to disk.
"""

from __future__ import annotations

import re
import subprocess
import sys

from resources_servers.cuagym.hub_apps import (
    APP_DIRS,
    PLACEHOLDER_MAP,
    SESSION_MODE_SID,
    endpoint_overrides,
)


SID_PLACEHOLDER = "__CUA_GYM_SID__"
_REWARD_LINE = re.compile(r"REWARD:\s*([-+0-9.eE]+)")
_PLACEHOLDER_TOKEN = re.compile(r"__CUA_GYM_[A-Z0-9_]+__")

# Episode programs come from the dataset, not from this server: run them with
# a minimal environment so Gym's merged config (which carries model API keys)
# and cluster credentials are never exposed to dataset-supplied code.
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TZ", "HOME", "TMPDIR", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")


def app_port(app_dir: str, base_port: int) -> int:
    try:
        return base_port + APP_DIRS.index(app_dir)
    except ValueError as exc:
        raise ValueError(f"unknown CUA-Gym-Hub app dir: {app_dir!r}") from exc


def app_url(app_dir: str, hub_base_url: str, base_port: int, fallback: str | None = None) -> str:
    """Base URL of an app, in precedence order.

    1. ``endpoints.<app_dir>`` in ``$CUA_GYM_ENDPOINTS_YAML`` (operator override);
    2. ``fallback`` — the row's own ``cuagym.external_base_url`` for apps that are
       not part of the hub (e.g. ``webarena_map_mock``);
    3. the hub convention ``hub_base_url:base_port + APP_DIRS index``.
    """
    endpoint = endpoint_overrides()["endpoints"].get(app_dir)
    if endpoint is not None:
        return endpoint
    if fallback:
        if "://" not in fallback:
            raise ValueError(f"external_base_url for {app_dir!r} must be an absolute URL, got {fallback!r}")
        return fallback.rstrip("/")
    return f"{hub_base_url}:{app_port(app_dir, base_port)}"


def app_session(app_dir: str) -> str:
    """Episode-isolation mode for an app: ``sessions.<app_dir>`` from the YAML, else ``sid``."""
    return endpoint_overrides()["sessions"].get(app_dir, SESSION_MODE_SID)


def placeholder_values(hub_base_url: str, base_port: int) -> dict[str, str]:
    """Resolve every known endpoint placeholder against the configured hub.

    Placeholders whose app has an explicit URL in $CUA_GYM_ENDPOINTS_YAML resolve
    to that URL; the rest fall back to the hub base_port + APP_DIRS index math.
    """
    overrides = endpoint_overrides()
    values: dict[str, str] = {}
    host = hub_base_url.split("://", 1)[-1]
    for placeholder, spec in {**PLACEHOLDER_MAP, **overrides["placeholders"]}.items():
        endpoint = overrides["endpoints"].get(spec["app_dir"])
        if endpoint is not None:
            values[placeholder] = endpoint.split("://", 1)[-1] if spec["kind"] == "host" else endpoint
            continue
        port = app_port(spec["app_dir"], base_port)
        if spec["kind"] == "host":
            values[placeholder] = f"{host}:{port}"
        else:
            values[placeholder] = f"{hub_base_url}:{port}"
    return values


def substitute(code: str, *, sid: str, hub_base_url: str, base_port: int) -> str:
    """Fill sid and hub-endpoint placeholders into an episode program.

    Raises if any ``__CUA_GYM_*__`` placeholder survives: an unresolved
    endpoint would otherwise reach the app as a literal URL and fail silently
    (setup appearing to succeed, reward scoring 0.0 for the wrong reason).
    """
    code = code.replace(SID_PLACEHOLDER, sid)
    for placeholder, value in placeholder_values(hub_base_url, base_port).items():
        code = code.replace(placeholder, value)
    unresolved = sorted(set(_PLACEHOLDER_TOKEN.findall(code)))
    if unresolved:
        raise ValueError(f"unresolved CUA-Gym placeholders: {unresolved[:5]}")
    return code


def run_code(
    code: str,
    *,
    timeout_seconds: float,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an episode program in a subprocess, passing the source via stdin.

    The child gets an allowlisted environment only: episode programs are
    dataset-supplied, and the server process holds Gym's merged config
    (``NEMO_GYM_CONFIG_DICT``, including model API keys) plus cluster
    credentials in its own environment.
    """
    import os
    import tempfile

    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    if extra_env:
        env.update(extra_env)
    # Run from a scratch directory so relative-path writes by episode code
    # cannot land in the server's working tree.
    with tempfile.TemporaryDirectory(prefix="cuagym-run-") as cwd:
        return subprocess.run(
            [sys.executable, "-"],
            input=code,
            env=env,
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
        )


def parse_reward(stdout: str) -> float | None:
    """Return the last REWARD: value printed by a reward program, if any."""
    matches = _REWARD_LINE.findall(stdout or "")
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except ValueError:
        return None
    if value != value:  # NaN
        return None
    return min(max(value, 0.0), 1.0)


def tail(text: str, limit: int = 2000) -> str:
    text = text or ""
    return text if len(text) <= limit else text[-limit:]
