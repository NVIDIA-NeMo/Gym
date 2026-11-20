# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve an opt-in harness version once, before the supervisor starts its workers."""

import logging
import re

import requests
from omegaconf import DictConfig, open_dict


LOG = logging.getLogger(__name__)
# OpenCode publishes two release lines. The release marked latest by its upstream
# repository determines which adapter runs, just as an explicit version does.
_RELEASES = {
    "opencode_agent": (
        "opencode_version",
        "https://api.github.com/repos/anomalyco/opencode/releases/latest",
        "tag_name",
    ),
    "openclaw_agent": ("openclaw_version", "https://registry.npmjs.org/openclaw/latest", "version"),
    "pi_agent": ("pi_version", "https://registry.npmjs.org/@earendil-works%2fpi-coding-agent/latest", "version"),
}
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?")
_SHA = re.compile(r"[0-9a-fA-F]{40}")
_HERMES_REPO = "https://github.com/NousResearch/hermes-agent"


def _lookup(url: str, key: str) -> str:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        value = response.json()[key]
    except (requests.RequestException, KeyError, ValueError, TypeError) as error:
        raise ValueError(f"Could not resolve harness version from {url}: {error}") from error
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {key} in harness metadata from {url}")
    return value


def resolve_harness_version(agent: str, config: DictConfig) -> str | None:
    """Set the installer's concrete version. Return the source to pass to the Hermes worker, if selected."""
    field = _RELEASES[agent][0] if agent in _RELEASES else None
    if agent == "hermes_agent":
        field = "hermes_version"
    requested = config.get("harness_version")
    if requested is None:
        if not field:
            return None
        requested = config.get(field)
        if requested is None or (agent != "hermes_agent" and requested != "latest"):
            return None
    if not field:
        raise ValueError(f"{agent} does not support harness_version")
    if not isinstance(requested, str):
        raise ValueError(f"{agent}: expected a quoted harness version or latest")

    requirement = None
    if agent == "hermes_agent":
        repo = config.get("hermes_repo") or _HERMES_REPO
        if not isinstance(repo, str) or not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", repo):
            raise ValueError("hermes_repo must be an HTTPS GitHub repository URL")
        repo = repo.removesuffix(".git")
        api = repo.replace("https://github.com/", "https://api.github.com/repos/", 1)
        ref = _lookup(f"{api}/releases/latest", "tag_name") if requested == "latest" else requested
        if _SHA.fullmatch(ref):
            resolved = ref.lower()
        else:
            if not _VERSION.fullmatch(ref.removeprefix("v")):
                raise ValueError("hermes_version must be a release, full Git commit SHA, or latest")
            tag = f"v{ref.removeprefix('v')}"
            resolved = _lookup(f"{api}/commits/{tag}", "sha")
            if not _SHA.fullmatch(resolved):
                raise ValueError(f"Invalid Git commit for {repo} at {tag}")
        requirement = f"git+{repo}.git@{resolved}"
    else:
        _, url, key = _RELEASES[agent]
        resolved = _lookup(url, key).removeprefix("v") if requested == "latest" else requested
        if not _VERSION.fullmatch(resolved):
            raise ValueError(f"{agent}: expected an exact release or latest, got {resolved!r}")

    with open_dict(config):
        config[field] = resolved
        config["harness_version"] = resolved
    LOG.info("%s harness version %s resolved to %s", agent, requested, requirement or resolved)
    return requirement
