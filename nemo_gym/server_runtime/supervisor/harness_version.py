# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve an opt-in harness version once, before the supervisor starts its workers."""

import hashlib
import logging
import re
from pathlib import Path

import requests
from omegaconf import DictConfig, OmegaConf, open_dict
from packaging.version import InvalidVersion, Version

from nemo_gym.config.harness_version import harness_version_target


LOG = logging.getLogger(__name__)
# OpenCode publishes two release lines. The release marked latest by its upstream
# repository determines which adapter runs, just as an explicit version does.
_RELEASES = {
    "opencode_agent": (
        "opencode_version",
        "https://api.github.com/repos/anomalyco/opencode/releases/latest",
        "tag_name",
    ),
    "goose_agent": ("goose_version", "https://api.github.com/repos/aaif-goose/goose/releases/latest", "tag_name"),
    "prime_agent": (
        "prime_agent_version",
        "https://api.github.com/repos/PrimeIntellect-ai/prime-agent/releases/latest",
        "tag_name",
    ),
    "pool_agent": ("pool_version", "https://downloads.poolside.ai/pool/pool-latest-version.txt", ""),
}
_NPM = {
    "openclaw_agent": ("openclaw_version", "openclaw"),
    "pi_agent": ("pi_version", "@earendil-works/pi-coding-agent"),
    "claude_code_agent": ("claude_code_version", "@anthropic-ai/claude-code"),
    "cline_agent": ("cline_version", "cline"),
    "codex_agent": ("codex_version", "@openai/codex"),
    "copilot_agent": ("copilot_version", "@github/copilot"),
    "droid_agent": ("droid_version", "@factory/cli"),
    "kimi_code_agent": ("kimi_version", "@moonshot-ai/kimi-code"),
    "kilocode_agent": ("kilo_version", "@kilocode/cli"),
    "qwen_code_agent": ("qwen_code_version", "@qwen-code/qwen-code"),
}
_SOURCE = {
    "nooa_agent": (
        "NVIDIA-NeMo/labs-OO-Agents",
        {"nooa": "", "nooa-cli": "packages/nooa-cli", "nooa-bench": "packages/nooa-bench"},
    ),
    "simple_strands_agent": ("strands-labs/benchmark-harnesses", {"simple-strands-agent": "simple-strands-agent"}),
    "tau2": ("bxyu-nvidia/tau2-bench", {"tau2": ""}),
}
_PYTHON = {
    "crewai_agent": ("crewai_version", ("crewai",)),
    "langchain_deepagents_agent": ("deepagents_version", ("deepagents",)),
    "langgraph_agent": ("langgraph_version", ("langgraph",)),
    "mini_swe_agent": ("mini_swe_version", ("mini-swe-agent",)),
    "miniswe_sandboxed_agent": ("mini_swe_version", ("mini-swe-agent",)),
    "mini_swe_agent_2": ("mini_swe_version", ("mini-swe-agent",)),
    "nemo_fabric_agent": ("nemo_fabric_version", ("nemo-fabric",)),
    "openhands_agent": ("openhands_version", ("openhands-sdk", "openhands-tools")),
    "stirrup_agent": ("stirrup_version", ("stirrup",)),
    "terminus_2_agent": ("harbor_version", ("harbor",)),
    "terminus_2_sandboxed_agent": ("harbor_version", ("harbor",)),
    "verifiers_agent": ("verifiers_version", ("verifiers",)),
}
_RELEASES["opencode_sandboxed_agent"] = _RELEASES["opencode_agent"]
_NPM["pi_sandboxed_agent"] = _NPM["pi_agent"]
_HOST_NPM_CLI = {
    "claude_code_agent": "claude",
    "cline_agent": "cline",
    "codex_agent": "codex",
    "kilocode_agent": "kilo",
    "opencode_agent": "opencode",
    "pi_agent": "pi",
}
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?")
_SHA = re.compile(r"[0-9a-fA-F]{40}")
_HERMES_REPO = "https://github.com/NousResearch/hermes-agent"


def _lookup(url: str, key: str) -> str:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        if key:
            value = response.json()
            for part in key.split("."):
                value = value[part]
        else:
            value = response.text.strip()
    except (requests.RequestException, KeyError, ValueError, TypeError) as error:
        raise ValueError(f"Could not resolve harness version from {url}: {error}") from error
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {key} in harness metadata from {url}")
    return value


def _version_field(agent: str) -> str | None:
    if agent in _NPM:
        return _NPM[agent][0]
    if agent in _RELEASES:
        return _RELEASES[agent][0]
    if agent in _PYTHON:
        return _PYTHON[agent][0]
    if agent in _SOURCE:
        return f"{agent.removesuffix('_agent')}_version"
    if agent in ("hermes_agent", "hermes_sandboxed_agent"):
        return "hermes_version"
    return None


def resolve_harness_version(agent: str, config: DictConfig) -> str | None:
    """Resolve latest in the adapter's installer field before serializing worker configs."""
    agent, config = harness_version_target(agent, config)
    field = _version_field(agent)
    if agent == "qwen_code_agent" and field not in config:
        field = "qwen_version"  # Input spelling used by older Gym configs.
    if field is None or (requested := config.get(field)) is None:
        return None
    if config.get("harness_repo") and agent not in _SOURCE and agent not in ("hermes_agent", "hermes_sandboxed_agent"):
        raise ValueError(f"{agent} installs published releases and does not support harness_repo")
    if not isinstance(requested, str):
        raise ValueError(f"{agent}: expected a quoted harness version or latest")

    requirement = None
    if agent in ("hermes_agent", "hermes_sandboxed_agent") or agent in _SOURCE:
        default_repo = "https://github.com/" + _SOURCE[agent][0] if agent in _SOURCE else _HERMES_REPO
        repo_field = "harness_repo" if agent in _SOURCE else "hermes_repo"
        repo = config.get(repo_field) or default_repo
        if not isinstance(repo, str) or not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", repo):
            raise ValueError("Expected an HTTPS GitHub repository URL")
        repo = repo.removesuffix(".git")
        api = repo.replace("https://github.com/", "https://api.github.com/repos/", 1)
        if requested == "latest":
            ref = "HEAD" if agent in _SOURCE else _lookup(f"{api}/releases/latest", "tag_name")
        else:
            ref = requested
        if _SHA.fullmatch(ref):
            resolved = ref.lower()
        else:
            if ref == "HEAD" and agent in _SOURCE:
                tag = "HEAD"
            elif _VERSION.fullmatch(ref.removeprefix("v")):
                tag = f"v{ref.removeprefix('v')}"
            else:
                raise ValueError(f"{agent}: expected a release, full Git commit SHA, or latest")
            resolved = _lookup(f"{api}/commits/{tag}", "sha")
            if not _SHA.fullmatch(resolved):
                raise ValueError(f"Invalid Git commit for {repo} at {tag}")
        if agent == "hermes_agent":
            requirement = f"git+{repo}.git@{resolved}"
        with open_dict(config):
            config[repo_field] = repo
    elif agent in _PYTHON:
        resolved = requested
        if requested == "latest":
            package = _PYTHON[agent][1][0]
            resolved = _lookup(f"https://pypi.org/pypi/{package}/json", "info.version")
        try:
            resolved = str(Version(resolved))
        except InvalidVersion as error:
            raise ValueError(f"{agent}: expected an exact Python package version or latest") from error
    elif agent in _NPM and requested != "latest" and not _VERSION.fullmatch(requested):
        return None  # The native installer owns validation of exact pins and other supported npm specs.
    else:
        if agent in _NPM:
            registry = (
                config.get("npm_registry", "https://registry.npmjs.org")
                if agent == "copilot_agent"
                else "https://registry.npmjs.org"
            )
            package = _NPM[agent][1].replace("/", "%2f")
            url, key = f"{registry.rstrip('/')}/{package}/latest", "version"
        else:
            _, url, key = _RELEASES[agent]
        resolved = _lookup(url, key).removeprefix("v") if requested == "latest" else requested
        if agent == "pool_agent":
            resolved = resolved.removeprefix("v")
        if not _VERSION.fullmatch(resolved):
            raise ValueError(f"{agent}: expected an exact release or latest, got {resolved!r}")

    with open_dict(config):
        config[field] = f"v{resolved}" if agent == "pool_agent" else resolved
    LOG.info("%s harness version %s resolved to %s", agent, requested, requirement or resolved)
    return requirement


def python_harness_setup(agent: str, config: DictConfig, global_config: DictConfig) -> DictConfig:
    """Give a selected Python harness its own component venv and resolver overrides."""
    field = _version_field(agent)
    if (agent not in _PYTHON and agent not in _SOURCE) or field is None or not (version := config.get(field)):
        return global_config
    if agent in _SOURCE:
        repo = config.harness_repo
        requirements = [
            f"{package} @ git+{repo}.git@{version}" + (f"#subdirectory={subdir}" if subdir else "")
            for package, subdir in _SOURCE[agent][1].items()
        ]
        key = hashlib.sha256("\n".join(requirements).encode()).hexdigest()
    else:
        requirements = [f"{package}=={version}" for package in _PYTHON[agent][1]]
        key = version
    root = Path(global_config.uv_venv_dir).absolute() / ".harness-versions" / key
    return OmegaConf.merge(
        global_config,
        {
            "uv_venv_dir": str(root),
            "head_server_deps": [*(global_config.get("head_server_deps") or ()), *requirements],
            "head_server_constraints": [*(global_config.get("head_server_constraints") or ()), *requirements],
        },
    )


def harness_environment(agent: str, config: DictConfig, *, hermes_source: str | None) -> dict[str, str]:
    """Set worker-local installer inputs for adapters whose runtime also has an environment interface."""
    field = _version_field(agent)
    version = config.get(field) if field else None
    if not version:
        return {}
    if hermes_source:
        return {"HERMES_SPEC": hermes_source}
    if agent in _HOST_NPM_CLI and _VERSION.fullmatch(version):
        return {"NEMO_GYM_SELECTED_NPM_CLI": _HOST_NPM_CLI[agent]}
    if agent == "mini_swe_agent":
        return {"NEMO_GYM_MINI_SWE_VERSION": version}
    if agent == "tau2":
        return {"NEMO_GYM_TAU2_BENCH_REPO_URL": config.harness_repo, "NEMO_GYM_TAU2_BENCH_REF": version}
    if agent == "nooa_agent":
        return {"NEMO_GYM_NOOA_SOURCE": f"{config.harness_repo}/archive/{version}.tar.gz"}
    return {}
