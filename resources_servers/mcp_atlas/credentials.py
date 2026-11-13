# Source: https://github.com/scaleapi/mcp-atlas/tree/5c11ac44e96d31b91b92a0affad5f2648255907a
# Original file: services/agent-environment/src/agent_environment/mcp_server_template.json (credential references).
# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2026 Scale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# NVIDIA modifications: independently implemented credential validation, task scoping, and dataset preflight.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and scope the service credentials required by an MCP-Atlas task."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from nemo_gym.config import INPUT_READ_ROOTS_KEY_NAME
from nemo_gym.core import is_materialized_task_row
from nemo_gym.data import normalize_task_fields


# Configuration facts from the pinned agent-environment template, independently implemented here:
# https://github.com/scaleapi/mcp-atlas/blob/5c11ac44e96d31b91b92a0affad5f2648255907a/services/agent-environment/src/agent_environment/mcp_server_template.json
# The recorded 1.2.7 amd64 image template has the same per-server placeholder requirements.
SERVER_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "airtable": ("AIRTABLE_API_KEY",),
    "alchemy": ("ALCHEMY_API_KEY",),
    "arxiv": (),
    "brave-search": ("BRAVE_API_KEY",),
    "calculator": (),
    "cli-mcp-server": (),
    "clinicaltrialsgov-mcp-server": (),
    "context7": (),
    "ddg-search": (),
    "desktop-commander": (),
    "e2b-server": ("E2B_API_KEY",),
    "exa": ("EXA_API_KEY",),
    "fetch": (),
    "filesystem": (),
    "git": (),
    "github": ("GITHUB_TOKEN",),
    "google-maps": ("GOOGLE_MAPS_API_KEY",),
    "google-workspace": ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"),
    "lara-translate": ("LARA_ACCESS_KEY_ID", "LARA_ACCESS_KEY_SECRET"),
    "mcp-code-executor": (),
    "mcp-server-code-runner": (),
    "memory": (),
    "met-museum": (),
    "mongodb": ("MONGODB_CONNECTION_STRING",),
    "national-parks": ("NPS_API_KEY",),
    "notion": ("NOTION_TOKEN",),
    "open-library": (),
    "osm-mcp-server": (),
    "oxylabs": ("OXYLABS_USERNAME", "OXYLABS_PASSWORD"),
    "pubmed": (),
    "slack": ("SLACK_MCP_XOXC_TOKEN", "SLACK_MCP_XOXD_TOKEN"),
    "twelvedata": ("TWELVE_DATA_API_KEY",),
    "weather": (),
    "weather-data": ("WEATHER_API_KEY",),
    "whois": (),
    "wikipedia": (),
}
SERVICE_ENV_NAMES = tuple(name for names in SERVER_CREDENTIALS.values() for name in names)
HOST_CREDENTIAL_NAMES = frozenset(
    {"OPENSANDBOX_API_KEY", "INFERENCE_API_KEY", "NVAUTH_TOKEN", "HF_TOKEN", "AWS_ACCESS_KEY_ID"}
)


class CredentialConfigurationError(ValueError):
    """The requested task cannot run with the configured service credentials."""


def enabled_servers(servers: object) -> list[str]:
    """Validate task server names before using them for credential checks or provider configuration."""
    if not isinstance(servers, list) or not servers or not all(isinstance(server, str) for server in servers):
        raise CredentialConfigurationError("enabled servers must be a nonempty list of server names")
    unknown = sorted(set(servers) - SERVER_CREDENTIALS.keys())
    if unknown:
        raise CredentialConfigurationError(f"unknown MCP-Atlas servers: {', '.join(unknown)}")
    return servers


def _host_credential(name: str) -> bool:
    return name in HOST_CREDENTIAL_NAMES or (
        name not in SERVICE_ENV_NAMES
        and re.search(
            r"(?:^|_)(?:API_KEY|TOKEN|PASSWORD|SECRET|PRIVATE_KEY|SECRET_ACCESS_KEY|CREDENTIALS)$", name.upper()
        )
        is not None
    )


def validate_provider_environment(provider: Mapping[str, object]) -> None:
    """Reject credentials inherited through the provider's setup environment before SDK allocation.

    OpenSandbox merges ``setup.env`` underneath ``SandboxSpec.env``. Omitting an unrelated credential from
    the spec would therefore leave it visible. Noncredential provider settings and connection identity stay
    untouched; task service credentials must instead enter through the scoped task environment.
    """
    for block in provider.values():
        if not isinstance(block, Mapping):
            continue
        setup = block.get("setup") or {}
        environment = setup.get("env", {}) if isinstance(setup, Mapping) else getattr(setup, "env", {})
        if not isinstance(environment, Mapping):
            raise CredentialConfigurationError("provider setup.env must be a mapping")
        if not all(isinstance(name, str) and isinstance(value, str) for name, value in environment.items()):
            raise CredentialConfigurationError("provider setup.env names and values must be strings")
        forbidden = sorted(name for name in environment if name in SERVICE_ENV_NAMES or _host_credential(name))
        if forbidden:
            raise CredentialConfigurationError(
                f"credentials must not enter MCP-Atlas through provider setup.env: {', '.join(forbidden)}"
            )


def validate_workload_environment(environment: Mapping[str, str]) -> None:
    """Reject host credentials independently of whether a configured task inventory is available."""
    forbidden = sorted(name for name in environment if _host_credential(name))
    if forbidden:
        raise CredentialConfigurationError(
            f"host credentials must not enter MCP-Atlas workloads: {', '.join(forbidden)}"
        )


def task_environment(environment: Mapping[str, str], servers: list[str]) -> dict[str, str]:
    """Require the task's credentials and exclude every unrelated known service credential.

    Noncredential configuration is retained. An explicit ``sandbox_env`` value is subject to the same
    scope and nonblank checks as a credential obtained from the host environment.
    """
    servers = enabled_servers(servers)
    validate_workload_environment(environment)
    required = {name for server in servers for name in SERVER_CREDENTIALS[server]}
    missing = sorted(name for name in required if not environment.get(name, "").strip())
    if missing:
        raise CredentialConfigurationError(f"missing service credentials for enabled servers: {', '.join(missing)}")
    return {name: value for name, value in environment.items() if name not in SERVICE_ENV_NAMES or name in required}


@dataclass(frozen=True)
class DatasetServerRequirements:
    """Servers found in configured input rows and inputs awaiting an explicit prepare script."""

    servers: frozenset[str]
    deferred_inputs: tuple[Path, ...]


def _dataset_declarations(
    global_config: Mapping[str, object], resources_server: str
) -> Iterator[Mapping[str, object]]:
    for name, component in global_config.items():
        if isinstance(component, Mapping):
            resources = component.get("resources_servers", {})
            if name == resources_server and isinstance(resources, Mapping):
                for resource in resources.values():
                    if isinstance(resource, Mapping):
                        yield resource
            agents = component.get("responses_api_agents", {})
            if isinstance(agents, Mapping):
                for agent in agents.values():
                    if isinstance(agent, Mapping):
                        target = agent.get("resources_server")
                        if isinstance(target, Mapping) and target.get("name") == resources_server:
                            yield agent


def _row_servers(row: object) -> list[str]:
    if not isinstance(row, Mapping):
        raise CredentialConfigurationError("task row must be an object")
    task_input = row["task_input"] if is_materialized_task_row(row) else row
    if not isinstance(task_input, Mapping):
        raise CredentialConfigurationError("task_input must be an object")
    task_data = task_input.get("task_data", {})
    if not isinstance(task_data, Mapping):
        raise CredentialConfigurationError("task_data must be an object")
    conflicts = sorted(
        key for key in task_data.keys() & task_input.keys() if key != "task_data" and task_data[key] != task_input[key]
    )
    if conflicts:
        raise CredentialConfigurationError(f"conflicting task_data fields: {', '.join(conflicts)}")
    fields = {**task_data, **{key: value for key, value in task_input.items() if key != "task_data"}}
    fields, conflicts = normalize_task_fields(fields)
    if conflicts:
        raise CredentialConfigurationError(f"conflicting task fields: {', '.join(conflicts)}")
    return enabled_servers(fields.get("servers"))


def _dataset_path(jsonl_fpath: str | Path, *, read_roots: object, root: Path) -> Path:
    path = Path(jsonl_fpath)
    if path.is_absolute():
        return path
    if read_roots is None:
        return root.absolute() / path  # Direct library callers supply their known input origin.
    if (
        not isinstance(read_roots, Sequence)
        or isinstance(read_roots, (str, bytes))
        or not read_roots
        or not all(isinstance(value, (str, Path)) and Path(value).is_absolute() for value in read_roots)
    ):
        raise CredentialConfigurationError(
            "internal controller read roots must be nonempty absolute resource-host-visible paths; "
            "reload the config in the controller before launching the resources server"
        )
    paths = [Path(value) / path for value in read_roots]
    return next((candidate for candidate in paths if candidate.exists()), paths[0])


def configured_dataset_servers(
    global_config: Mapping[str, object], *, resources_server: str, root: Path
) -> DatasetServerRequirements:
    """Read server requirements only from datasets wired to this resources server.

    Relative reads follow the controller's captured read roots; direct callers provide their known origin in
    ``root``. A declared prepare script permits a missing input to be materialized later; every actual seed
    is checked again before provider allocation, including deferred inputs.
    """
    servers: set[str] = set()
    deferred: set[Path] = set()
    for declaration in _dataset_declarations(global_config, resources_server):
        for dataset in declaration.get("datasets") or ():
            if not isinstance(dataset, Mapping) or not isinstance(dataset.get("jsonl_fpath"), (str, Path)):
                raise CredentialConfigurationError("MCP-Atlas datasets require an explicit jsonl_fpath")
            path = _dataset_path(
                dataset["jsonl_fpath"], read_roots=global_config.get(INPUT_READ_ROOTS_KEY_NAME), root=root
            )
            if not path.exists():
                if dataset.get("prepare_script"):
                    deferred.add(path)
                else:
                    raise CredentialConfigurationError(
                        f"MCP-Atlas dataset does not exist: {path}; use an absolute resource-host-visible "
                        "jsonl_fpath or --search-dir available on the resources-server host"
                    )
            else:
                try:
                    with path.open() as source:
                        for line_number, line in enumerate(source, 1):
                            if line.strip():
                                try:
                                    enabled = _row_servers(json.loads(line))
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise CredentialConfigurationError(
                                        f"invalid MCP-Atlas server metadata at {path}:{line_number}: {exc}"
                                    ) from exc
                                servers.update(enabled)
                except OSError as exc:
                    raise CredentialConfigurationError(
                        f"MCP-Atlas dataset is not readable: {path}; use an absolute resource-host-visible "
                        "jsonl_fpath or --search-dir available on the resources-server host"
                    ) from exc
    return DatasetServerRequirements(frozenset(servers), tuple(sorted(deferred)))
