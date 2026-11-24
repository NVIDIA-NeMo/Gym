# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``scale_live``: the public MCP-Atlas backend, one upstream agent-environment sandbox per rollout.

The sandbox runs the real MCP servers of ``ghcr.io/scaleapi/mcp-atlas`` against live third-party services with
the operator's credentials, so this backend owns the credential scoping and preflight (``credentials.py``), the
sandbox lifecycle (``sandbox.py``) and the recognition of service and authentication failures in tool errors
(``authentication.py``). It is the default backend and the only one in the public tree.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from aiohttp import ClientError

from nemo_gym.sandbox import resolve_provider_config, resolve_provider_metadata


try:
    from ..authentication import is_authentication_failure, tool_error_diagnostic, tool_server_name
    from ..credentials import (
        SERVICE_ENV_NAMES,
        CredentialConfigurationError,
        configured_dataset_servers,
        enabled_servers,
        task_environment,
        validate_workload_environment,
    )
    from ..sandbox import AtlasSandbox, SandboxError, SandboxPool, call_sandbox_tool
    from .base import (
        BackendConfigurationError,
        BackendStartError,
        BackendTask,
        BackendTransportError,
        EnvironmentDestroyedError,
        MCPAtlasBackend,
        ToolOutcome,
        ToolResult,
    )
except ImportError:
    # Gym starts a resources server from its own directory, so the server's modules import at top level.
    from authentication import is_authentication_failure, tool_error_diagnostic, tool_server_name
    from backends.base import (
        BackendConfigurationError,
        BackendStartError,
        BackendTask,
        BackendTransportError,
        EnvironmentDestroyedError,
        MCPAtlasBackend,
        ToolOutcome,
        ToolResult,
    )
    from credentials import (
        SERVICE_ENV_NAMES,
        CredentialConfigurationError,
        configured_dataset_servers,
        enabled_servers,
        task_environment,
        validate_workload_environment,
    )
    from sandbox import AtlasSandbox, SandboxError, SandboxPool, call_sandbox_tool


logger = logging.getLogger(__name__)

# The image's uvx and npx servers install packages when they start. These user configs pin that resolution to
# the 1.2.7 image date: later releases (mcp 2.x) break several servers. They are files, not variables, because
# MCP stdio clients pass only HOME, PATH and a few more variables to the servers they start.
DEFAULT_SANDBOX_FILES = {
    "/root/.config/uv/uv.toml": 'exclude-newer = "2026-07-17T00:00:00Z"\n',
    "/root/.npmrc": "before=2026-07-17\nupdate-notifier=false\n",
}
# Apptainer's image paths may be read-only; its development sandboxes need writable scratch caches.
APPTAINER_SANDBOX_FILES = {
    "/root/.config/uv/uv.toml": 'exclude-newer = "2026-07-17T00:00:00Z"\ncache-dir = "/tmp/uv-cache"\n',
    "/root/.npmrc": "before=2026-07-17\ncache=/tmp/npm-cache\nupdate-notifier=false\n",
}

# Require an HTTP status or a diagnostic at the start of an error, rather than matching words/numbers
# in model arguments (for example an object ID of 502 or a query containing "credits").
SERVICE_HTTP_STATUSES = {401, 402, 403, 429, 502, 503, 504}
SERVICE_ERROR_PATTERN = re.compile(
    r"^(?:HTTP(?:/\d(?:\.\d)?)?\s+(?:401|402|403|408|429|5\d\d)\b|"
    r"(?:HTTPError|ClientResponseError):\s*(?:401|402|403|408|429|5\d\d)\b|"
    r"unauthori[sz]ed\b|forbidden\b|authentication failed\b|(?:API )?rate[ _-]?limit exceeded\b|"
    r"You have exceeded a secondary rate limit\b|"
    r"too many requests\b|payment required\b|certificate verify failed\b|"
    r"(?:SSLError|SSLCertVerificationError|ConnectionError|ConnectTimeout|ReadTimeout):|"
    r"connection (?:refused|reset|closed|aborted)\b|name or service not known\b|"
    r"name resolution failed\b|getaddrinfo failed\b|econnrefused\b|enotfound\b|etimedout\b|econnreset\b|"
    r"client failed to connect\b|session was closed\b)",
    re.IGNORECASE,
)
# These tools return model-written program/file outcomes. Their error text is not a service diagnostic.
MODEL_RESULT_SERVERS = frozenset(
    {
        "calculator",
        "cli-mcp-server",
        "desktop-commander",
        "filesystem",
        "git",
        "memory",
        "mcp-code-executor",
        "mcp-server-code-runner",
    }
)
# These tools run model-written programs or commands inside the sandbox, so they can end its own server. A
# transport failure after one of them, while the provider still answers for the sandbox, is scored (see
# EnvironmentDestroyedError); the call order alone is not evidence, the probe is.
PROCESS_SERVERS = frozenset({"cli-mcp-server", "desktop-commander", "mcp-code-executor", "mcp-server-code-runner"})
PROBE_TIMEOUT_S = 30
# Services a deployment's egress may be unable to reach. Such a reply is a tool result the model sees and the
# row is scored, as upstream does; the count lets a report label the affected rows without rescanning
# trajectories. Each entry: the server, and the text its tools return when the service is unavailable.
KNOWN_UNAVAILABLE_MARKERS: dict[str, tuple[str, str]] = {
    "ddg_empty_results": ("ddg-search", "No results were found"),
    "overpass_unreachable": ("osm-mcp-server", "Cannot connect to host overpass-api.de"),
}


def _is_service_error(status: int, text: str, tool_name: str) -> bool:
    if status in SERVICE_HTTP_STATUSES:
        return True
    error, wrapped = tool_error_diagnostic(text, tool_name)
    # The image wraps upstream exceptions in "Failed to call tool:"; only there does a leading status
    # have HTTP meaning. A number elsewhere in a tool's message is not a service failure.
    if wrapped and re.match(r"(?:401|402|403|429|502|503|504)\b", error):
        return True
    return (status >= 400 or wrapped) and SERVICE_ERROR_PATTERN.match(error) is not None


class ScaleLiveBackend(MCPAtlasBackend):
    """Per-rollout sandboxes of the upstream image, or one shared agent-environment URL for development."""

    name: ClassVar[str] = "scale_live"

    def __init__(self, *, config: Any, global_config_dict: Mapping[str, Any]) -> None:
        if config.backend_options:
            raise BackendConfigurationError(
                "the scale_live backend reads its settings from the sandbox_* fields; backend_options must be empty"
            )
        if bool(config.sandbox_image) == bool(config.sandbox_base_url):
            raise ValueError("set exactly one of sandbox_image or sandbox_base_url")
        if config.sandbox_base_url and not config.sandbox_development_mode:
            raise ValueError("sandbox_base_url requires sandbox_development_mode=true; do not use it for scores")
        self.config = config
        self.pool: SandboxPool | None = None
        self.provider_name = "shared_base_url"
        if not config.sandbox_image:
            return
        provider = resolve_provider_config(config.sandbox_provider, global_config_dict)
        self.provider_name = next(iter(provider))
        if self.provider_name == "apptainer":
            if not config.sandbox_development_mode:
                raise ValueError("Apptainer requires sandbox_development_mode=true; do not use it for scores")
            if not os.path.isfile(config.sandbox_image):
                raise ValueError(f"sandbox_image is not a file: {config.sandbox_image}")
        elif self.provider_name == "opensandbox":
            if os.path.isabs(config.sandbox_image) or config.sandbox_image.endswith(".sif"):
                raise ValueError("OpenSandbox sandbox_image must be an OCI image reference, not a local SIF")
            policy = config.sandbox_options.get("network_policy")
            if not isinstance(policy, dict) or policy.get("defaultAction") != "deny":
                raise ValueError(
                    "OpenSandbox requires sandbox_options.network_policy with defaultAction=deny; "
                    "deployment enforcement and allowed external services must be validated before scoring"
                )
            from opensandbox.models.sandboxes import NetworkPolicy

            encoded_policy = NetworkPolicy.model_validate(policy).model_dump(
                by_alias=True, mode="json", exclude_unset=True
            )
            if encoded_policy != policy:
                raise ValueError(
                    "network_policy contains fields the installed OpenSandbox SDK drops or changes; "
                    "use a policy the SDK preserves exactly"
                )
        else:
            raise ValueError("MCP-Atlas supports OpenSandbox, or Apptainer in explicit development mode")
        sandbox_files = config.sandbox_files
        if sandbox_files is None:
            sandbox_files = APPTAINER_SANDBOX_FILES if self.provider_name == "apptainer" else DEFAULT_SANDBOX_FILES
        unknown_names = sorted(set(config.service_env_names) - set(SERVICE_ENV_NAMES))
        if unknown_names:
            raise CredentialConfigurationError(f"unknown service credential names: {', '.join(unknown_names)}")
        services = {name: os.environ[name] for name in config.service_env_names if os.environ.get(name)}
        environment = {**services, **config.sandbox_env}
        required = configured_dataset_servers(
            global_config_dict, resources_server=config.name, root=Path(__file__).resolve().parents[3]
        )
        validate_workload_environment(environment)
        if required.servers:
            task_environment(environment, sorted(required.servers))
        for path in required.deferred_inputs:
            logger.warning(
                "MCP-Atlas credential preflight awaits prepared input %s; each task is checked before startup", path
            )
        self.pool = SandboxPool(
            image=config.sandbox_image,
            sandbox_provider=provider,
            sandbox_options=config.sandbox_options,
            sandbox_metadata=resolve_provider_metadata(config.sandbox_provider, global_config_dict),
            sandbox_env=environment,
            sandbox_files=dict(sandbox_files),
            port_range=config.sandbox_port_range,
            max_concurrency=config.sandbox_max_concurrency,
            startup_timeout_s=config.sandbox_startup_timeout_s,
            ttl_s=config.sandbox_ttl_s,
            scratch_root=config.sandbox_scratch_root,
        )

    async def start(self, session_id: str, task: BackendTask) -> AtlasSandbox:
        servers = enabled_servers(list(task.servers))
        if self.pool is None:
            # The shared development environment serves every task; nothing to start or scope.
            names = {name: name for name in task.tools}
            return AtlasSandbox(base_url=self.config.sandbox_base_url, tool_names=names, missing_tools=[])
        try:
            return await self.pool.start(session_id, servers=servers, tools=sorted(task.tools))
        except SandboxError as exc:
            raise BackendStartError(str(exc)) from exc

    async def call_tool(
        self, handle: AtlasSandbox, task: BackendTask, tool_name: str, arguments: dict[str, Any], *, timeout_s: float
    ) -> ToolResult:
        if tool_name == "met-museum_get-museum-object":
            # Upstream's harness does the same: image blocks do not fit a text tool output.
            arguments = {**arguments, "returnImage": False}
        server = tool_server_name(tool_name, task.servers)
        if server in PROCESS_SERVERS:
            handle.process_tool_calls += 1
        try:
            status, payload = await call_sandbox_tool(handle, tool_name, arguments, timeout_s=timeout_s)
        except asyncio.TimeoutError:
            raise
        except (ClientError, OSError, SandboxError) as exc:
            if handle.destroyed or await self._destroyed_from_inside(handle):
                handle.destroyed = True
                raise EnvironmentDestroyedError(type(exc).__name__, str(exc)) from exc
            raise BackendTransportError(type(exc).__name__, str(exc)) from exc
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        for counter, (marker_server, marker) in KNOWN_UNAVAILABLE_MARKERS.items():
            if server == marker_server and marker in text:
                handle.tool_outcomes[counter] = handle.tool_outcomes.get(counter, 0) + 1
        return ToolResult(status, payload)

    async def _destroyed_from_inside(self, handle: AtlasSandbox) -> bool:
        """Whether the sandbox still answers the provider although its server does not, after a program tool ran.

        A shared development environment has no provider to ask, and a sandbox no program or command ran in
        has no model-caused explanation, so both stay infrastructure failures.
        """
        if not handle.process_tool_calls or handle.sandbox is None or handle.stopped:
            return False
        try:
            result = await handle.sandbox.exec("true", timeout_s=PROBE_TIMEOUT_S)
        except Exception as exc:
            logger.info("MCP-Atlas sandbox probe failed after a transport error; provider loss: %s", exc)
            return False
        return result.error_type is None and result.return_code == 0

    def classify_tool_error(
        self, handle: AtlasSandbox, task: BackendTask, tool_name: str, status: int, diagnostic: str
    ) -> ToolOutcome:
        served_name = handle.tool_names.get(tool_name, tool_name)
        if is_authentication_failure(
            status, diagnostic, servers=task.servers, tool_name=tool_name, served_name=served_name
        ):
            return ToolOutcome.AUTHENTICATION_FAILURE
        if tool_server_name(tool_name, task.servers) not in MODEL_RESULT_SERVERS and _is_service_error(
            status, diagnostic, served_name
        ):
            return ToolOutcome.SERVICE_ERROR
        return ToolOutcome.MODEL_RESULT

    def provenance(self, handle: AtlasSandbox | None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "backend": self.name,
            "sandbox_image": self.config.sandbox_image or None,
            "sandbox_provider": self.provider_name,
            "development_mode": self.config.sandbox_development_mode,
        }
        if handle is not None and handle.tool_outcomes:
            record["known_unavailable"] = dict(handle.tool_outcomes)
        if handle is not None and handle.destroyed:
            record["environment_destroyed"] = True
        return record

    async def stop(self, session_id: str) -> bool:
        if self.pool is None:
            return False
        return await self.pool.stop(session_id)

    async def reap_expired(self) -> int:
        if self.pool is None:
            return 0
        return await self.pool.reap_expired()

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
