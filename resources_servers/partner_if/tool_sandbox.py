# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run each episode of a task's own tool backend in a fresh sandbox, for a sandboxed mock-tool server.

The sandbox holds the generic runner (tool_runner.py), the protocol module the backend imports (tool_backend.py)
and the row's `tool_backend` bundle, nothing else. The backend is given only the row's `responses_create_params`,
which the actor sees anyway: rubrics, checks, the contract and the judge stay in the server. Starting an episode
installs the bundle's pinned requirements, starts the runner, checks that the backend serves every tool the task
advertises and creates the episode's state; calls then reach the runner through the sandbox's endpoint for its
port. The server stops the sandbox when it closes or retires the episode or shuts down, and the sandbox's
`ttl_s` is the backstop.
"""

import asyncio
import json
import logging
import re
import secrets
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from aiohttp import ClientError, ClientTimeout
from pydantic import Field, JsonValue

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxEndpoint
from nemo_gym.sandbox.agent_tools import restricted_network_policy
from nemo_gym.sandbox.config import SandboxConfig, resolve_provider_config
from nemo_gym.server_utils import request
from resources_servers.partner_if import tool_backend, tool_runner
from resources_servers.partner_if.task_data import ToolBackendBundle
from resources_servers.partner_if.tool_backend import ToolCallError
from resources_servers.partner_if.tool_runner import BUNDLE_DIR, LIB_DIR, SITE_DIR, TOKEN_HEADER


logger = logging.getLogger(__name__)

ROOT = "/opt/tool_backend"
RUNNER_PORT = 8000
# Every sandbox's files besides the bundle, by path under ROOT. The protocol module sits where a backend imports it.
RUNTIME_FILES = {
    "tool_runner.py": Path(tool_runner.__file__).read_text(),
    f"{LIB_DIR}/resources_servers/partner_if/tool_backend.py": Path(tool_backend.__file__).read_text(),
}


class ToolSandboxConfig(SandboxConfig):
    """A sandboxed mock-tool server's `sandbox_config`: SandboxConfig's settings for each episode's sandbox, and
    the image, network and timeouts of the backend in it."""

    image: str = "python:3.13-slim"
    # The backstop that stops a sandbox whose close never came.
    ttl_s: float = Field(default=3600, gt=0)
    # package_index: egress only to the hosts of package_index_egress, to install each bundle's requirements from
    # package_index_url. none: no egress; the image must already have the requirements. Both need OpenSandbox's
    # network policies. provider_default: no policy from this server, for a provider without them such as Docker;
    # requirements are installed.
    network: Literal["package_index", "none", "provider_default"] = "package_index"
    package_index_url: str = "https://pypi.org/simple"
    package_index_egress: list[str] = Field(
        default_factory=lambda: ["https://pypi.org", "https://files.pythonhosted.org"]
    )
    install_timeout_s: float = Field(default=300, gt=0)
    # Until the runner has loaded the backend and its fixture.
    runner_timeout_s: float = Field(default=60, gt=0)
    # Each request to the runner, such as one tool call.
    call_timeout_s: float = Field(default=60, gt=0)


class UnservedToolsError(ValueError):
    """The task advertises tools its backend does not serve: a defect of the task's data, found at its first seed."""


def network_options(config: ToolSandboxConfig, provider: str) -> dict[str, Any]:
    """The provider options that apply `config.network`; a restricted network fails closed without OpenSandbox."""
    if config.network == "provider_default":
        return {}
    if "network_policy" in config.provider_options:
        raise ValueError("Set the tool sandbox's network with sandbox_config.network, not a network_policy option")
    if config.network == "package_index":
        return {"network_policy": restricted_network_policy(provider, config.package_index_egress)}
    if provider != "opensandbox":
        raise ValueError("network: none requires the OpenSandbox network-policy provider")
    return {"network_policy": {"defaultAction": "deny", "egress": []}}


def sandbox_files(bundle: ToolBackendBundle, runner: dict[str, JsonValue], *, install: bool) -> dict[str, str]:
    """Everything a backend's sandbox holds, by path under ROOT: the runtime, runner.json and the bundle."""
    files = {
        **RUNTIME_FILES,
        "runner.json": json.dumps(runner),
        **{f"{BUNDLE_DIR}/{path}": text for path, text in bundle.files.items()},
    }
    if install:
        files["requirements.txt"] = "".join(f"{requirement}\n" for requirement in bundle.requirements)
    return files


def _label(value: str) -> str:
    # OpenSandbox turns sandbox metadata into Kubernetes labels.
    return re.sub(r"[^\w.-]", "-", value)[:63].strip("._-") or "unknown"


class SandboxEpisode:
    """One episode's tools, served by its task's own backend in the episode's own sandbox."""

    def __init__(self, sandbox: AsyncSandbox, *, name: str, token: str, call_timeout_s: float, verify_tls: bool):
        self.sandbox = sandbox
        self.name = name
        self.requires_final_answer = False
        self._token = token
        self._call_timeout_s = call_timeout_s
        self._verify_tls = verify_tls
        self._endpoint: SandboxEndpoint | None = None

    @property
    def sandbox_id(self) -> str | None:
        return self.sandbox.sandbox_id

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> JsonValue:
        # Not resent after a dropped connection: the backend may already have run the call.
        payload = await self._expect("POST", "call", {"name": name, "arguments": arguments}, resend=False)
        if "tool_call_error" in payload:
            raise ToolCallError(payload["tool_call_error"])
        return payload["output"]

    async def artifacts(self) -> dict[str, str]:
        return (await self._expect("GET", "artifacts"))["artifacts"]

    async def receipts(self) -> dict[str, JsonValue]:
        return (await self._expect("GET", "receipts"))["receipts"]

    async def stop(self) -> None:
        """Stop the sandbox; a failed stop raises and can be retried."""
        await self.sandbox.stop()

    async def _request(
        self, method: str, route: str, body: dict | None = None, *, resend: bool = True, tries: int = 3
    ) -> tuple[int, dict]:
        response = await request(
            method,
            f"{self._endpoint.endpoint.rstrip('/')}/{route}",
            headers={**self._endpoint.headers, TOKEN_HEADER: self._token},
            json=body,
            timeout=ClientTimeout(total=self._call_timeout_s),
            ssl=self._verify_tls,
            # A connection that was never made is retried; one that dropped is retried only if `resend`.
            _max_num_tries=tries,
            _resend_after_disconnect=resend,
        )
        payload = json.loads(await response.read() or b"{}")
        return response.status, payload if isinstance(payload, dict) else {}

    async def _expect(self, method: str, route: str, body: dict | None = None, *, resend: bool = True) -> dict:
        status, payload = await self._request(method, route, body, resend=resend)
        if status != 200:
            # The runner's traceback, for the server's log only.
            raise RuntimeError(f"Tool backend {self.name} failed in /{route} (HTTP {status}): {payload.get('error')}")
        return payload

    async def _wait_until_ready(self, timeout_s: float) -> None:
        loop = asyncio.get_running_loop()
        deadline, last = loop.time() + timeout_s, "no response"
        while True:
            try:
                status, payload = await self._request("GET", "requires_final_answer", tries=1)
            except (ClientError, OSError, asyncio.TimeoutError, ValueError) as error:
                last = f"{type(error).__name__}: {error}"
            else:
                if payload.get("status") == "ready":
                    self.requires_final_answer = bool(payload["requires_final_answer"])
                    return
                if payload.get("status") == "failed":
                    raise RuntimeError(f"Tool backend {self.name} failed to load:\n{payload.get('error')}")
                # Still loading, or the provider's proxy has no route to the runner yet.
                last = f"HTTP {status}"
            if loop.time() >= deadline:
                break
            await asyncio.sleep(0.5)
        try:
            log = (await self.sandbox.exec(f"tail -c 4000 {ROOT}/runner.log", timeout_s=30)).stdout
        except Exception as error:
            log = f"unavailable ({error!r})"
        raise RuntimeError(
            f"Tool backend {self.name} was not ready after {timeout_s:g}s (last: {last}); runner log:\n{log}"
        )


class ToolSandboxes:
    """Starts the episodes of one server's sandboxed tasks, each in its own sandbox from the server's provider."""

    def __init__(self, sandbox_provider: str, config: ToolSandboxConfig, *, server_name: str):
        self._sandbox_provider = sandbox_provider
        self._provider_config = resolve_provider_config(sandbox_provider, get_global_config_dict())
        self._network = network_options(config, next(iter(self._provider_config)))
        # Endpoints go through the provider's server, so the runner is reached with the provider's own TLS setting:
        # an OpenSandbox deployment with a self-signed certificate sets `connection.tls_verify: false`.
        connection = next(iter(self._provider_config.values())).get("connection") or {}
        self._verify_tls = connection.get("tls_verify", True) is not False
        self._config = config
        self._server_name = server_name

    async def start(
        self, task_id: str, bundle: ToolBackendBundle, row: dict[str, JsonValue], advertised: set[str]
    ) -> SandboxEpisode:
        """A ready episode of the task: its backend serves `advertised` and holds a fresh state."""
        config = self._config
        token = secrets.token_urlsafe(32)
        runner = {
            "entrypoint": bundle.entrypoint,
            "fixture": bundle.fixture,
            "requirements": bundle.requirements,
            "host": "0.0.0.0",
            "port": RUNNER_PORT,
            "token": token,
        }
        install = bool(bundle.requirements) and config.network != "none"
        spec = config.spec(
            self._sandbox_provider,
            image=config.image,
            workdir=None,
            metadata={"task": _label(task_id), "nemo_gym_agent": _label(self._server_name)},
            files={f"{ROOT}/{path}": text for path, text in sandbox_files(bundle, runner, install=install).items()},
            named_configs=get_global_config_dict(),
        )
        spec = replace(spec, ports=(RUNNER_PORT,), provider_options=spec.provider_options | self._network)
        episode = SandboxEpisode(
            AsyncSandbox(self._provider_config),
            name=bundle.entrypoint,
            token=token,
            call_timeout_s=config.call_timeout_s,
            verify_tls=self._verify_tls,
        )
        # The backend gets what the actor sees, never the row's contract.
        actor_row = {"responses_create_params": row["responses_create_params"]}

        async def setup(sandbox: AsyncSandbox) -> None:
            if install:
                installed = await sandbox.exec(
                    "python3 -m pip install --disable-pip-version-check --no-input --quiet --no-cache-dir "
                    f"--only-binary=:all: --index-url {shlex.quote(config.package_index_url)} "
                    f"--target {ROOT}/{SITE_DIR} -r {ROOT}/requirements.txt",
                    timeout_s=config.install_timeout_s,
                )
                if installed.return_code != 0 or installed.error_type:
                    output = (installed.stderr or "") + (installed.stdout or "")
                    raise RuntimeError(
                        f"Installing {bundle.entrypoint}'s requirements failed ({installed.error_type or 'exit'}): "
                        f"{output[-2000:]}"
                    )
            started = await sandbox.exec(
                # Only the runner goes to the background, with none of this command's streams, so the command
                # ends at once; a backgrounded `cd && ...` list would hold them open and never end.
                f"nohup python3 -I {ROOT}/tool_runner.py {ROOT}/runner.json > {ROOT}/runner.log 2>&1 < /dev/null &",
                timeout_s=60,
                preserve_background_services=True,
            )
            if started.return_code != 0 or started.error_type:
                raise RuntimeError(
                    f"Starting the tool runner failed ({started.error_type or 'exit'}): {started.stderr or started.stdout}"
                )
            episode._endpoint = await sandbox.endpoint(RUNNER_PORT)
            await episode._wait_until_ready(config.runner_timeout_s)
            served = (await episode._expect("POST", "tools", {"row": actor_row}))["tools"]
            if unserved := advertised - set(served):
                raise UnservedToolsError(
                    f"Task {task_id} advertises tools its backend does not serve: {', '.join(sorted(unserved))}"
                )
            await episode._expect("POST", "new_state", {"row": actor_row}, resend=False)

        try:
            await episode.sandbox.start_with_setup(spec, setup)
        except BaseException:
            try:
                # Also releases the provider of a sandbox that never started.
                await episode.stop()
            except Exception:
                logger.exception("Could not stop the tool sandbox of task %s; its ttl_s will", task_id)
            raise
        return episode
