# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BrowseComp agent whose bash_command runs in a per-rollout sandbox.

The loop, the model calls, search, browse, and verification are unchanged from
BrowsecompAgent and stay on the host. Only bash_command moves: before each call
the agent copies the session's new pages from the resources server into the
sandbox, then runs the same read-only pocketshell there (pod_bash.py).
"""

import asyncio
import base64
import io
import tarfile
import tempfile
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from typing import Any, Mapping, Optional
from uuid import uuid4

from anyio import CancelScope
from fastapi import Body, Request, Response
from pydantic import Field, PrivateAttr

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.browsecomp_agent.app import (
    BrowsecompAgent,
    BrowsecompAgentConfig,
    BrowsecompAgentRunRequest,
    BrowsecompAgentVerifyResponse,
)


# Carries the rollout's sandbox key from run() to the /v1/responses call it makes.
SANDBOX_KEY_METADATA = "ng_browsecomp_sandbox_key"

_HARNESS_DIR = Path(__file__).resolve().parents[2] / "resources_servers" / "browsecomp_advanced_harness"
_RUNTIME_FILES = ("bash_guard.py", "pod_bash.py")
_POCKETSHELL_DIR = _HARNESS_DIR / "pocketshell" / "src" / "pocketshell"
_WORKSPACE_DISABLED = "[bash error: terminal workspace is not enabled for this server]"


class BrowsecompSandboxedAgentConfig(BrowsecompAgentConfig):
    # A sandbox block name (e.g. "sandbox") or an inline {provider: {...}} mapping.
    sandbox_provider: str | dict[str, Any] = "sandbox"
    # SandboxSpec fields: image, ttl_s, ready_timeout_s, resources, env, metadata, provider_options.
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_root: str = "/tmp/ng_browsecomp"
    sandbox_python: str = "python3"
    # Same clamp as the resources server's bash_timeout_s, so the model sees the same budget.
    bash_timeout_s: float = 60.0
    # Margin over the command budget before the outer exec timeout fires.
    bash_exec_grace_s: float = 30.0


@dataclass
class _RolloutSandbox:
    sandbox: Optional[AsyncSandbox] = None
    synced: int = 0  # pages already copied into the sandbox
    bash_calls: int = 0
    started_at: Optional[float] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _sandbox_label(sandbox: AsyncSandbox) -> str:
    handle = getattr(sandbox, "_handle", None)
    return f"provider={getattr(handle, 'provider_name', '?')} id={getattr(handle, 'sandbox_id', '?')}"


_ROLLOUT: ContextVar[Optional[_RolloutSandbox]] = ContextVar("browsecomp_sandboxed_rollout", default=None)


def _tarball(entries: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _runtime_tarball() -> bytes:
    entries = {name: (_HARNESS_DIR / name).read_bytes() for name in _RUNTIME_FILES}
    for path in sorted(_POCKETSHELL_DIR.glob("*.py")):
        entries[f"pocketshell/{path.name}"] = path.read_bytes()
    return _tarball(entries)


class BrowsecompSandboxedAgent(BrowsecompAgent):
    config: BrowsecompSandboxedAgentConfig
    _rollouts: dict[str, _RolloutSandbox] = PrivateAttr(default_factory=dict)

    async def run(self, request: Request, body: BrowsecompAgentRunRequest) -> BrowsecompAgentVerifyResponse:
        key = uuid4().hex
        state = self._rollouts[key] = _RolloutSandbox()
        body.responses_create_params.metadata = {
            **(body.responses_create_params.metadata or {}),
            SANDBOX_KEY_METADATA: key,
        }
        try:
            return await super().run(request, body)
        finally:
            self._rollouts.pop(key, None)
            # Shielded: a cancelled run must still release its sandbox, not leave it to the TTL.
            with CancelScope(shield=True):
                if state.sandbox is not None:
                    status = "ok"
                    try:
                        await state.sandbox.stop()
                    except Exception as e:
                        status = f"failed:{e!r}"
                    lifetime = time.monotonic() - state.started_at if state.started_at is not None else -1.0
                    print(
                        f"[browsecomp_sandboxed][sandbox_stop] {_sandbox_label(state.sandbox)} status={status} "
                        f"lifetime_s={lifetime:.1f} bash_calls={state.bash_calls} pages={state.synced}",
                        flush=True,
                    )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        metadata = dict(body.metadata or {})
        state = self._rollouts.get(metadata.pop(SANDBOX_KEY_METADATA, None))
        body = body.model_copy(update={"metadata": metadata or None})
        token = _ROLLOUT.set(state)
        try:
            return await super().responses(request, response, body)
        finally:
            _ROLLOUT.reset(token)

    async def _call_resources_tool(self, name: str, tool_args: dict, cookies) -> tuple[int, Any, str]:
        state = _ROLLOUT.get()
        if name != "bash_command" or state is None or not isinstance(tool_args, dict):
            # Malformed bash arguments also go to the host, which answers with its usual 422.
            return await super()._call_resources_tool(name, tool_args, cookies)
        try:
            duration = max(1.0, min(self.config.bash_timeout_s, float(tool_args.get("duration") or 10)))
        except (TypeError, ValueError):
            return await super()._call_resources_tool(name, tool_args, cookies)
        keystrokes = str(tool_args.get("keystrokes") or "")
        async with state.lock:
            return 200, cookies, await self._bash_in_sandbox(state, keystrokes, duration, cookies)

    async def _bash_in_sandbox(self, state: _RolloutSandbox, keystrokes: str, duration: float, cookies) -> str:
        sync = await self._workspace_files(state.synced, cookies)
        if not sync["enabled"]:
            return _WORKSPACE_DISABLED
        if state.sandbox is None:
            started = time.monotonic()
            state.sandbox = await self._start_sandbox()
            state.started_at = time.monotonic()
            print(
                f"[browsecomp_sandboxed][sandbox_start] {_sandbox_label(state.sandbox)} "
                f"start_s={state.started_at - started:.1f}",
                flush=True,
            )
        state.bash_calls += 1
        root = self.config.sandbox_root
        if sync["files"]:
            pages = {f["path"]: f["content"].encode() for f in sync["files"]}
            await self._extract(state.sandbox, _tarball(pages), f"{root}/ws")
            state.synced = sync["next"]

        encoded = base64.b64encode(keystrokes.encode()).decode()
        command = " ".join(
            quote(part)
            for part in (self.config.sandbox_python, f"{root}/rt/pod_bash.py", f"{root}/ws", str(duration), encoded)
        )
        result = await state.sandbox.exec(command, timeout_s=duration + self.config.bash_exec_grace_s)
        if result.error_type == "timeout":
            return f"[command timed out after {duration:.0f}s]\n[exit_code=-1]"
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(
                f"pod_bash failed in sandbox: return_code={result.return_code} "
                f"error_type={result.error_type} stderr={(result.stderr or '')[:300]!r}"
            )
        return result.stdout or ""

    async def _workspace_files(self, since: int, cookies) -> dict[str, Any]:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/workspace_files",
            json={"since": since},
            cookies=cookies,
        )
        await raise_for_status(response)
        return await get_response_json(response)

    async def _start_sandbox(self) -> AsyncSandbox:
        global_config = get_global_config_dict() if isinstance(self.config.sandbox_provider, str) else None
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config))
        options = dict(self.config.sandbox_config)
        resources = SandboxResources.from_mapping(options.pop("resources", None))
        metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        metadata = metadata | options.pop("metadata", {}) | {"nemo_gym_agent": self.config.name}
        sandbox = AsyncSandbox(provider)
        await sandbox.start(SandboxSpec(resources=resources, metadata=metadata, **options))
        try:
            await self._extract(sandbox, _runtime_tarball(), f"{self.config.sandbox_root}/rt")
        except BaseException:
            await sandbox.stop()
            raise
        return sandbox

    async def _extract(self, sandbox: AsyncSandbox, tarball: bytes, target_dir: str) -> None:
        remote = f"{self.config.sandbox_root}/upload-{uuid4().hex}.tar.gz"
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as local:
            local.write(tarball)
            local.flush()
            await sandbox.upload(local.name, remote)
        result = await sandbox.exec(
            f"mkdir -p {quote(target_dir)} && tar -xzf {quote(remote)} -C {quote(target_dir)} && rm -f {quote(remote)}",
            timeout_s=120,
        )
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"Could not unpack files into the sandbox: {result.stderr or result.stdout!r}")


if __name__ == "__main__":
    BrowsecompSandboxedAgent.run_webserver()
