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

import json
import logging
import sqlite3
import tarfile
from asyncio import Semaphore, to_thread
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from time import time
from typing import Any, Literal
from uuid import uuid4

from anyio import CancelScope
from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.agent_tools import (
    restricted_network_policy,
    sandbox_server_url,
    seed_mcp_servers,
    verify_agent_response,
)
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status


LOG = logging.getLogger(__name__)

GOOSE_VERSION = "1.54.0"
# sha256 of the official static (musl) release archives, keyed by `uname -m`.
GOOSE_ARCHIVE_SHA256 = {
    "x86_64": "777e49b6ae128fae06600f94bf51ff05c81572fce63c1581e9ea94bc49d64e42",  # pragma: allowlist secret
    "aarch64": "baeb3e1b188fce75b444af35b5955f255e68fde87f60a5ba01f032227618a4ed",  # pragma: allowlist secret
}
GOOSE_RELEASE_URL = f"https://github.com/aaif-goose/goose/releases/download/v{GOOSE_VERSION}"


def goose_archive_name(arch: str) -> str:
    """Return the release archive name for a sandbox architecture reported by ``uname -m``."""
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
    if arch not in GOOSE_ARCHIVE_SHA256:
        raise ValueError(f"No Goose {GOOSE_VERSION} release for sandbox architecture {arch!r}")
    return f"goose-{arch}-unknown-linux-musl.tar.gz"


class GooseAgentConfig(BaseResponsesAPIAgentConfig):
    """Goose CLI settings plus sandbox allocation and host-side tool services."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 32
    timeout: float = 3600

    provider: str = "openai"
    model: str = "policy_model"
    # Defaults to the Gym model server, with the rollout prefix used for token capture.
    base_url: str | None = None
    api_key: str = "dummy"  # pragma: allowlist secret
    subagent_model: str | None = None
    delegate: bool = True
    max_turns: int = 1000
    goose_mode: Literal["auto", "approve", "smart_approve", "chat"] = "auto"
    system_prompt: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    # Release archives are fetched inside the sandbox unless a host directory holding them is given.
    goose_release_url: str = GOOSE_RELEASE_URL
    goose_archive_dir: str | None = None

    sandbox_provider: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    network_access: Literal["inherit", "model_only", "model_and_tools"] = "inherit"
    tool_servers: list[ResourcesServerRef] = Field(default_factory=list)
    execution_failure_reward_zero: bool = False
    artifacts_dir: str = "results/goose"


class GooseAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class GooseAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    goose_failed: bool = False
    goose_exit_code: int | None = None
    goose_error_type: str | None = None
    goose_results_dir: str = ""
    goose_subagents: int = 0
    goose_max_concurrent_subagents: int = 0


def read_goose_sessions(db_path: Path) -> list[dict[str, Any]]:
    """Read every session (root and delegate children) with its messages, in creation order."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        sessions = [dict(row) for row in con.execute("select * from sessions order by created_at, id")]
        for session in sessions:
            session["messages"] = [
                {"role": row["role"], "content": json.loads(row["content_json"]), "created": row["created_timestamp"]}
                for row in con.execute(
                    "select role, content_json, created_timestamp from messages where session_id = ? order by id",
                    (session["id"],),
                )
            ]
            session["usage"] = [
                dict(row)
                for row in con.execute(
                    "select input_tokens, output_tokens, cache_read_tokens from usage_ledger where session_id = ?",
                    (session["id"],),
                )
            ]
    finally:
        con.close()
    return sessions


def goose_messages_to_output(messages: list[dict[str, Any]]) -> list[Any]:
    """Convert the root session's Goose messages into Responses output items.

    Goose stores one assistant message per tool call of a turn and repeats the turn's thinking in each, so
    a thinking block equal to the previous one is dropped. User text (the task and injected turn context)
    is input, not output.
    """
    output, last_thinking = [], None
    for message in messages:
        for block in message["content"]:
            kind = block.get("type")
            if message["role"] == "assistant" and kind == "thinking" and block.get("thinking"):
                if block["thinking"] != last_thinking:
                    output.append(
                        NeMoGymResponseReasoningItem(
                            id=f"rs_{uuid4().hex}",
                            summary=[NeMoGymSummary(text=block["thinking"], type="summary_text")],
                        )
                    )
                last_thinking = block["thinking"]
            elif message["role"] == "assistant" and kind == "text" and block.get("text"):
                output.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}", content=[NeMoGymResponseOutputText(text=block["text"])]
                    )
                )
            elif kind == "toolRequest":
                call = block.get("toolCall") or {}
                value = call.get("value") or {}
                output.append(
                    NeMoGymResponseFunctionToolCall(
                        call_id=block["id"],
                        name=value.get("name", ""),
                        arguments=json.dumps(value.get("arguments") or {}),
                    )
                )
            elif kind == "toolResponse":
                result = block.get("toolResult") or {}
                content = (result.get("value") or {}).get("content") or []
                text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
                output.append(
                    NeMoGymFunctionCallOutput(call_id=block["id"], output=text or str(result.get("error", "")))
                )
    return output


def max_concurrent(intervals: list[tuple[int, int]]) -> int:
    """Largest number of closed intervals sharing a point."""
    events = sorted(
        [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals], key=lambda e: (e[0], -e[1])
    )
    best = current = 0
    for _, step in events:
        current += step
        best = max(best, current)
    return best


class GooseAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: GooseAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        # Goose is installed in each sandbox, never on the Gym host.
        super().model_post_init(context)
        self._sem = Semaphore(self.config.concurrency)

    def _model_base_url(self, body: GooseAgentRunRequest) -> str:
        if self.config.base_url:
            return self.config.base_url
        base = sandbox_server_url(
            self.config.model_server.name, require_reachable=self.config.network_access != "inherit"
        )
        return self.base_url_for_run(base, body).rstrip("/") + "/v1"

    async def _start_sandbox(self, seed: dict) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config))
        descriptor = seed.get("sandbox_descriptor")
        if not descriptor and seed.get("sandbox_handle"):
            descriptor = {"sandbox_id": seed["sandbox_handle"], "workdir": seed.get("workdir")}
        if descriptor:
            if self.config.network_access != "inherit":
                raise ValueError("Cannot verify network policy on a resource-owned sandbox")
            return await AsyncSandbox.connect(descriptor, provider=provider)
        options = deepcopy(self.config.sandbox_config)
        resources = SandboxResources.from_mapping(options.pop("resources", {}))
        env = cpu_cap_env(resources.cpu) if options.pop("derive_cpu_env", True) else {}
        env.update(options.pop("env", {}))
        metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        metadata = metadata | options.pop("metadata", {}) | {"nemo_gym_agent": self.config.name}
        if self.config.network_access != "inherit":
            # Allow the endpoint Goose actually calls: a configured base_url, else the Gym model server.
            urls = [self.config.base_url or sandbox_server_url(self.config.model_server.name, require_reachable=True)]
            if self.config.network_access == "model_and_tools":
                if not self.config.tool_servers:
                    raise ValueError("model_and_tools requires tool_servers")
                urls.extend(sandbox_server_url(s.name, require_reachable=True) for s in self.config.tool_servers)
            options.setdefault("provider_options", {})["network_policy"] = restricted_network_policy(
                provider.name, urls
            )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(SandboxSpec(resources=resources, env=env, metadata=metadata, **options))
        return sandbox

    async def _install_goose(self, sandbox: AsyncSandbox, remote: str) -> None:
        """Install the pinned, checksum-verified release. Failures are infrastructure errors."""
        arch = await sandbox.exec(command="uname -m", timeout_s=60)
        name = goose_archive_name((arch.stdout or "").strip())
        archive = f"{remote}/{name}"
        if self.config.goose_archive_dir:
            await sandbox.upload(Path(self.config.goose_archive_dir) / name, archive)
            fetch = ":"
        else:
            url = quote(f"{self.config.goose_release_url.rstrip('/')}/{name}")
            fetch = f"(curl -fsSL -o {quote(archive)} {url} || wget -qO {quote(archive)} {url})"
        command = (
            f"mkdir -p {quote(remote)} && {fetch} "
            f"&& echo {quote(GOOSE_ARCHIVE_SHA256[name.split('-')[1]] + '  ' + archive)} | sha256sum -c - "
            f"&& tar xzf {quote(archive)} -C {quote(remote)} "
            f'&& test "$({quote(remote + "/goose")} --version | tr -d " ")" = {GOOSE_VERSION}'
        )
        result = await sandbox.exec(command=command, timeout_s=900)
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(
                f"Goose {GOOSE_VERSION} install failed: {(result.stderr or result.stdout or '')[-500:]}"
            )

    def _goose_config(self, mcp: dict[str, dict[str, Any]]) -> dict[str, Any]:
        # Goose fills in every other platform extension with its own default on load.
        extensions = {"summon": {"enabled": self.config.delegate}}
        for name, server in mcp.items():
            extensions[name] = {
                "type": "streamable_http",
                "name": name,
                "description": "",
                "uri": server["url"],
                "headers": server["headers"],
                "timeout": int(self.config.timeout),
                "enabled": True,
            }
        return {"extensions": extensions}

    async def _run_goose(
        self, sandbox: AsyncSandbox, body: GooseAgentRunRequest, mcp: dict[str, dict[str, Any]], root: Path
    ) -> dict[str, Any]:
        params = body.responses_create_params
        messages = [{"role": "user", "content": params.input}] if isinstance(params.input, str) else params.input
        prompt, system = [], [text for text in (self.config.system_prompt, params.instructions) if text]
        for item in messages:
            item = item if isinstance(item, dict) else item.model_dump()
            content = item.get("content")
            text = content if isinstance(content, str) else "\n".join(p.get("text", "") for p in content or [])
            (system if item.get("role") in ("system", "developer") else prompt).append(text)
        if not prompt:
            raise ValueError("Goose needs a user message")

        remote = "/tmp/nemo-gym-goose-" + uuid4().hex
        await self._install_goose(sandbox, remote)
        files = {
            # JSON is valid YAML, and Goose reads its config file as YAML.
            f"{remote}/root/config/config.yaml": json.dumps(self._goose_config(mcp)),
            f"{remote}/prompt.txt": "\n\n".join(prompt),
            f"{remote}/system.txt": "\n\n".join(system),
        }
        with TemporaryDirectory() as directory:
            for i, (target, contents) in enumerate(files.items()):
                local = Path(directory) / str(i)
                local.write_text(contents)
                await sandbox.upload(local, target)
        # The config holds per-session MCP credentials.
        protect = await sandbox.exec(command=f"chmod -R go-rwx {quote(remote)}", timeout_s=60)
        if protect.return_code != 0 or protect.error_type:
            raise RuntimeError("Unable to protect Goose session configuration")

        env = {
            "GOOSE_PATH_ROOT": f"{remote}/root",
            "GOOSE_PROVIDER": self.config.provider,
            "GOOSE_MODEL": self.config.model,
            "GOOSE_MODE": self.config.goose_mode,
            "GOOSE_DISABLE_KEYRING": "1",
            "GOOSE_TELEMETRY_OFF": "1",
            "OPENAI_BASE_URL": self._model_base_url(body),
            "OPENAI_API_KEY": self.config.api_key,
        }
        if self.config.subagent_model:
            env["GOOSE_SUBAGENT_MODEL"] = self.config.subagent_model
        if params.temperature is not None:
            env["GOOSE_TEMPERATURE"] = str(params.temperature)
        env |= self.config.env
        system_arg = f'--system "$(cat {quote(remote)}/system.txt)"' if system else ""
        # Sessions are archived even when Goose fails, so partial transcripts stay inspectable.
        command = (
            f"{quote(remote)}/goose run --name gym -i {quote(remote)}/prompt.txt --max-turns {self.config.max_turns} "
            f"--output-format stream-json {system_arg} > {quote(remote)}/stdout.jsonl 2> {quote(remote)}/stderr.log; "
            f"code=$?; tar czf {quote(remote)}/sessions.tgz -C {quote(remote)}/root/data sessions; exit $code"
        )
        result, error_type = None, None
        try:
            result = await sandbox.exec(command=command, env=env, timeout_s=self.config.timeout)
        except Exception as exc:
            error_type = type(exc).__name__
            LOG.exception("Goose sandbox execution failed")
        # The Docker provider returns a timeout as a result (error_type="timeout") instead of raising it.
        error_type = error_type or getattr(result, "error_type", None)
        if error_type and "timeout" in error_type.lower():
            # The run was cut off before archiving its sessions.
            await sandbox.exec(
                command=f"tar czf {quote(remote)}/sessions.tgz -C {quote(remote)}/root/data sessions", timeout_s=300
            )
        # Export failures propagate, so retrying such a request cannot become a scored zero.
        for name in ("stdout.jsonl", "stderr.log", "sessions.tgz"):
            await sandbox.download(f"{remote}/{name}", root / name)
        with tarfile.open(root / "sessions.tgz") as archive:
            archive.extractall(root, filter="data")
        sessions = await to_thread(read_goose_sessions, root / "sessions" / "sessions.db")
        return_code = getattr(result, "return_code", None)
        return {
            "sessions": sessions,
            "goose_failed": bool(error_type or return_code != 0),
            "goose_exit_code": return_code,
            "goose_error_type": error_type,
            "goose_results_dir": str(root),
        }

    def _response(self, body: GooseAgentRunRequest, sessions: list[dict[str, Any]]) -> NeMoGymResponse:
        roots = [s for s in sessions if not s.get("parent_session_id")]
        if len(roots) != 1:
            raise RuntimeError(f"Expected one root Goose session, found {len(roots)}")
        usage = [u for s in sessions for u in s["usage"]]
        cached = sum(u["cache_read_tokens"] or 0 for u in usage)
        input_tokens = sum(u["input_tokens"] or 0 for u in usage)
        output_tokens = sum(u["output_tokens"] or 0 for u in usage)
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=goose_messages_to_output(roots[0]["messages"]),
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
            # Child sessions are included: delegate calls are part of the rollout's cost.
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise ValueError("GooseAgent requires /run to seed tools and allocate its sandbox")

    async def run(self, request: Request, body: GooseAgentRunRequest) -> GooseAgentVerifyResponse:
        async with self._sem:
            seeded = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=request.cookies,
            )
            await raise_for_status(seeded)
            cookies = request.cookies | seeded.cookies
            seed = await get_response_json(seeded)
            mcp = await seed_mcp_servers(
                self.server_client,
                self.config.tool_servers,
                body,
                cookies,
                timeout_s=self.config.timeout,
                require_reachable=self.config.network_access != "inherit",
            )
            rollout_id = self.rollout_id_from_run(body) or uuid4().hex
            root = Path(self.config.artifacts_dir) / sha256(rollout_id.encode()).hexdigest()
            root.mkdir(parents=True, exist_ok=True)
            sandbox = await self._start_sandbox(seed)
            try:
                execution = await self._run_goose(sandbox, body, mcp, root)
                sessions = execution.pop("sessions")
                response = self._response(body, sessions)
                children = [s for s in sessions if s.get("parent_session_id")]
                execution["goose_subagents"] = len(children)
                execution["goose_max_concurrent_subagents"] = max_concurrent(
                    [
                        (min(m["created"] for m in s["messages"]), max(m["created"] for m in s["messages"]))
                        for s in children
                        if s["messages"]
                    ]
                )
                payload = body.model_dump(mode="json") | {"response": response.model_dump(mode="json")}
                (root / "generation.json").write_text(json.dumps(payload | execution))
                result = await verify_agent_response(
                    self.server_client,
                    self.config.resources_server,
                    body,
                    response,
                    cookies,
                    force_zero_reward=execution["goose_failed"] and self.config.execution_failure_reward_zero,
                )
                return GooseAgentVerifyResponse.model_validate(result | execution)
            finally:
                # A resource-owned sandbox is graded in place, so it stays up until verification returns.
                # Goose keeps running in the sandbox after a cancelled request. Only stopping the sandbox ends it.
                with CancelScope(shield=True):
                    try:
                        await sandbox.stop()
                    except Exception:
                        LOG.exception("Failed to stop Goose sandbox")


if __name__ == "__main__":
    GooseAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = GooseAgent.run_webserver()  # noqa: F401
