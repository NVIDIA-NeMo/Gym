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

"""Run the Kimi Code CLI (`kimi -p`) in a provider-managed sandbox, with its Agent and AgentSwarm subagents."""

import json
import logging
import tarfile
from asyncio import Semaphore
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from shlex import join, quote
from tempfile import TemporaryDirectory
from time import time
from typing import Any, Literal
from uuid import uuid4

from anyio import CancelScope
from fastapi import Request
from pydantic import ConfigDict, Field, FilePath

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, SandboxObservation
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
INSTALLER_URL = "https://code.kimi.com/kimi-code/install.sh"


class KimiCodeSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    """Kimi Code CLI settings, subagent controls, and sandbox allocation."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Children use this server by default; the root may still pick "primary" per spawn unless forced.
    subagent_model_server: ModelServerRef | None = None
    subagent_model_force: bool = False
    # False disables the Agent and AgentSwarm tools (solo control).
    subagents: bool = True
    swarm_max_concurrency: int | None = Field(default=None, gt=0)
    max_steps_per_turn: int | None = Field(default=None, ge=0)
    # Bounds the extra root turns that background subagent completions trigger in print mode.
    print_max_turns: int | None = Field(default=None, gt=0)
    thinking_effort: str | None = None
    max_context_size: int = 262144
    disabled_tools: list[str] = Field(default_factory=lambda: ["CronCreate", "CronList", "CronDelete"])
    permission_rules: list[dict[str, str]] = Field(default_factory=list)
    # Deep-merged into the generated config.toml last.
    kimi_config: dict[str, Any] = Field(default_factory=dict)

    kimi_version: str = "2.1.1"
    # Host-local Linux release binary uploaded into the box; otherwise the pinned official installer runs in the box.
    kimi_binary_path: FilePath | None = None

    concurrency: int = 32
    timeout: float = 7200
    sandbox_provider: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    network_access: Literal["inherit", "model_only", "model_and_tools"] = "inherit"
    tool_servers: list[ResourcesServerRef] = Field(default_factory=list)
    execution_failure_reward_zero: bool = False
    artifacts_dir: str = "results/kimi_code_sandboxed"


class KimiCodeSandboxedAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class KimiCodeSandboxedAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    kimi_failed: bool = False
    kimi_exit_code: int | None = None
    kimi_error_type: str | None = None
    kimi_results_dir: str = ""
    kimi_num_subagents: int = 0
    kimi_max_concurrent_subagents: int = 0
    kimi_subagents: list[dict[str, Any]] = Field(default_factory=list)
    ng_agent_observations: AgentObservationBundle | None = Field(default=None, exclude_if=lambda value: value is None)


def _toml_str(value: Any) -> str:
    # JSON with raw Unicode is a valid TOML value. TOML rejects surrogate escapes like \ud83d and a literal DEL.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def to_toml(data: dict[str, Any], path: tuple[str, ...] = (), header: str | None = None) -> str:
    """Serialize nested dicts, arrays of tables, and scalars; JSON strings and arrays are valid TOML values."""
    lines = [header or f"[{'.'.join(path)}]"] if path else []
    tables = []
    for key, value in data.items():
        name = _toml_str(str(key))
        if value is None:
            continue
        if isinstance(value, dict):
            tables.append(to_toml(value, (*path, name)))
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            array = ".".join((*path, name))
            tables.extend(to_toml(item, (*path, name), header=f"[[{array}]]") for item in value)
        else:
            lines.append(f"{name} = {_toml_str(value)}")
    return "\n".join(["\n".join(lines), *tables]).strip() + "\n"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _json_or_none(line: str) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def parse_stream_json(stdout: str) -> list[Any]:
    """Convert the root agent's `--output-format stream-json` chat messages to Responses output items.

    Gym model servers return reasoning inside `<think>` tags in the content, which the converter splits off again.
    """
    converter = ResponsesConverter(return_token_id_information=False)
    output = []
    for message in map(_json_or_none, stdout.splitlines()):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            output.extend(converter.postprocess_assistant_message_dict(message))
        elif message.get("role") == "tool":
            content = message.get("content")
            output.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=message.get("tool_call_id", ""),
                    output=content if isinstance(content, str) else json.dumps(content),
                )
            )
    return output


def parse_session(session_dir: Path) -> tuple[NeMoGymResponseUsage | None, list[dict[str, Any]]]:
    """Sum model usage over the root and every subagent, and list the subagents from the wire logs."""
    usages, subagents = [], {}
    for wire in sorted(session_dir.glob("agents/*/wire.jsonl")):
        for event in map(_json_or_none, wire.read_text(errors="replace").splitlines()):
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "usage.record":
                usage = event.get("usage") or {}
                cached = usage.get("inputCacheRead", 0)
                input_tokens = usage.get("inputOther", 0) + cached + usage.get("inputCacheCreation", 0)
                usages.append(
                    NeMoGymResponseUsage(
                        input_tokens=input_tokens,
                        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached),
                        output_tokens=usage.get("output", 0),
                        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                        total_tokens=input_tokens + usage.get("output", 0),
                    )
                )
            elif kind == "subagent.spawned":
                subagents[event["subagentId"]] = {
                    "id": event["subagentId"],
                    "parent": event.get("parentAgentId"),
                    "tool_call_id": event.get("parentToolCallId"),
                    "type": event.get("subagentName"),
                    "model": event.get("model"),
                    "swarm_index": event.get("swarmIndex"),
                    "background": event.get("runInBackground", False),
                    "status": "incomplete",
                }
            elif kind and kind.startswith("subagent.") and event.get("subagentId") in subagents:
                child = subagents[event["subagentId"]]
                if kind == "subagent.started":
                    child["started"] = event.get("time")
                else:
                    child["ended"] = event.get("time")
                    child["status"] = "completed" if kind == "subagent.completed" else "failed"
    usage = NeMoGymResponseUsage.sum_from_list(usages) if usages else None
    return usage, list(subagents.values())


def max_concurrency(subagents: list[dict[str, Any]]) -> int:
    edges = sorted(
        (t, step)
        for child in subagents
        if child.get("started") is not None
        for t, step in ((child["started"], 1), (child.get("ended", float("inf")), -1))
    )
    running = peak = 0
    for _, step in edges:
        running += step
        peak = max(peak, running)
    return peak


class KimiCodeSandboxedAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: KimiCodeSandboxedAgentConfig

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._sem = Semaphore(self.config.concurrency)

    def _model_base_url(self, server: ModelServerRef, body: KimiCodeSandboxedAgentRunRequest) -> str:
        reachable = self.config.network_access != "inherit"
        return self.base_url_for_run(sandbox_server_url(server.name, require_reachable=reachable), body) + "/v1"

    def build_kimi_config(self, body: KimiCodeSandboxedAgentRunRequest) -> dict[str, Any]:
        """Point Kimi at Gym's model servers; every root and child call keeps this rollout's capture prefix."""
        model = {
            "max_context_size": self.config.max_context_size,
            "capabilities": ["thinking", "tool_use"],
        }
        config: dict[str, Any] = {
            "default_model": "policy",
            "telemetry": False,
            "auto_session_title": False,
            "providers": {
                "policy": {"type": "openai", "base_url": self._model_base_url(self.config.model_server, body)},
            },
            "models": {"policy": model | {"provider": "policy", "model": self.config.model_server.name}},
            "tools": {"disabled": list(self.config.disabled_tools)},
            "loop_control": {"max_steps_per_turn": self.config.max_steps_per_turn},
            "background": {"print_max_turns": self.config.print_max_turns},
        }
        if self.config.thinking_effort:
            config["thinking"] = {"effort": self.config.thinking_effort}
        if not self.config.subagents:
            config["tools"]["disabled"] += ["Agent", "AgentSwarm"]
        if server := self.config.subagent_model_server:
            config["providers"]["subagent"] = {"type": "openai", "base_url": self._model_base_url(server, body)}
            config["models"]["subagent"] = model | {"provider": "subagent", "model": server.name}
            config["secondary_model"] = {"default_model": "subagent", "force": self.config.subagent_model_force}
        for provider in config["providers"].values():
            provider["api_key"] = "dummy"  # pragma: allowlist secret
        if self.config.permission_rules:
            config["permission"] = {"rules": self.config.permission_rules}
        return deep_merge(config, self.config.kimi_config)

    async def _start_sandbox(self, seed: dict[str, Any], body: KimiCodeSandboxedAgentRunRequest) -> AsyncSandbox:
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
        meta = getattr(body.responses_create_params, "metadata", None) or {}
        if meta.get("docker_image"):
            options["image"] = meta["docker_image"]
        resources = SandboxResources.from_mapping(options.pop("resources", {}))
        env = cpu_cap_env(resources.cpu) if options.pop("derive_cpu_env", True) else {}
        env.update(options.pop("env", {}))
        metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        metadata = metadata | options.pop("metadata", {}) | {"nemo_gym_agent": self.config.name}
        if self.config.network_access != "inherit":
            servers = [self.config.model_server, self.config.subagent_model_server]
            if self.config.network_access == "model_and_tools":
                if not self.config.tool_servers:
                    raise ValueError("model_and_tools requires tool_servers")
                servers += self.config.tool_servers
            urls = [sandbox_server_url(s.name, require_reachable=True) for s in servers if s is not None]
            options.setdefault("provider_options", {})["network_policy"] = restricted_network_policy(
                provider.name, urls
            )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(SandboxSpec(resources=resources, env=env, metadata=metadata, **options))
        return sandbox

    async def _run_kimi(
        self, sandbox: AsyncSandbox, body: KimiCodeSandboxedAgentRunRequest, mcp: dict[str, Any], root: Path
    ) -> dict[str, Any]:
        """Install, run one print-mode session, and download its native artifacts to ``root``."""
        params = body.responses_create_params
        messages = [params.input] if isinstance(params.input, str) else params.input
        prompts, instructions = [], [params.instructions] if getattr(params, "instructions", None) else []
        for item in messages:
            if isinstance(item, str):
                prompts.append(item)
                continue
            role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            text = (
                content
                if isinstance(content, str)
                else "\n".join(
                    (part.get("text") if isinstance(part, dict) else getattr(part, "text", "")) or ""
                    for part in content or []
                )
            )
            (instructions if role in ("system", "developer") else prompts).append(text)
        if not prompts:
            raise ValueError("Kimi Code needs at least one user message")

        # Outside the usual task workdirs, so the agent's workspace listing and searches do not include the runtime.
        remote = "/var/tmp/nemo-gym-kimi-" + uuid4().hex
        home = remote + "/home"
        files = {home + "/config.toml": to_toml(self.build_kimi_config(body))}
        if mcp:
            files[home + "/mcp.json"] = json.dumps({"mcpServers": mcp})
        if instructions:
            files[home + "/AGENTS.md"] = "\n\n".join(instructions)
        with TemporaryDirectory() as directory:
            for i, (target, contents) in enumerate(files.items()):
                local = Path(directory) / str(i)
                local.write_text(contents)
                await sandbox.upload(local, target)
        if self.config.kimi_binary_path is not None:
            await sandbox.upload(self.config.kimi_binary_path, remote + "/upload/kimi")
            install = f"mkdir -p {remote}/bin && install -m 0755 {remote}/upload/kimi {remote}/bin/kimi"
        else:
            install = (
                f"curl -fsSL {INSTALLER_URL} | KIMI_VERSION={quote(self.config.kimi_version)} "
                f"KIMI_INSTALL_DIR={remote} KIMI_NO_MODIFY_PATH=1 bash"
            )
        kimi = remote + "/bin/kimi"
        setup = await sandbox.exec(
            command=(
                f"chmod -R go-rwx {remote} && ({install}) > {remote}/install.log 2>&1 "
                f"|| {{ tail -c 1000 {remote}/install.log >&2; exit 1; }}; {kimi} --version"
            ),
            timeout_s=900,
        )
        # Installation failures are infrastructure errors, not zero-reward generations.
        if setup.return_code != 0 or setup.error_type or (setup.stdout or "").strip() != self.config.kimi_version:
            raise RuntimeError(f"Kimi Code {self.config.kimi_version} install failed: {(setup.stderr or '')[-500:]}")

        env = {
            "KIMI_CODE_HOME": home,
            "KIMI_DISABLE_TELEMETRY": "1",
            "KIMI_CODE_NO_AUTO_UPDATE": "1",
            "KIMI_DISABLE_CRON": "1",
            "KIMI_CODE_TRUST_WORKSPACE": "1",
        }
        if self.config.swarm_max_concurrency:
            env["KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY"] = str(self.config.swarm_max_concurrency)
        command = join([kimi, "-p", "\n\n".join(prompts), "--output-format", "stream-json"])
        result, error_type = None, None
        try:
            result = await sandbox.exec(
                command=f"{command} > {remote}/stdout.jsonl 2> {remote}/stderr.log",
                env=env,
                timeout_s=self.config.timeout,
            )
        except Exception as exc:
            error_type = type(exc).__name__
            LOG.exception("Kimi Code sandbox execution failed")
        # Export failures propagate; retrying such a request must not become a scored zero.
        archive = await sandbox.exec(
            command=f"tar czf {remote}/sessions.tgz -C {home} sessions", timeout_s=self.config.timeout
        )
        if archive.return_code != 0:
            raise RuntimeError(f"Kimi Code session export failed: {(archive.stderr or '')[-500:]}")
        for name in ("stdout.jsonl", "stderr.log", "sessions.tgz"):
            await sandbox.download(f"{remote}/{name}", root / name)
        with tarfile.open(root / "sessions.tgz") as tar:
            tar.extractall(root, filter="data")
        return {
            "return_code": getattr(result, "return_code", None),
            "error_type": error_type or getattr(result, "error_type", None),
        }

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise ValueError("KimiCodeSandboxedAgent requires /run to seed tools and allocate its sandbox")

    async def run(
        self, request: Request, body: KimiCodeSandboxedAgentRunRequest
    ) -> KimiCodeSandboxedAgentVerifyResponse:
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
            rollout_id = self.rollout_id_from_run(body)
            root = Path(self.config.artifacts_dir) / sha256((rollout_id or uuid4().hex).encode()).hexdigest()
            root.mkdir(parents=True, exist_ok=True)
            sandbox = await self._start_sandbox(seed, body)
            try:
                execution = await self._run_kimi(sandbox, body, mcp, root)
                stdout = (root / "stdout.jsonl").read_text(errors="replace")
                output = parse_stream_json(stdout)
                # Agents can start other kimi processes in the box, so pick this run's session by the id it printed.
                session_id = next(
                    (
                        m["session_id"]
                        for m in map(_json_or_none, stdout.splitlines())
                        if isinstance(m, dict) and m.get("session_id")
                    ),
                    None,
                )
                sessions = list((root / "sessions").glob(f"*/{session_id}")) if session_id else []
                usage, subagents = parse_session(sessions[0]) if sessions else (None, [])
                failed = bool(
                    execution["error_type"]
                    or execution["return_code"] != 0
                    or not any(getattr(item, "type", None) == "message" for item in output)
                )
                response = NeMoGymResponse(
                    id=f"resp_{uuid4().hex}",
                    created_at=int(time()),
                    model=self.config.model_server.name,
                    object="response",
                    output=output,
                    tool_choice="auto",
                    tools=[],
                    parallel_tool_calls=True,
                    usage=usage,
                )
                metadata = {
                    "kimi_failed": failed,
                    "kimi_exit_code": execution["return_code"],
                    "kimi_error_type": execution["error_type"],
                    "kimi_results_dir": str(root),
                    "kimi_num_subagents": len(subagents),
                    "kimi_max_concurrent_subagents": max_concurrency(subagents),
                    "kimi_subagents": subagents,
                }
                if rollout_id is not None:
                    metadata["ng_agent_observations"] = self._observations(
                        rollout_id, subagents, sandbox, execution, failed
                    ).model_dump(mode="json")
                payload = body.model_dump(mode="json") | {"response": response.model_dump(mode="json")}
                (root / "generation.json").write_text(json.dumps(payload | metadata))
                result = await verify_agent_response(
                    self.server_client,
                    self.config.resources_server,
                    body,
                    response,
                    cookies,
                    force_zero_reward=failed and self.config.execution_failure_reward_zero,
                )
                return KimiCodeSandboxedAgentVerifyResponse.model_validate(result | metadata)
            finally:
                # A resource-owned sandbox is graded in place, so it stays up until verification returns.
                # Cancellation is re-delivered at every await; an unshielded stop would leave the box generating.
                with CancelScope(shield=True):
                    try:
                        await sandbox.stop()
                    except Exception:
                        LOG.exception("Failed to stop Kimi Code sandbox")

    @staticmethod
    def _observations(
        rollout_id: str,
        subagents: list[dict[str, Any]],
        sandbox: AsyncSandbox,
        execution: dict[str, Any],
        failed: bool,
    ) -> AgentObservationBundle:
        def invocation_id(agent_id: str | None) -> str:
            return rollout_id if agent_id in (None, "main") else f"{rollout_id}/{agent_id}"

        records: list[Any] = [AgentInvocation(invocation_id=rollout_id, status="failed" if failed else "completed")]
        for child in subagents:
            duration = (
                child["ended"] - child["started"]
                if child.get("started") is not None and child.get("ended") is not None
                else None
            )
            records.append(
                AgentInvocation(
                    invocation_id=invocation_id(child["id"]),
                    parent_invocation_id=invocation_id(child["parent"]),
                    spawned_by_tool_call_id=child["tool_call_id"],
                    status=child["status"],
                    duration_ms=duration,
                )
            )
        error_type = execution["error_type"]
        handle = sandbox._handle
        records.append(
            SandboxObservation(
                role="agent",
                sandbox_id=handle.sandbox_id,
                provider=handle.provider_name,
                outcome="timeout" if "timeout" in (error_type or "").lower() else "failed" if failed else "completed",
                exit_code=execution["return_code"],
                error_type=error_type,
            )
        )
        return AgentObservationBundle(source="kimi_code", records=records)


if __name__ == "__main__":
    KimiCodeSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = KimiCodeSandboxedAgent.run_webserver()  # noqa: F401
