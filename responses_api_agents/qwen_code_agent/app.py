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

"""Run the Qwen Code CLI headless in a provider-managed sandbox; parse and grade on the host."""

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

from fastapi import Request
from pydantic import ConfigDict, Field, model_validator

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
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.responses_converter import ResponsesConverter
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

# Tools that start other agents; removing them is the solo control.
SUBAGENT_TOOLS = ["agent", "list_agents"]
API_KEY_ENV = "NEMO_GYM_MODEL_API_KEY"  # pragma: allowlist secret
SUBAGENT_PROMPT = Path(__file__).with_name("general-purpose-agent.md")
_CONVERTER = ResponsesConverter(return_token_id_information=False)


class QwenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    """Qwen Code CLI settings plus sandbox allocation and host-side tool services."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Route id Qwen Code sends as the model name. Gym model servers substitute their own model.
    model: str
    concurrency: int = 32
    timeout: int = Field(default=3600, description="Seconds for the whole Qwen Code run, children included.")

    subagents: bool = Field(default=True, description="False removes the agent tools (solo control).")
    # A second route for the built-in general-purpose subagent. Forks and teammates keep the root model.
    subagent_model: str | None = None
    subagent_model_server: ModelServerRef | None = Field(
        default=None, description="Model server for subagent_model; defaults to model_server."
    )
    max_subagent_depth: int | None = Field(default=None, description="Nesting levels; Qwen Code defaults to 5.")
    max_parallel_agents: int | None = Field(
        default=None, description="Concurrent background subagents; Qwen Code defaults to 10."
    )
    agent_team: bool = Field(default=False, description="Enable the experimental Agent Team tools.")
    max_session_turns: int | None = None
    max_tool_calls: int | None = None
    approval_mode: Literal["yolo", "auto-edit", "auto", "default", "plan"] = "yolo"
    system_prompt: str | None = None
    exclude_tools: list[str] = Field(default_factory=list)
    # Deep-merged over the generated ~/.qwen/settings.json.
    settings: dict[str, Any] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)

    qwen_version: str = "0.25.1"
    runtime_archive: str | None = Field(
        default=None,
        description="Host tar.gz from build_runtime.sh, uploaded into each sandbox; null uses `qwen` on the image PATH.",
    )
    sandbox_provider: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    network_access: Literal["inherit", "model_only", "model_and_tools"] = "inherit"
    tool_servers: list[ResourcesServerRef] = Field(default_factory=list)
    execution_failure_reward_zero: bool = False
    artifacts_dir: str = "results/qwen_code"

    @model_validator(mode="after")
    def check_subagent_settings(self) -> "QwenCodeAgentConfig":
        if not self.subagents and (self.subagent_model or self.agent_team):
            raise ValueError("subagent_model and agent_team require subagents: true")
        if self.subagent_model_server and not self.subagent_model:
            raise ValueError("subagent_model_server requires subagent_model")
        if self.subagent_model == self.model:
            raise ValueError("subagent_model must differ from model; leave it null to inherit the root model")
        return self


class QwenCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class QwenCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    qwen_failed: bool = False
    qwen_exit_code: int | None = None
    qwen_error_type: str | None = None
    qwen_results_dir: str = ""
    qwen_subagents: int = 0
    qwen_subagents_background: int = 0
    qwen_subagent_max_depth: int = 0
    qwen_subagent_input_tokens: int = 0
    qwen_subagent_output_tokens: int = 0


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "") for p in content)
    return "" if content is None else json.dumps(content)


def parse_stream(stdout: str) -> tuple[list[Any], dict[str, Any] | None]:
    """Convert root-session ``-o stream-json`` events into output items; also return the result event.

    Subagent events carry a ``parent_tool_use_id`` and are left to the exported session transcripts.
    """
    items: list[Any] = []
    result = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if not isinstance(event, dict) or event.get("parent_tool_use_id"):
            continue
        if event.get("type") == "result":
            result = event
            continue
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else None
            if event["type"] == "assistant" and kind in ("thinking", "text"):
                # Gym model servers inline reasoning as <think> tags; split it out of the graded text.
                text = f"<think>{block['thinking']}</think>" if kind == "thinking" else block.get("text", "")
                if text.strip():
                    items += _CONVERTER.postprocess_assistant_message_dict({"role": "assistant", "content": text})
            elif event["type"] == "assistant" and kind == "tool_use":
                items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=json.dumps(block.get("input") or {}),
                        call_id=block.get("id", ""),
                        name=block.get("name", ""),
                        id=block.get("id"),
                        status="completed",
                    )
                )
            elif event["type"] == "user" and kind == "tool_result":
                items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=block.get("tool_use_id", ""), output=_text(block.get("content")), status="completed"
                    )
                )
    return items, result


def summarize_subagents(sessions_archive: Path) -> dict[str, int]:
    """Count child sessions from the exported ``projects/*/subagents/<session>/*.meta.json`` files."""
    keys = [
        "subagents",
        "subagents_background",
        "subagent_max_depth",
        "subagent_input_tokens",
        "subagent_output_tokens",
    ]
    stats = {f"qwen_{key}": 0 for key in keys}
    with tarfile.open(sessions_archive) as archive:
        for member in archive.getmembers():
            # Child transcripts are subagents/<session>/agent-<id>.jsonl plus agent-<id>.meta.json.
            name = Path(member.name).name
            if "/subagents/" not in member.name or not name.startswith("agent-") or not member.isfile():
                continue
            data = archive.extractfile(member).read().decode(errors="replace")
            if name.endswith(".meta.json"):
                meta = json.loads(data)
                stats["qwen_subagents"] += 1
                stats["qwen_subagents_background"] += bool(meta.get("isBackgrounded"))
                # depth is 0 for a top-level subagent; report levels.
                stats["qwen_subagent_max_depth"] = max(stats["qwen_subagent_max_depth"], meta.get("depth", 0) + 1)
            elif name.endswith(".jsonl"):
                for line in data.splitlines():
                    usage = json.loads(line).get("usageMetadata") or {}
                    stats["qwen_subagent_input_tokens"] += usage.get("promptTokenCount", 0)
                    stats["qwen_subagent_output_tokens"] += usage.get("candidatesTokenCount", 0)
    return stats


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class QwenCodeAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: QwenCodeAgentConfig
    sem: Semaphore | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        # Qwen Code runs only inside the sandbox; nothing is installed on the Gym host.
        super().model_post_init(__context)
        self.sem = Semaphore(self.config.concurrency)

    def _model_url(self, server: ModelServerRef, body: QwenCodeAgentRunRequest) -> str:
        url = sandbox_server_url(server.name, require_reachable=self.config.network_access != "inherit")
        return self.base_url_for_run(url, body).rstrip("/") + "/v1"

    def _build_settings(self, body: QwenCodeAgentRunRequest, mcp: dict[str, dict[str, Any]]) -> dict[str, Any]:
        routes = [{"id": self.config.model, "baseUrl": self._model_url(self.config.model_server, body)}]
        if self.config.subagent_model:
            server = self.config.subagent_model_server or self.config.model_server
            routes.append({"id": self.config.subagent_model, "baseUrl": self._model_url(server, body)})
        settings: dict[str, Any] = {
            "modelProviders": {"openai": [route | {"envKey": API_KEY_ENV} for route in routes]},
            "security": {"auth": {"selectedType": "openai"}},
            "model": {"name": self.config.model},
            "general": {"enableAutoUpdate": False},
            "privacy": {"usageStatisticsEnabled": False},
            # Background memory extraction makes extra model calls outside the task.
            "memory": {"enableManagedAutoMemory": False},
            "mcpServers": {
                name: {
                    "httpUrl": entry["url"],
                    "headers": entry["headers"],
                    "timeout": entry["timeout"],
                    "trust": True,
                }
                for name, entry in mcp.items()
            },
        }
        if self.config.max_parallel_agents:
            settings["agents"] = {"maxParallelAgents": self.config.max_parallel_agents}
        if self.config.agent_team:
            settings["experimental"] = {"agentTeam": True}
        return _deep_merge(settings, deepcopy(self.config.settings))

    def _build_command(self, system_prompt: str | None) -> list[str]:
        cmd = ["qwen", "--approval-mode", self.config.approval_mode, "-o", "stream-json"]
        for flag, value in [
            ("--max-session-turns", self.config.max_session_turns),
            ("--max-tool-calls", self.config.max_tool_calls),
            ("--max-subagent-depth", self.config.max_subagent_depth),
        ]:
            if value is not None:
                cmd += [flag, str(value)]
        excluded = self.config.exclude_tools + ([] if self.config.subagents else SUBAGENT_TOOLS)
        for tool in excluded:
            cmd += ["--exclude-tools", tool]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        return cmd + self.config.extra_args

    @staticmethod
    def _instruction(body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str | None]:
        """Return (prompt, system prompt) from a single-turn request."""
        if isinstance(body.input, str):
            return body.input, body.instructions
        system, user = [body.instructions] if body.instructions else [], []
        for item in body.input:
            role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
            content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
            if role in ("system", "developer"):
                system.append(_text(content))
            elif role == "user":
                user.append(_text(content))
        return "\n\n".join(user), "\n\n".join(system) or None

    async def _start_sandbox(self, seed: dict[str, Any]) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config))
        descriptor = seed.get("sandbox_descriptor")
        if not descriptor and seed.get("sandbox_handle"):
            descriptor = {"sandbox_id": seed["sandbox_handle"]}
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
            servers = [self.config.model_server, self.config.subagent_model_server]
            if self.config.network_access == "model_and_tools":
                if not self.config.tool_servers:
                    raise ValueError("model_and_tools requires tool_servers")
                servers += self.config.tool_servers
            urls = [sandbox_server_url(s.name, require_reachable=True) for s in servers if s]
            options.setdefault("provider_options", {})["network_policy"] = restricted_network_policy(
                provider.name, urls
            )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(SandboxSpec(resources=resources, env=env, metadata=metadata, **options))
        return sandbox

    async def _setup(self, sandbox: AsyncSandbox, remote: str, files: dict[str, str]) -> None:
        """Stage session files and the runtime; check the installed Qwen Code version."""
        with TemporaryDirectory() as directory:
            for i, (target, contents) in enumerate(files.items()):
                local = Path(directory) / str(i)
                local.write_text(contents)
                await sandbox.upload(local, target)
            steps = [f"chmod -R go-rwx {quote(remote)}"]
            if self.config.runtime_archive:
                await sandbox.upload(self.config.runtime_archive, remote + "/runtime.tar.gz")
                steps.append(f"tar xzf {quote(remote)}/runtime.tar.gz -C {quote(remote)}/runtime")
            steps.append(f"PATH={quote(remote)}/runtime/bin:$PATH qwen --version")
        result = await sandbox.exec(
            command=f"mkdir -p {quote(remote)}/runtime && " + " && ".join(steps), timeout_s=600
        )
        # The version is the last line; archive tools may warn before it.
        version = (result.stdout or "").strip().rpartition("\n")[2]
        # Setup failures are infrastructure errors, not zero-reward generations.
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"Qwen Code setup failed: {(result.stderr or result.stdout or '')[-500:]}")
        if version != self.config.qwen_version:
            raise RuntimeError(
                f"Installed Qwen Code {version!r} does not match qwen_version {self.config.qwen_version}"
            )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise ValueError("QwenCodeAgent requires /run to seed tools and allocate its sandbox")

    async def _run_qwen(
        self, sandbox: AsyncSandbox, body: QwenCodeAgentRunRequest, mcp: dict[str, dict[str, Any]], root: Path
    ) -> tuple[NeMoGymResponse, dict[str, Any]]:
        params = body.responses_create_params
        prompt, input_system = self._instruction(params)
        system_prompt = "\n\n".join(p for p in [self.config.system_prompt, input_system] if p) or None
        remote = "/tmp/nemo-gym-qwen-" + uuid4().hex
        home = remote + "/home"
        files = {
            home + "/.qwen/settings.json": json.dumps(self._build_settings(body, mcp)),
            remote + "/prompt.txt": prompt,
        }
        if self.config.subagent_model:
            files[home + "/.qwen/agents/general-purpose.md"] = SUBAGENT_PROMPT.read_text().replace(
                "{model}", self.config.subagent_model
            )
        await self._setup(sandbox, remote, files)

        env = {
            "HOME": home,
            API_KEY_ENV: "EMPTY",
            "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1",
        }
        command = (
            f"PATH={quote(remote)}/runtime/bin:$PATH {join(self._build_command(system_prompt))}"
            f" < {quote(remote)}/prompt.txt > {quote(remote)}/stdout.jsonl 2> {quote(remote)}/stderr.log"
        )
        result, error_type = None, None
        try:
            result = await sandbox.exec(command=command, env=env, timeout_s=self.config.timeout)
            error_type = result.error_type
        except Exception as exc:
            error_type = type(exc).__name__
            LOG.exception("Qwen Code sandbox execution failed")
        exit_code = getattr(result, "return_code", None)

        # Export failures propagate; a retried request must not become a scored zero.
        export = await sandbox.exec(
            command=f"tar czf {quote(remote)}/sessions.tar.gz -C {quote(home)}/.qwen projects", timeout_s=300
        )
        if export.return_code != 0:
            raise RuntimeError(f"Qwen Code session export failed: {(export.stderr or '')[-500:]}")
        for name in ["stdout.jsonl", "stderr.log", "sessions.tar.gz"]:
            await sandbox.download(f"{remote}/{name}", root / name)

        output, final = parse_stream((root / "stdout.jsonl").read_text(errors="replace"))
        usage = (final or {}).get("usage") or {}
        input_tokens, output_tokens = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=output,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
            # Qwen Code's result usage includes subagent calls.
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cached_tokens=int(usage.get("cache_read_input_tokens", 0))
                ),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )
        execution = {
            "qwen_failed": bool(error_type or exit_code != 0 or not final or final.get("is_error")),
            "qwen_exit_code": exit_code,
            "qwen_error_type": error_type,
            "qwen_results_dir": str(root),
        } | summarize_subagents(root / "sessions.tar.gz")
        return response, execution

    async def run(self, request: Request, body: QwenCodeAgentRunRequest) -> QwenCodeAgentVerifyResponse:
        async with self.sem:
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
            # A resource-owned sandbox must stay alive through verification.
            try:
                response, execution = await self._run_qwen(sandbox, body, mcp, root)
                payload = body.model_dump(mode="json") | {"response": response.model_dump(mode="json")}
                (root / "generation.json").write_text(json.dumps(payload | execution))
                result = await verify_agent_response(
                    self.server_client,
                    self.config.resources_server,
                    body,
                    response,
                    cookies,
                    force_zero_reward=execution["qwen_failed"] and self.config.execution_failure_reward_zero,
                )
                return QwenCodeAgentVerifyResponse.model_validate(result | execution)
            finally:
                try:
                    await sandbox.stop()
                except Exception:
                    LOG.exception("Failed to stop Qwen Code sandbox")


if __name__ == "__main__":
    QwenCodeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = QwenCodeAgent.run_webserver()
