# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
from asyncio import Semaphore
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path, PurePosixPath
from shlex import quote
from time import time
from typing import Any, ClassVar, Optional
from uuid import uuid4

from fastapi import Body, HTTPException, Request
from nemo_fabric import Fabric, FabricConfig, RunRequest
from pydantic import ConfigDict, Field, JsonValue, PositiveInt

from nemo_gym.agents import (
    AGENT_TIMEOUT_METADATA_KEY,
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SandboxSession,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
    resolve_agent_timeout,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import SKILLS_REF_KEY_NAME, ModelServerRef, ResourcesServerRef, get_first_server_config_dict
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest, BaseVerifyResponse
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap, SandboxObservation
from responses_api_agents.nemo_fabric_agent.sandbox import FabricSandboxSession
from responses_api_agents.nemo_fabric_agent.sandbox_runtime import prepare_fabric_runtime, runtime_requirements


LOG = logging.getLogger(__name__)
_MODEL_API_KEY_ENV = "NEMO_GYM_FABRIC_MODEL_API_KEY"  # pragma: allowlist secret
# Adapter failures that are the model's own outcome, as (error code, metadata subtype): the model ran out of turns or
# its step budget, so the rollout is verified and scored.
_MODEL_STOPS = frozenset(
    {
        ("mini_swe_agent_incomplete", None),
        ("deepagents_recursion_limit_reached", None),
        ("claude_result_failed", "error_max_turns"),
    }
)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if isinstance(part, dict):
            text = part.get("text")
        else:
            text = getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _extract_request_input(body_input: Any) -> tuple[Any, Optional[str]]:
    if isinstance(body_input, str):
        return body_input, None

    messages: list[dict[str, Any]] = []
    instruction_parts: list[str] = []
    for item in body_input:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        text = _content_text(content)
        if role in {"system", "developer"}:
            if text:
                instruction_parts.append(text)
            continue
        message = item.model_dump(mode="json", exclude_none=True) if hasattr(item, "model_dump") else dict(item)
        messages.append(message)

    if len(messages) == 1 and messages[0].get("role") == "user":
        request_input: Any = _content_text(messages[0].get("content"))
    else:
        request_input = messages
    return request_input, "\n\n".join(instruction_parts) or None


def _reasoning_item(text: str) -> NeMoGymResponseReasoningItem:
    return NeMoGymResponseReasoningItem(
        id=f"rs_{uuid4().hex}", summary=[NeMoGymSummary(text=text, type="summary_text")]
    )


def _fabric_event_items(events: list[Any], response_text: str) -> list[Any]:
    items: list[Any] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "commandExecution":
            call_id = str(event.get("id") or f"call_{uuid4().hex}")
            arguments = {"command": event.get("command")}
            if event.get("cwd"):
                arguments["cwd"] = event["cwd"]
            items.append(
                NeMoGymResponseFunctionToolCall(
                    id=f"fc_{uuid4().hex}",
                    call_id=call_id,
                    name="shell",
                    arguments=json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                    status="completed",
                )
            )
            items.append(
                NeMoGymResponseFunctionCallOutput(
                    id=f"fco_{uuid4().hex}",
                    call_id=call_id,
                    output=_content_text(event.get("aggregatedOutput")),
                    status="completed",
                    type="function_call_output",
                )
            )
        elif isinstance(event.get("message"), dict) and isinstance(event["message"].get("content"), list):
            for block in event["message"]["content"]:
                if not isinstance(block, dict):
                    continue
                if block.get("name") and "input" in block:
                    call_id = str(block.get("id") or f"call_{uuid4().hex}")
                    items.append(
                        NeMoGymResponseFunctionToolCall(
                            id=f"fc_{uuid4().hex}",
                            call_id=call_id,
                            name=str(block["name"]),
                            arguments=json.dumps(block.get("input") or {}, ensure_ascii=False, sort_keys=True),
                            status="completed",
                        )
                    )
                elif block.get("tool_use_id"):
                    items.append(
                        NeMoGymResponseFunctionCallOutput(
                            id=f"fco_{uuid4().hex}",
                            call_id=str(block["tool_use_id"]),
                            output=_content_text(block.get("content")),
                            status="completed",
                            type="function_call_output",
                        )
                    )
                elif isinstance(block.get("thinking"), str):
                    if block["thinking"]:
                        items.append(_reasoning_item(block["thinking"]))
                else:
                    text = _content_text(block.get("text"))
                    if text and text.strip() != response_text.strip():
                        items.append(
                            NeMoGymResponseOutputMessage(
                                id=f"msg_{uuid4().hex}",
                                content=[NeMoGymResponseOutputText(text=text, annotations=[])],
                                role="assistant",
                                status="completed",
                                type="message",
                            )
                        )
        elif etype == "reasoning":
            summary = "\n".join(
                part if isinstance(part, str) else _content_text([part]) for part in event.get("summary") or []
            )
            if summary:
                items.append(_reasoning_item(summary))
        elif etype == "agentMessage":
            text = _content_text(event.get("text"))
            if text and text.strip() != response_text.strip():
                items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        content=[NeMoGymResponseOutputText(text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
    return items


def _fabric_output_items(output: dict[str, Any], response_text: str) -> list[Any]:
    messages = output.get("messages")
    items: list[Any] = []
    if not (isinstance(messages, list) and messages):
        events = output.get("events")
        if isinstance(events, list) and events:
            items.extend(_fabric_event_items(events, response_text))
    if isinstance(messages, list) and messages:
        pending: list[str] = []
        for message in messages[:-1]:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or message.get("type")
            if role in {"human", "user", "system", "developer"}:
                continue
            if role == "tool":
                call_id = pending.pop(0) if pending else str(message.get("tool_call_id") or message.get("id") or "")
                items.append(
                    NeMoGymResponseFunctionCallOutput(
                        id=f"fco_{uuid4().hex}",
                        call_id=call_id,
                        output=_content_text(message.get("content")),
                        status="completed",
                        type="function_call_output",
                    )
                )
                continue
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                items.append(_reasoning_item(reasoning))
            text = _content_text(message.get("content"))
            if text:
                items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        content=[NeMoGymResponseOutputText(text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                call_id = str(tool_call.get("id") or tool_call.get("call_id") or f"call_{uuid4().hex}")
                pending.append(call_id)
                arguments = tool_call.get("args", tool_call.get("arguments", function.get("arguments")))
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
                items.append(
                    NeMoGymResponseFunctionToolCall(
                        id=f"fc_{uuid4().hex}",
                        call_id=call_id,
                        name=str(tool_call.get("name") or function.get("name") or ""),
                        arguments=arguments,
                        status="completed",
                    )
                )
    items.append(
        NeMoGymResponseOutputMessage(
            id=f"msg_{uuid4().hex}",
            content=[NeMoGymResponseOutputText(text=response_text, annotations=[])],
            role="assistant",
            status="completed",
            type="message",
        )
    )
    return items


def _atif_trajectories(output: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the ATIF trajectories that NeMo Relay wrote for this invocation."""
    trajectories = []
    for artifact in output.get("relay_artifacts") or []:
        if not isinstance(artifact, dict) or artifact.get("kind") != "atif" or not artifact.get("path"):
            continue
        try:
            trajectory = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            LOG.warning("Could not read Relay ATIF trajectory %s (%s)", artifact["path"], type(error).__name__)
            continue
        if isinstance(trajectory, dict):
            trajectories.append(trajectory)
        else:
            LOG.warning("Relay ATIF trajectory %s is not a JSON object", artifact["path"])
    # Relay names the files by session id, so order the trajectories by their first step's timestamp.
    return sorted(trajectories, key=_first_step_timestamp)


def _first_step_timestamp(trajectory: dict[str, Any]) -> str:
    steps = trajectory.get("steps")
    return str(_mapping(steps[0] if isinstance(steps, list) and steps else None).get("timestamp"))


def _emitted_arguments(value: Any, call_id: str) -> Optional[str]:
    """The arguments string of tool call call_id in a raw chat or Responses API model response, if present."""
    if isinstance(value, list):
        return next((found for item in value if (found := _emitted_arguments(item, call_id)) is not None), None)
    if not isinstance(value, dict):
        return None
    arguments = _mapping(value.get("function")).get("arguments", value.get("arguments"))
    if call_id in (value.get("id"), value.get("call_id")) and isinstance(arguments, str):
        return arguments
    return _emitted_arguments(list(value.values()), call_id)


def _atif_trajectory_items(trajectory: dict[str, Any], response_text: str, items: list[Any]) -> None:
    """Append the agent-step items of one ATIF trajectory, inlining embedded subagent trajectories at their call."""
    subagents = {
        str(sub.get("trajectory_id")): sub
        for sub in trajectory.get("subagent_trajectories") or []
        if isinstance(sub, dict)
    }
    for step in trajectory.get("steps") or []:
        # Copied-context steps repeat earlier steps (for example after compaction), so they are not new output.
        if not isinstance(step, dict) or step.get("source") != "agent" or step.get("is_copied_context"):
            continue
        # A step without LLM calls only wraps a subagent (for example the Deep Agents orchestrator scope).
        # The policy did not emit it, so only its subagent steps become items.
        wrapper = step.get("llm_call_count") == 0
        if not wrapper:
            reasoning = step.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                items.append(_reasoning_item(reasoning))
            text = _content_text(step.get("message"))
            if text and text.strip() != response_text.strip():
                items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        content=[NeMoGymResponseOutputText(text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for tool_call in step.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("tool_call_id") or f"call_{uuid4().hex}")
                # Relay records arguments that are not a JSON object as {"raw": text} or {"value": json}, so take them
                # from its copy of the model's response when it has one.
                arguments = _emitted_arguments(_mapping(step.get("extra")).get("llm_response"), call_id)
                if arguments is None:
                    arguments = tool_call.get("arguments")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
                items.append(
                    NeMoGymResponseFunctionToolCall(
                        id=f"fc_{uuid4().hex}",
                        call_id=call_id,
                        name=str(tool_call.get("function_name") or ""),
                        arguments=arguments,
                        status="completed",
                    )
                )
        for result in _mapping(step.get("observation")).get("results") or []:
            if not isinstance(result, dict):
                continue
            for ref in result.get("subagent_trajectory_ref") or []:
                subagent = subagents.get(str(_mapping(ref).get("trajectory_id")))
                if subagent is not None:
                    _atif_trajectory_items(subagent, response_text, items)
            if wrapper:
                continue
            content = result.get("content")
            extra = _mapping(result.get("extra"))
            if content is None and "tool_result" in extra:
                # Relay stores non-text tool returns here because ATIF content is text or image parts only.
                value = extra["tool_result"]
                if isinstance(value, dict) and "__nv_pydantic__" in value:
                    # A serialized object such as a LangChain ToolMessage. Its content is what the model saw.
                    value = _mapping(value.get("data")).get("content", value)
                    if isinstance(value, list):
                        value = _content_text(value)
                content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            items.append(
                NeMoGymResponseFunctionCallOutput(
                    id=f"fco_{uuid4().hex}",
                    call_id=str(result.get("source_call_id") or ""),
                    output=_content_text(content),
                    status="completed",
                    type="function_call_output",
                )
            )


def _atif_output_items(trajectories: list[dict[str, Any]], response_text: str) -> list[Any]:
    """Response items from the agent steps of Relay ATIF trajectories, or [] when there are none."""
    items: list[Any] = []
    for trajectory in trajectories:
        _atif_trajectory_items(trajectory, response_text, items)
    if not items:
        return []
    items.append(
        NeMoGymResponseOutputMessage(
            id=f"msg_{uuid4().hex}",
            content=[NeMoGymResponseOutputText(text=response_text, annotations=[])],
            role="assistant",
            status="completed",
            type="message",
        )
    )
    return items


def _record_counts(items: list[Any]) -> tuple[int, int, int]:
    """(non-empty tool outputs, all tool outputs, reasoning items). An empty output is still a completed call."""
    outputs = [item.output for item in items if getattr(item, "type", None) == "function_call_output"]
    reasoning = sum(1 for item in items if getattr(item, "type", None) == "reasoning")
    return sum(1 for output in outputs if output), len(outputs), reasoning


def _skill_paths(skills_root: Optional[str]) -> list[str]:
    if not skills_root:
        return []
    root = Path(skills_root).resolve()
    if not root.is_dir():
        raise ValueError(f"skills_ref path is not a directory: {root}")
    if (root / "SKILL.md").is_file():
        return [str(root)]
    paths = [str(path) for path in sorted(root.iterdir()) if path.is_dir() and (path / "SKILL.md").is_file()]
    if not paths:
        raise ValueError(f"skills_ref path contains no SKILL.md directories: {root}")
    return paths


def _mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_mapping"):
        value = value.to_mapping()
    return dict(value) if isinstance(value, dict) else {}


def _usage_value(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _normalized_usage(result: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    output_usage = _mapping(output.get("usage"))
    # Fabric's normalized usage wins. The adapter's own usage fills fields it lacks, such as Codex reasoning tokens.
    usage = _deep_merge(_mapping(output_usage.get("total")) or output_usage, _mapping(result.get("usage")))
    details = _deep_merge(_mapping(usage.get("extensions")), _mapping(usage.get("metadata")))
    return _deep_merge(details, usage)


def _has_token_usage(usage: dict[str, Any]) -> bool:
    """API-call counts alone are not evidence that zero tokens were consumed."""

    def known(keys: tuple[str, ...]) -> bool:
        return any(
            isinstance(usage.get(key), int) and not isinstance(usage[key], bool) and usage[key] >= 0 for key in keys
        )

    return known(("input_tokens", "prompt_tokens", "inputTokens")) and known(
        ("output_tokens", "completion_tokens", "outputTokens")
    )


def _turns_used(output: dict[str, Any]) -> int:
    usage = _mapping(output.get("usage"))
    for value in (
        output.get("turns_used"),
        output.get("num_turns"),
        output.get("api_calls"),
        usage.get("api_calls"),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    messages = output.get("messages")
    if isinstance(messages, list):
        turns = sum(
            1 for message in messages if isinstance(message, dict) and message.get("role") in {"ai", "assistant"}
        )
        if turns:
            return turns
    return 1


class NeMoFabricAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = None
    model_server: ModelServerRef
    adapter_id: str
    model_provider: str = "openai-compatible"
    model: str
    model_api_key: str = "local"  # pragma: allowlist secret
    concurrency: PositiveInt = 32
    timeout: PositiveInt = 600
    system_prompt: Optional[str] = None
    cwd: Optional[str] = None
    harness_settings: dict[str, Any] = Field(default_factory=dict)
    fabric_config: dict[str, Any] = Field(default_factory=dict)
    # Record each run with NeMo Relay. The response is then built from Relay's ATIF trajectory unless the adapter's own
    # output is fuller, and fabric_result carries the trajectory as relay_atif.
    relay: bool = False
    # Adapters whose model schema has no base_url (Factory Droid reads custom endpoints from its own settings)
    # reject a config that sets it. False leaves models.default.base_url out.
    send_model_base_url: bool = True
    # Optional host-only adapter environment, installed on first use. Sessions install inside their sandbox.
    adapter_project: str | None = None
    # Agent sessions keep process-local state, so they require one worker. The host /run path accepts more.
    num_workers: PositiveInt = 1
    # Provider block for a sandbox this agent creates when Resources lends none. Null requires sandbox_access.
    sandbox_provider: str | None = None
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    # One budget for starting an owned sandbox and installing the Fabric runtime during seed.
    sandbox_install_timeout_seconds: PositiveInt = 600


class NeMoFabricAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NeMoFabricAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    fabric_result: dict[str, Any] = Field(default_factory=dict)


class NeMoFabricAgent(SimpleResponsesAPIAgent):
    config: NeMoFabricAgentConfig
    calls_model_from_sandbox: ClassVar[bool] = True
    sem: Semaphore | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _relay_supported: Optional[bool] = None
    _adapter_python: str | None = None
    _adapter_setup: asyncio.Task[str] | None = None

    async def _ensure_host_adapter(self) -> None:
        """Share first-use installation; surface failures and allow a subsequent call to retry."""
        if not self.config.adapter_project or self._adapter_python:
            return
        if self._adapter_setup is None or self._adapter_setup.done():
            self._adapter_setup = asyncio.create_task(asyncio.to_thread(self._install_host_adapter))
        self._adapter_python = await asyncio.shield(self._adapter_setup)

    def _install_host_adapter(self) -> str:
        project = Path(__file__).parent / self.config.adapter_project
        python = str(project / ".venv" / "bin" / "python")
        for command in (
            ["uv", "venv", "--python", "3.14", "--allow-existing", ".venv"],
            ["uv", "pip", "install", "--python", python, "-r", "pyproject.toml"],
        ):
            subprocess.run(command, cwd=project, check=True, timeout=self.config.sandbox_install_timeout_seconds)
        return python

    async def _run_host_adapter(self, config: FabricConfig, *, workspace: str, request: RunRequest) -> dict[str, Any]:
        """Use a separate process for a separate interpreter, without changing other host agents' environment."""
        with tempfile.TemporaryDirectory(prefix="nemo_gym_fabric_host_") as directory:
            path = Path(directory)
            (path / "input.json").write_text(
                json.dumps(
                    {
                        "config": config.to_mapping(),
                        "input": request.input,
                        "request_id": request.request_id,
                        "workdir": workspace,
                    }
                )
            )
            with (path / "worker.log").open("wb") as output:
                process = await asyncio.create_subprocess_exec(
                    self._adapter_python,
                    "-I",
                    "-B",
                    str(Path(__file__).with_name("sandbox_runner.py")),
                    directory,
                    start_new_session=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                try:
                    await asyncio.wait_for(process.wait(), timeout=self.config.timeout + 30)
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()
            if process.returncode:
                detail = (path / "worker.log").read_text(errors="replace")[-8000:]
                raise RuntimeError(f"Fabric adapter worker exited {process.returncode}: {detail}")
            return json.loads((path / "result.json").read_text())

    def _adapter_supports_relay(self, fabric_config: FabricConfig, base_dir: str) -> bool:
        """Whether the adapter descriptor declares the Relay telemetry provider. Planned once per agent."""
        if self._relay_supported is None:
            adapter = _mapping(Fabric().plan(fabric_config, base_dir=base_dir).to_mapping().get("adapter"))
            self._relay_supported = "relay" in _mapping(_mapping(adapter.get("telemetry")).get("providers"))
            if self._relay_supported and self.config.adapter_id == "nvidia.fabric.hermes":
                # Validate in the interpreter that will actually run Hermes.
                check = subprocess.run(
                    [
                        self._adapter_python or sys.executable,
                        "-I",
                        "-c",
                        "from nemo_fabric_adapters.hermes.telemetry import validate_hermes_relay_compatibility as v; v()",
                    ],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=30,
                )
                if check.returncode:
                    LOG.warning("Hermes Relay compatibility check failed: %s", check.stderr[-2000:])
                    self._relay_supported = False
            if not self._relay_supported:
                LOG.warning("%s does not support NeMo Relay, so it runs without Relay", self.config.adapter_id)
        return self._relay_supported

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _resources_server_base_url(self) -> str:
        config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.resources_server.name,
        )
        return self.server_client._build_server_base_url(config)

    def _rollout_mcp_servers(self, seed_response: dict[str, Any]) -> dict[str, Any]:
        metadata = seed_response.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict):
            return {}

        name = str(metadata.get("server_name") or self.config.resources_server.name)
        url_path = str(metadata.get("url_path") or "/mcp")
        server: dict[str, Any] = {
            "transport": "streamable-http",
            "url": f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}",
        }
        headers = metadata.get("headers")
        if isinstance(headers, dict) and headers:
            server["custom_headers"] = {str(key): str(value) for key, value in headers.items()}
        else:
            LOG.warning("MCP seed metadata for %r has no session headers", name)
        return {name: server}

    def _fabric_config(
        self,
        *,
        model_base_url: str,
        workspace: Path,
        system_prompt: Optional[str],
        mcp_servers: dict[str, Any],
        skills: list[str],
        relay_dir: Optional[Path] = None,
    ) -> FabricConfig:
        authored = deepcopy(self.config.fabric_config)
        generated: dict[str, Any] = {
            "metadata": {"name": self.config.name or "nemo-gym-fabric-agent"},
            "harness": {
                "adapter_id": self.config.adapter_id,
                "settings": deepcopy(self.config.harness_settings),
            },
            "runtime": {"timeout_seconds": self.config.timeout},
            "environment": {
                "provider": "local",
                "workspace": str(workspace),
                "env": {_MODEL_API_KEY_ENV: self.config.model_api_key},
            },
            "models": {
                "default": {
                    "provider": self.config.model_provider,
                    "model": self.config.model,
                    "api_key_env": _MODEL_API_KEY_ENV,
                    "base_url": (
                        model_base_url.removesuffix("/v1")
                        if self.config.model_provider == "anthropic"
                        else model_base_url
                    ),
                }
            },
        }
        if relay_dir is not None:
            # Relay only runs with an artifacts root, and only writes ATIF when observability enables it. Keep a
            # configured artifacts root, since relay_dir is removed after the run.
            atif_dir = str(relay_dir / "atif")
            if not (authored.get("runtime") or {}).get("artifacts"):
                generated["runtime"]["artifacts"] = str(relay_dir / "artifacts")
            generated["telemetry"] = {"providers": {"relay": {}}}
            generated["relay"] = {
                "project": "nemo-gym",
                "output_dir": atif_dir,
                "observability": {"version": 4, "atif": {"enabled": True, "output_directory": atif_dir}},
            }
        if not self.config.send_model_base_url:
            del generated["models"]["default"]["base_url"]
        merged = _deep_merge(authored, generated)

        authored_instruction = ((authored.get("instructions") or {}).get("system") or {}).get("content")
        instruction_parts = [part for part in (authored_instruction, system_prompt) if isinstance(part, str) and part]
        if instruction_parts:
            instructions = deepcopy(merged.get("instructions") or {})
            system = deepcopy(instructions.get("system") or {})
            system.update({"content": "\n\n".join(instruction_parts), "mode": "replace"})
            instructions["system"] = system
            merged["instructions"] = instructions

        static_servers = (authored.get("mcp") or {}).get("servers") or {}
        all_servers = _deep_merge(static_servers, mcp_servers)
        if all_servers:
            mcp = deepcopy(merged.get("mcp") or {})
            mcp["servers"] = all_servers
            merged["mcp"] = mcp

        static_skills = list((authored.get("skills") or {}).get("paths") or [])
        all_skills = list(dict.fromkeys([*static_skills, *skills]))
        if all_skills:
            skill_config = deepcopy(merged.get("skills") or {})
            skill_config["paths"] = all_skills
            merged["skills"] = skill_config

        return FabricConfig.from_mapping(merged)

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        mcp_servers: Optional[dict[str, Any]] = None,
        skills_path: Optional[str] = None,
        rollout_id: Optional[str] = None,
    ) -> tuple[NeMoGymResponse, dict[str, Any]]:
        await self._ensure_host_adapter()
        request_input, input_system = _extract_request_input(body.input)
        system_parts = [part for part in (self.config.system_prompt, body.instructions, input_system) if part]
        system_prompt = "\n\n".join(system_parts) or None
        model_base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)

        configured_workspace = Path(self.config.cwd).resolve() if self.config.cwd else None
        if configured_workspace is not None and not configured_workspace.is_dir():
            raise ValueError(f"configured cwd is not a directory: {configured_workspace}")
        workspace_context = (
            tempfile.TemporaryDirectory(prefix="nemo_gym_fabric_")
            if configured_workspace is None
            else nullcontext(str(configured_workspace))
        )
        relay_context = (
            tempfile.TemporaryDirectory(prefix="nemo_gym_relay_") if self.config.relay else nullcontext(None)
        )
        with relay_context as relay_dir, workspace_context as workspace:
            config_kwargs = dict(
                model_base_url=model_base_url,
                workspace=Path(workspace),
                system_prompt=system_prompt,
                mcp_servers=mcp_servers or {},
                skills=_skill_paths(skills_path),
            )
            if relay_dir and not await asyncio.to_thread(
                self._adapter_supports_relay, self._fabric_config(**config_kwargs), workspace
            ):
                relay_dir = None
            fabric_config = self._fabric_config(**config_kwargs, relay_dir=Path(relay_dir) if relay_dir else None)
            request = RunRequest(
                input=request_input,
                request_id=rollout_id or f"request-{uuid4().hex}",
                context={"nemo_gym_rollout_id": rollout_id} if rollout_id else {},
            )
            if self._adapter_python:
                result_mapping = await self._run_host_adapter(fabric_config, workspace=workspace, request=request)
            else:
                # Fabric otherwise starts Python adapters on the task venv, which lacks its adapter.
                os.environ.setdefault("ADAPTER_PYTHON", sys.executable)
                result = await Fabric().run(fabric_config, base_dir=workspace, request=request)
                result_mapping = result.to_mapping()
            # The trajectory files live in relay_dir, so read them before it is removed.
            trajectories = _atif_trajectories(_mapping(result_mapping.get("output"))) if relay_dir else []

        return self._response_from_result(body, result_mapping, trajectories)

    def _response_from_result(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        result_mapping: dict[str, Any],
        trajectories: list[dict[str, Any]],
        *,
        incomplete: bool = False,
    ) -> tuple[NeMoGymResponse, dict[str, Any]]:
        """Use the same output conversion for host execution and sandbox sessions."""
        error_mapping = _mapping(result_mapping.get("error"))
        model_stop = (
            error_mapping.get("code"),
            _mapping(error_mapping.get("metadata")).get("subtype"),
        ) in _MODEL_STOPS
        if result_mapping.get("status") != "succeeded" and not model_stop and not incomplete:
            error = error_mapping.get("message") or "unknown Fabric failure"
            details = {k: v for k, v in error_mapping.items() if k != "message" and v not in (None, {}, [])}
            if details:
                error = f"{error} ({json.dumps(details, default=str, sort_keys=True)})"
            raise RuntimeError(f"NeMo Fabric invocation failed: {error}")
        output = _mapping(result_mapping.get("output"))
        usage = _normalized_usage(result_mapping, output)
        input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens", "inputTokens")
        output_tokens = _usage_value(usage, "output_tokens", "completion_tokens", "outputTokens")
        cached_tokens = _usage_value(usage, "cached_input_tokens", "cache_read_input_tokens", "cachedInputTokens")
        cache_write_tokens = _usage_value(
            usage, "cache_write_input_tokens", "cache_creation_input_tokens", "cacheWriteInputTokens"
        )
        reasoning_tokens = _usage_value(usage, "reasoning_tokens", "reasoning_output_tokens", "reasoningOutputTokens")
        total_tokens = _usage_value(usage, "total_tokens", "totalTokens") or input_tokens + output_tokens
        response_text = output.get("response", output.get("output"))
        if not isinstance(response_text, str):
            response_text = (
                "" if response_text is None else json.dumps(response_text, ensure_ascii=False, sort_keys=True)
            )
        atif_items = _atif_output_items(trajectories, response_text)
        native_items = _fabric_output_items(output, response_text)
        if self.config.adapter_id == "nvidia.fabric.claude":
            # Claude's Relay results can be hook payloads; its native events preserve model-visible text and call IDs.
            native_outputs = {
                item.call_id: item.output
                for item in native_items
                if isinstance(item, NeMoGymResponseFunctionCallOutput) and item.call_id
            }
            for item in atif_items:
                if isinstance(item, NeMoGymResponseFunctionCallOutput):
                    item.output = native_outputs.get(item.call_id, item.output)
        # Some adapters' trajectories drop tool outputs or collapse the run into one step, and Relay's ATIF steps carry
        # no reasoning. Keep the fuller record.
        atif_counts, native_counts = _record_counts(atif_items), _record_counts(native_items)
        output_items = (
            atif_items
            if atif_items and all(atif >= native for atif, native in zip(atif_counts, native_counts))
            else native_items
        )
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=str(output.get("model") or self.config.model),
            object="response",
            status="incomplete" if model_stop or incomplete else "completed",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cache_write_tokens=cache_write_tokens,
                    cached_tokens=cached_tokens,
                ),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
                total_tokens=total_tokens,
            ),
        )
        if trajectories:
            result_mapping["relay_atif"] = trajectories
        return response, json.loads(json.dumps(result_mapping, default=str))

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> FabricSandboxSession:
        """Attach to the resources sandbox, or own one for tasks without a sandbox."""
        try:
            runtime_requirements(self.config.adapter_id, relay=self.config.relay)
        except ValueError as error:
            raise AgentSeedRejectedError(str(error)) from error
        if self.config.adapter_project not in {None, "hermes_env"} or (
            self.config.adapter_project and self.config.adapter_id != "nvidia.fabric.hermes"
        ):
            raise AgentSeedRejectedError("Fabric sessions use built-in sandbox runtimes, not host adapter projects")
        if self.config.num_workers != 1:
            raise AgentSeedRejectedError("Fabric sessions require num_workers=1")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError(
                "Fabric sessions currently support harness-native tools, not required remote grants"
            )
        if not self.config.send_model_base_url:
            raise AgentSeedRejectedError("Fabric sessions require the Gym model-server URL")
        if self._token_id_capture_enabled():
            raise AgentSeedRejectedError("Fabric sessions are evaluation-only; disable token_id_capture")
        authored = self.config.fabric_config
        if authored.get("mcp") or authored.get("skills"):
            raise AgentSeedRejectedError("Fabric sessions do not yet stage MCP configuration or skills")
        if set(authored.get("models") or {}) - {"default"}:
            raise AgentSeedRejectedError("Fabric sessions currently support only the default model role")
        # Session locations belong to Gym; authored paths could write into the task or bypass its sandbox.
        if (authored.get("runtime") or {}).get("artifacts") or authored.get("relay"):
            raise AgentSeedRejectedError("Fabric session artifact and Relay paths are managed by the agent server")
        environment = authored.get("environment") or {}
        if set(environment) - {"env", "env_names"}:
            raise AgentSeedRejectedError("Fabric session environment is provided by Resources")
        access = body.sandbox_access
        if access and access.user is not None:
            raise AgentSeedRejectedError("Fabric sessions do not yet support a task-specific agent user")
        owns_sandbox = access is None
        if owns_sandbox and not self.config.sandbox_provider:
            raise AgentSeedRejectedError("Fabric requires sandbox_access or a configured sandbox_provider")
        spec = None
        if owns_sandbox:
            spec_config = {"workdir": "/app", **self.config.sandbox_config}
            # Provider TTLs outlive this server, so a crash between start and close would otherwise leak the sandbox.
            spec_config.setdefault("ttl_s", self.config.sandbox_install_timeout_seconds + self.config.timeout + 600)
            try:
                spec_config["metadata"] = resolve_provider_metadata(
                    self.config.sandbox_provider, self.server_client.global_config_dict
                ) | (spec_config.get("metadata") or {})
                spec = SandboxSpec(**spec_config)
            except (TypeError, ValueError) as error:
                raise AgentSeedRejectedError(f"Invalid Fabric sandbox configuration: {error}") from error
        workdir = access.workdir if access else spec.workdir
        path = PurePosixPath(workdir or "")
        if not path.is_absolute() or ".." in path.parts or "\x00" in workdir or str(path) in {"/", "/tmp", "/opt"}:
            raise AgentSeedRejectedError("Fabric workdir must be an absolute task directory")
        if str(path).startswith(("/tmp/nemo-gym-fabric", "/opt/nemo-gym-fabric")):
            raise AgentSeedRejectedError("Fabric task workdir must be separate from runtime/session storage")
        if self.config.cwd is not None and self.config.cwd != workdir:
            raise AgentSeedRejectedError("Fabric cwd conflicts with the Resources sandbox workdir")
        provider_ref = access.connection.provider_config_ref if access else self.config.sandbox_provider
        try:
            provider_config = resolve_provider_config(provider_ref, self.server_client.global_config_dict)
        except (TypeError, ValueError) as error:
            raise AgentSeedRejectedError(f"Invalid Fabric sandbox provider: {error}") from error
        # Fabric calls the Model Server from inside the sandbox, where a loopback URL is the container itself.
        try:
            assert_model_url_reachable_from_sandbox(
                self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key),
                provider_name=next(iter(provider_config), None),
            )
        except ValueError as error:
            raise AgentSeedRejectedError(str(error)) from error
        provider = create_provider(provider_config)
        try:
            sandbox = (
                AsyncSandbox(provider)
                if owns_sandbox
                else await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
            )
        except BaseException:
            await provider.aclose()
            raise
        directory = f"/tmp/nemo-gym-fabric-sessions/{uuid4().hex}"
        state = FabricSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=directory,
                workdir=workdir,
                harness="NeMo Fabric",
                owns_sandbox=owns_sandbox,
            ),
            observations=AgentObservationBundle(
                source="nemo_fabric",
                gaps=[
                    ObservationGap(
                        code="agent_invocation_unobserved",
                        detail="Evaluation integration: Fabric invocation, compaction and model-call ownership are not normalized",
                    )
                ]
                + [
                    ObservationGap(code="optional_tool_access_ignored", detail=access.name)
                    for access in self.effective_tool_accesses(body)
                ],
            ),
        )
        try:
            async with asyncio.timeout(self.config.sandbox_install_timeout_seconds):
                if owns_sandbox:
                    await sandbox.start(spec)
                command = f"mkdir -p -- {quote(workdir)} && " if owns_sandbox else ""
                command += f"test -d {quote(workdir)} && umask 077 && mkdir -p -- {quote(directory)}"
                result = await sandbox.exec(command, cwd="/", timeout_s=30)
                if result.return_code != 0 or result.error_type:
                    raise RuntimeError(f"Fabric session setup failed: {result.stderr}")
                await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
                state.python = await prepare_fabric_runtime(
                    sandbox,
                    adapter_id=self.config.adapter_id,
                    relay=self.config.relay,
                    timeout=self.config.sandbox_install_timeout_seconds,
                )
            return state
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Release only after termination and artifact collection have completed."""
        assert isinstance(state, FabricSandboxSession)
        await state.close(self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=state.session.cleanup_confirmed,
            agent_observations=state.observations
            or AgentObservationBundle(source="nemo_fabric", gaps=[ObservationGap(code="agent_activation_unobserved")]),
        )

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, FabricSandboxSession) else None

    def _validate_session_request(self, body: NeMoGymResponseCreateParamsNonStreaming) -> None:
        for name, field in type(body).model_fields.items():
            value = getattr(body, name)
            if name in {"input", "instructions", "model"} or (name in {"stream", "background"} and value is False):
                continue
            if name == "metadata" and value is not None and set(value) <= {AGENT_TIMEOUT_METADATA_KEY}:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Fabric session execution does not support request field {name}")
        if body.model not in {None, "", self.config.model, self.config.model_server.name} or body.model_extra:
            raise HTTPException(422, "Fabric session model is selected by the agent configuration")
        if not isinstance(body.input, str):
            for item in body.input:
                item = item.model_dump(mode="json", exclude_none=True) if hasattr(item, "model_dump") else item
                if item.get("role") not in {"user", "assistant", "system", "developer"}:
                    raise HTTPException(
                        422, "Fabric sessions accept text conversations, not pre-existing tool histories"
                    )
                content = item.get("content")
                if not isinstance(content, str) and (
                    not isinstance(content, list)
                    or any(part.get("type") not in {"input_text", "output_text"} for part in content)
                ):
                    raise HTTPException(422, "Fabric sessions currently accept text-only messages")
        if not _extract_request_input(body.input)[0]:
            raise HTTPException(422, "Fabric sessions need a user message")

    async def _sandbox_response(
        self, state: FabricSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        request_input, input_system = _extract_request_input(body.input)
        system_prompt = (
            "\n\n".join(part for part in (self.config.system_prompt, body.instructions, input_system) if part) or None
        )
        rollout_id = state.request.episode_id.capture_key
        config = self._fabric_config(
            model_base_url=self.resolve_model_base_url(self.config.model_server.name, rollout_id),
            workspace=Path(state.session.workdir),
            system_prompt=system_prompt,
            mcp_servers={},
            skills=[],
            relay_dir=Path(state.session.session_dir) / "relay" if self.config.relay else None,
        ).to_mapping()
        config["runtime"]["artifacts"] = f"{state.session.session_dir}/artifacts"
        timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        if state.request.agent_timeout_sec is not None:
            timeout = min(timeout, state.request.agent_timeout_sec)
        # Gym supervises the deadline, including subprocess cleanup; Fabric gets the same time budget. Fabric then gives
        # its adapters a deadline ten seconds earlier, and Claude and Codex report that as a "timeout" error result.
        config["runtime"]["timeout_seconds"] = timeout
        payload = {
            "config": config,
            "input": request_input,
            "request_id": rollout_id,
            "workdir": state.session.workdir,
        }
        failure = None
        result = None
        try:
            async with self.sem:
                result = await state.session.execute(
                    stage_activation=lambda: state.stage_activation(payload),
                    collect=state.collect_artifacts,
                    timeout=timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            failure = error
            result = state.session.artifacts
        cleanup = state.session.cleanup
        timed_out = bool(cleanup and cleanup["timed_out"])
        # The adapter stopped itself at its own deadline: an incomplete rollout to verify, like a supervisor timeout.
        adapter_timeout = failure is None and _mapping(_mapping(result).get("error")).get("code") == "timeout"
        if cleanup:
            access = state.request.sandbox_access
            state.observations.records.append(
                SandboxObservation(
                    role="agent",
                    provider=access.connection.provider_config_ref if access else self.config.sandbox_provider,
                    outcome=(
                        "timeout"
                        if timed_out or adapter_timeout
                        else "failed"
                        if failure or cleanup["return_code"]
                        else "completed"
                    ),
                    exit_code=cleanup["return_code"],
                )
            )
        if timed_out and state.session.cleanup_confirmed and not state.session.closing:
            if result is not None:
                response, _ = self._response_from_result(body, result, result.get("relay_atif") or [], incomplete=True)
                if not _has_token_usage(_normalized_usage(result, _mapping(result.get("output")))):
                    response.usage = None
            else:
                state.observations.gaps.append(ObservationGap(code="partial_fabric_result_unavailable"))
                response = NeMoGymResponse(
                    id=f"resp_{uuid4().hex}",
                    created_at=int(time()),
                    model=self.config.model,
                    object="response",
                    status="incomplete",
                    output=[],
                    usage=None,
                    tools=body.tools,
                    tool_choice=body.tool_choice,
                    parallel_tool_calls=body.parallel_tool_calls,
                )
            return self.report_rollout_stop(response, state.session)
        if failure is not None:
            raise failure
        if state.session.closing:
            raise asyncio.CancelledError
        if cleanup and cleanup["return_code"] != 0:
            raise RuntimeError(f"Fabric worker exited unsuccessfully: {cleanup['return_code']}")
        response, _ = self._response_from_result(
            body, result, result.get("relay_atif") or [], incomplete=adapter_timeout
        )
        if not _has_token_usage(_normalized_usage(result, _mapping(result.get("output")))):
            response.usage = None
            state.observations.gaps.append(ObservationGap(code="usage_unavailable"))
        if self.config.adapter_id == "nvidia.fabric.mini-swe-agent" and not result.get("relay_atif"):
            state.observations.gaps.append(
                ObservationGap(
                    code="harness_trajectory_unavailable",
                    detail="Fabric mini-SWE returns final output and API-call count without intermediate turns when Relay is off",
                )
            )
        return response

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Serve /v1/responses as one agent-session activation, or as a host-local Fabric run without a session.

        With an agent-session cookie, run the seeded session's single Fabric activation in its sandbox. An identical
        retry joins that activation; a different body or rollout, or a closing session, answers 409, and a request
        control the session path cannot honor answers 422. Without a session (AnySWE, AnyTerminal and the host /run
        path), run Fabric on the agent-server host.
        """
        session_id = self._agent_session_id_from_request(request)
        if session_id is not None:
            state = self._require_agent_session(session_id)
            assert isinstance(state, FabricSandboxSession)
            if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
                raise HTTPException(409, "Fabric activation does not match its seeded rollout")
            if state.session.closing:
                raise HTTPException(409, "Fabric session is closing")
            self._validate_session_request(body)
            if state.task is None:
                state.activation_request = body.model_copy(deep=True)
                state.task = asyncio.create_task(self._sandbox_response(state, state.activation_request))
            elif body != state.activation_request:
                raise HTTPException(409, "Fabric sessions support one activation; retry the same request")
            return (await asyncio.shield(state.task)).model_copy(deep=True)
        rollout_id = request.path_params.get("rollout_id")
        response, _ = await self._create_response(body, rollout_id=rollout_id)
        return response

    async def run(self, request: Request, body: NeMoFabricAgentRunRequest) -> NeMoFabricAgentVerifyResponse:
        """Compatibility episode: seed Resources, run Fabric on this host, then verify.

        Requires resources_server. Sessions through an Environment Server do not use this route.
        """
        if self.config.resources_server is None:
            raise HTTPException(
                422, "Fabric /run requires resources_server; compose agent sessions through an Environment Server"
            )
        if self.sem is None:  # pragma: no cover
            raise RuntimeError("NeMo Fabric agent concurrency control is not initialized")
        async with self.sem:
            cookies = request.cookies
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_response)
            cookies = seed_response.cookies
            seed_json = await get_response_json(seed_response)

            skills_path = ((body.model_extra or {}).get(SKILLS_REF_KEY_NAME) or {}).get("path")
            agent_response, fabric_result = await self._create_response(
                body.responses_create_params,
                mcp_servers=self._rollout_mcp_servers(seed_json),
                skills_path=skills_path,
                rollout_id=self.rollout_id_from_run(body),
            )
            agent_json = agent_response.model_dump(mode="json")

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_json},
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            verify_json = await get_response_json(verify_response)
            fabric_output = _mapping(fabric_result.get("output"))
            finished_naturally = fabric_result.get("status") == "succeeded" and not fabric_output.get("failed", False)
            if finished_naturally and "completed" in fabric_output:
                finished_naturally = bool(fabric_output["completed"])
            return NeMoFabricAgentVerifyResponse.model_validate(
                verify_json
                | {
                    "turns_used": _turns_used(fabric_output),
                    "finished_naturally": finished_naturally,
                    "fabric_result": fabric_result,
                }
            )


if __name__ == "__main__":
    NeMoFabricAgent.run_webserver()
