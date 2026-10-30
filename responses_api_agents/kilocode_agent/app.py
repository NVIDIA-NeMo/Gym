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

import asyncio
import copy
import json
import logging
import os
import re
import shlex
import shutil
import signal
from asyncio import Semaphore
from pathlib import Path
from time import perf_counter, time
from typing import Any, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from openai.types.responses import ResponseError
from pydantic import ConfigDict, Field, JsonValue, PrivateAttr

from nemo_gym.agent_utils.sandbox_session import HARNESS_COMMAND_NOT_STARTED, HARNESS_RUNNER_FAILED, SandboxSession
from nemo_gym.agent_utils.supervisor_client import CleanupReceipt
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef, get_global_config_dict
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap, SandboxObservation
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.tool_access import MCPToolAccess
from responses_api_agents.kilocode_agent.observability import read_kilo_observations
from responses_api_agents.kilocode_agent.sandbox import KiloArtifacts, KiloSandboxSession
from responses_api_agents.kilocode_agent.setup_kilocode import ensure_kilo


LOG = logging.getLogger(__name__)


def _think_message(index: int, text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"msg-{index}",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


def parse_kilo_events(stdout: str) -> tuple[list[Any], dict[str, int]]:
    """Convert ``kilo run --format json`` stdout into (output_items, usage).

    Kilo (an OpenCode fork) emits one JSON object per line for each root-session
    ``message.part.updated`` event: ``{"type", "timestamp", "sessionID", "part"|"error"}``. The
    ``part`` objects reuse OpenCode's part shapes, so the mapping mirrors the OpenCode agent's
    sqlite parser: ``text`` -> assistant message, ``tool_use`` -> function_call (+ output),
    ``step_finish`` -> token usage, ``reasoning`` -> buffered <think> block prepended to the next
    assistant message (a trailing reasoning run is surfaced on its own so it is not dropped).
    ``error`` events are left to ``_scan_run_events`` and never logged here: they can carry MCP grant
    credentials, which only the session path redacts.
    """
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0
    buffered_think: Optional[str] = None
    # kilo's `--format json` writer emits every message.part.updated event twice with the same part
    # id (persists with KILO_NO_DAEMON=1, so it is the writer, not a daemon double-subscription). The
    # two copies are identical for every event type we parse, so first-wins dedup by part id counts
    # each tool call, message, and token total once. Verified against @kilocode/cli 7.4.15.
    seen_part_ids: set[str] = set()
    warned_empty_length = False

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        etype = event.get("type")
        part = event.get("part") or {}
        if not isinstance(part, dict):
            part = {}

        part_id = part.get("id")
        if part_id is not None:
            if part_id in seen_part_ids:
                continue
            seen_part_ids.add(part_id)

        if etype == "step_finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            input_tokens += int(tokens.get("input") or 0) + int(cache.get("read") or 0)
            step_output = int(tokens.get("output") or 0)
            output_tokens += step_output
            # Stopping on "length" with nothing generated is what an over-window request looks like by
            # the time it reaches kilo; see KiloCodeAgentConfig.max_output_tokens. Warn once: the run
            # is otherwise silent about why it produced no answer.
            if part.get("reason") == "length" and not step_output and not warned_empty_length:
                warned_empty_length = True
                LOG.warning(
                    "kilo step stopped on 'length' with no output tokens; max_output_tokens is likely "
                    "too large for the model server's context window"
                )

        elif etype == "reasoning":
            text = (part.get("text") or "").strip()
            if text:
                buffered_think = (buffered_think + "\n" + text) if buffered_think else text

        elif etype == "text":
            text = part.get("text") or ""
            if not text.strip():
                continue
            if buffered_think:
                text = f"<think>\n{buffered_think}\n</think>\n\n{text}"
                buffered_think = None
            output_items.append(_think_message(len(output_items), text))

        elif etype == "tool_use":
            state = part.get("state") or {}
            call_id = part.get("callID") or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=part.get("tool", ""),
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )
            output = state.get("output")
            if output is None and state.get("status") == "error":
                output = state.get("error") or ""
            if output is not None:
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=str(output),
                        status="completed",
                    )
                )

    # A run that ends on reasoning (some vLLM reasoning parsers route the final answer through the
    # reasoning channel) is surfaced as a think-tagged message rather than dropped.
    if buffered_think:
        output_items.append(_think_message(len(output_items), f"<think>\n{buffered_think}\n</think>"))

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _scan_run_events(stdout: str) -> tuple[Any, bool]:
    """Return the CLI's first `error` payload (or None) and whether a model step stopped on length."""
    error, length_stop = None, False
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error" and error is None:
            error = event.get("error") or "unknown error"
        elif event.get("type") == "step_finish":
            length_stop = length_stop or (event.get("part") or {}).get("reason") == "length"
    return error, length_stop


def _error_name(error: Any) -> str:
    name = error.get("name") if isinstance(error, dict) else None
    return name if isinstance(name, str) and name else "kilo_api_error"


def _runtime_failure(artifacts: KiloArtifacts, cleanup: CleanupReceipt | None) -> str | None:
    """Return why a run failed at runtime, from the supervisor receipt; a timeout stays a gradable partial."""
    if cleanup is None:
        return "Kilo activation ended without a cleanup receipt"
    if cleanup["error"]:
        return f"Kilo supervision failed: {cleanup['error']}"
    if not artifacts.task_environment_ready:
        # Kilo continues when a plugin fails to load; its tools then ran with Kilo's private HOME.
        return "Kilo did not confirm its task-environment plugin"
    if not artifacts.required_mcp_ready:
        # The plugin confirms required MCP servers before the first model request.
        return "Kilo did not confirm its required MCP servers before model activation"
    if cleanup["timed_out"]:
        return None
    if cleanup["return_code"] == HARNESS_RUNNER_FAILED:
        return f"Kilo runner failed before starting Kilo (exit {HARNESS_RUNNER_FAILED})"
    if cleanup["return_code"] in HARNESS_COMMAND_NOT_STARTED:
        return f"Kilo could not be started (exit {cleanup['return_code']})"
    if cleanup["return_code"] not in (0, None):
        return f"Kilo exited {cleanup['return_code']}"
    if artifacts.exit_code not in (0, None):
        return f"Kilo exited {artifacts.exit_code}"
    if artifacts.exit_code is None:
        return "Kilo exit status is unavailable"
    return None


def _run_outcome(
    artifacts: KiloArtifacts, cleanup: CleanupReceipt | None, *, cancelled: bool
) -> tuple[str, str | None]:
    """Classify one sandbox run as a SandboxObservation (outcome, error_type) pair."""
    error, _ = _scan_run_events(artifacts.stdout)
    if error is not None:
        return "failed", _error_name(error)
    if not artifacts.task_environment_ready:
        return "failed", "task_environment_unconfirmed"
    if not artifacts.required_mcp_ready:
        return "failed", "required_mcp_unconfirmed"
    if cleanup is not None and cleanup["timed_out"]:
        return "timeout", "agent_timeout"
    if cancelled:
        return "cancelled", "cancelled"
    if _runtime_failure(artifacts, cleanup):
        return "failed", "agent_run_error"
    return "completed", None


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    """Return (user_message, system_message) from a responses body input list."""
    items = list(body_input)
    system_message: Optional[str] = None

    if items:
        first = items[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "system":
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            system_message = content or ""
            items = items[1:]

    user_message = ""
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            user_message = content or ""
            break

    return user_message, system_message


class KiloCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = None
    # When set, kilo's model calls go through this Gym model server instead of straight to a provider,
    # so they are captured. `model` is then the bare model name; the agent registers it under a `nemo`
    # provider pointed at the server (see _build_kilo_config).
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "kilo"
    model: str = "openai/gpt-4o-mini"
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/kilocode_agent/workspaces"
    repo_dir: Optional[str] = None
    thinking: bool = False
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    # written to kilo.json in the run dir (OpenCode-compatible config schema)
    kilo_config: dict[str, Any] = Field(default_factory=dict)
    # The three fields below describe the served model to kilo. They apply only to the `model_server`
    # path, where the agent owns the `nemo` model entry; a provider declared by hand in `kilo_config`
    # keeps whatever it declares. They are config fields rather than `kilo_config` keys because the
    # config merge is struct-mode, so a config using `_inherit_from` cannot add keys under `models`.
    #
    # `context_window` is what kilo measures a session against. It only drives auto-compaction when
    # `kilo_config` also sets `compaction.threshold_percent`; 0 disables that accounting entirely.
    context_window: int = 32768
    # `max_output_tokens` is the per-request output budget; kilo asks the model server for
    # min(this, its own OUTPUT_TOKEN_MAX of 32000), so values above 32000 have no effect.
    #
    # Setting it too high fails silently, which is why the default is conservative. Kilo's system
    # prompt and tool definitions run to ~10k tokens, and vLLM rejects prompt + max_tokens >
    # max_model_len with a 400 that the Gym model server turns into an empty completion with
    # finish_reason "length" (vllm_model/app.py `is_out_of_context_length`) rather than an error, so
    # the run yields no assistant message at all. Raise it only alongside a model server whose window
    # has room for it. Verified against @kilocode/cli 7.4.15 and vLLM serving a 32768-token window.
    max_output_tokens: int = 8192
    # Response field carrying reasoning text, for kilo's `interleaved.field`. Gym model servers write
    # `reasoning_content` (vLLM >= 0.16 also sends `reasoning`, which is why this is configurable).
    # Kilo turns interleaved reasoning off for custom openai-compatible providers unless the field is
    # named, so without this the reasoning channel is dropped. Null leaves kilo's default in place.
    reasoning_field: Optional[str] = "reasoning_content"
    kilo_version: Optional[str] = None
    sandbox_provider: str | None = None
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class KiloCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class KiloCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )


# Carries local observations from the /run self-call's response to /run; never sent to the verifier.
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"


class KiloCodeAgent(SimpleResponsesAPIAgent):
    """Runs the Kilo Code CLI (``kilo run --auto --format json``).

    Kilo runs its own tools internally; we parse its JSON event stream into Gym format and use the
    resources server to verify. Kilo's model calls go through ``model_server`` when one is set — see
    that field. Eval-only either way: token IDs and logprobs are not wired up.
    """

    config: KiloCodeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _local_setup_task: asyncio.Task[None] | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> KiloSandboxSession:
        grants = self._mcp_grants(body)
        if self.config.model_server is None:
            raise HTTPException(422, "Kilo sandbox sessions require a Gym model_server")
        if not self.config.kilo_version or self.config.command != "kilo":
            raise HTTPException(422, "Kilo sandbox sessions require kilo_version and the default kilo command")
        if self.config.repo_dir or self.config.extra_args or self.config.env:
            raise HTTPException(
                422, "Kilo sandbox sessions do not support repo_dir, extra_args, or custom env overrides"
            )
        if self.config.timeout <= 0 or self.config.setup_timeout <= 0:
            raise HTTPException(422, "Kilo sandbox installation and execution timeouts must be positive")
        owns_sandbox = body.sandbox_access is None
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise HTTPException(422, "Kilo requires sandbox_access or a configured sandbox_provider")
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir") or "/app"
        else:
            if not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
                raise HTTPException(422, "Kilo requires direct SandboxAccess")
            provider_ref = body.sandbox_access.connection.provider_config_ref
            workdir = body.sandbox_access.workdir
        if not isinstance(workdir, str) or not workdir.startswith("/") or ".." in workdir.split("/"):
            raise HTTPException(422, "Kilo sandbox workdir must be an absolute path without '..'")
        session_dir = f"/tmp/nemo-gym-kilo-sessions/{uuid4().hex}"
        if session_dir.startswith(workdir.rstrip("/") + "/") or workdir.startswith("/tmp/nemo-gym-kilo-sessions"):
            raise HTTPException(422, "Kilo task workdir must be disjoint from its session directory")
        global_config = get_global_config_dict()
        provider = create_provider(resolve_provider_config(provider_ref, global_config))
        if owns_sandbox:
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(body.sandbox_access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        # Retain the session before starting an owned sandbox: start() can allocate it and then fail, and a
        # retained session lets close stop it.
        state = KiloSandboxSession(
            request=body,
            runtime_dir=session_dir,
            required_mcp_servers=tuple(access.name for access in grants if access.required),
            mcp_header_values=tuple(
                value for access in grants for value in access.connection.headers.values() if value
            ),
            provider_name=provider.name,
            sandbox_id=sandbox.sandbox_id,
            session=SandboxSession(
                sandbox=sandbox,
                session_dir=session_dir,
                workdir=workdir,
                owns_sandbox=owns_sandbox,
                harness="KiloCode",
            ),
        )
        try:
            if owns_sandbox:
                settings = self.config.sandbox_config.copy()
                settings["metadata"] = {
                    **resolve_provider_metadata(provider_ref, global_config),
                    **(settings.get("metadata") or {}),
                    "gym-agent": "kilocode",
                    "gym-session": body.agent_session_id[:63],
                    "gym-rollout": body.episode_id.capture_key[:63],
                }
                settings.setdefault("workdir", workdir)
                setup_budget = self.config.setup_timeout * (2 if state.required_mcp_servers else 1)
                settings.setdefault("ttl_s", setup_budget + self.config.timeout + 600)
                await sandbox.start(SandboxSpec(**settings))
                state.sandbox_id = sandbox.sandbox_id
                # Providers need not create the workdir; borrowed task preparation belongs to Resources.
                workspace = await sandbox.exec(f"mkdir -p -- {shlex.quote(workdir)}", cwd="/", timeout_s=30)
                if workspace.return_code != 0 or workspace.error_type:
                    raise RuntimeError(f"Cannot create Kilo sandbox workdir: {workspace.stderr or workspace.stdout}")
            await state.install_runtime(version=self.config.kilo_version, timeout=self.config.setup_timeout)
            if state.required_mcp_servers:
                await state.prepare_mcp(
                    config=json.dumps(self._sandbox_config(state, rollout_id=body.episode_id.capture_key)),
                    timeout=self.config.setup_timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            try:
                await state.close(timeout=self.config.session_close_timeout_seconds)
            except BaseException:
                LOG.exception("Kilo setup cleanup failed; retaining session for close")
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    def _mcp_grants(self, body: AgentSeedSessionRequest) -> list[MCPToolAccess]:
        """Validate external tool access before allocating a sandbox or overwriting static config."""
        accesses = self.effective_tool_accesses(body)
        unsupported = [access.name for access in accesses if access.required and not isinstance(access, MCPToolAccess)]
        if unsupported:
            raise HTTPException(
                422, "Kilo supports only MCP tool grants; unsupported required grants: " + ", ".join(unsupported)
            )
        grants = [access for access in accesses if isinstance(access, MCPToolAccess)]
        static = self.config.kilo_config.get("mcp") or {}
        if not isinstance(static, dict):
            raise HTTPException(422, "kilo_config.mcp must be a mapping")
        names: dict[str, str] = {}
        for name in [*static, *(access.name for access in grants)]:
            # Kilo uses this namespace in model-visible MCP tool names.
            normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
            if normalized in names:
                raise HTTPException(422, f"Kilo MCP server names collide: {names[normalized]!r} and {name!r}")
            names[normalized] = name
        return grants

    def _sandbox_config(self, state: KiloSandboxSession, *, rollout_id: str) -> dict[str, Any]:
        config = self._build_kilo_config(self._resolve_model_base_url(rollout_id))
        # The no-model seed preflight must resolve the same provider/model as activation.
        config["model"] = self._effective_model()
        grants = self._mcp_grants(state.request)
        if grants:
            # As in _mcp_grants, a null kilo_config.mcp means no static servers.
            servers = config["mcp"] = config.get("mcp") or {}
            for access in grants:
                servers[access.name] = {
                    "type": "remote",
                    "url": str(access.connection.url),
                    "headers": dict(access.connection.headers),
                    "enabled": True,
                    "oauth": False,
                }
        config.setdefault("plugin", []).append(state.task_environment_plugin)
        return config

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, KiloSandboxSession):
            raise TypeError("Expected Kilo sandbox session")
        await state.close(timeout=self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=self._sandbox_observations(state),
        )

    def _sandbox_observations(self, state: KiloSandboxSession) -> AgentObservationBundle:
        """Annotate the persisted session tree with the root run's outcome and sandbox identity."""
        artifacts = state.session.artifacts
        sandbox = SandboxObservation(role="agent", provider=state.provider_name, sandbox_id=state.sandbox_id)
        if artifacts is None:
            sandbox.error_type = "artifact_capture_failed"
            return AgentObservationBundle(
                source="kilocode", records=[sandbox], gaps=[ObservationGap(code="observation_capture_failed")]
            )
        outcome, error_type = _run_outcome(
            artifacts, state.session.cleanup, cancelled=state.task is not None and state.task.cancelled()
        )
        sandbox.outcome, sandbox.error_type = outcome, error_type
        # A cancelled Kilo exits because the stop told it to; that status is not one Kilo chose.
        sandbox.exit_code = None if outcome == "cancelled" else artifacts.exit_code
        sandbox.wall_time_s = artifacts.wall_time_s
        _, length_stop = _scan_run_events(artifacts.stdout)
        if artifacts.observations is None:
            # The CLI stdout carries only the root session; without the database there is no session tree.
            observations = AgentObservationBundle(
                source="kilocode", gaps=[ObservationGap(code="agent_artifact_unavailable")]
            )
        else:
            observations = artifacts.observations.model_copy(deep=True)
        for record in observations.records:
            if isinstance(record, AgentInvocation) and record.parent_invocation_id is None:
                if outcome == "failed" or record.status == "failed":
                    record.status = "failed"
                elif outcome != "completed" or length_stop:
                    record.status = "incomplete"
                record.error_type = record.error_type or error_type
                record.duration_ms = artifacts.wall_time_s * 1000 if artifacts.wall_time_s is not None else None
        if artifacts.runner_hostname is None or artifacts.runner_pid is None:
            observations.gaps.append(ObservationGap(code="runtime_info_unavailable"))
        if state.session.cleanup is not None and state.session.cleanup["return_code"] is None:
            observations.gaps.append(ObservationGap(code="worker_exit_code_unavailable"))
        observations.records.append(sandbox)
        return observations

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                KiloCodeAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"kilo_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _repo_dir(self, fallback: Path) -> Path:
        """Return the configured persistent repository or the temporary fallback."""
        if not self.config.repo_dir:
            return fallback
        root = Path(self.config.repo_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> str:
        """The Gym model server's ``/v1`` URL with the per-rollout capture prefix, "" if unconfigured."""
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    def _effective_model(self) -> str:
        """The ``-m`` argument: ``model`` lives under the agent-owned `nemo` provider when one is set."""
        return f"nemo/{self.config.model}" if self.config.model_server else self.config.model

    def _build_kilo_config(self, model_base_url: str = "") -> dict[str, Any]:
        """Assemble the kilo.json contents from `kilo_config` plus the Gym model server wiring.

        Kilo resolves `-m <provider>/<name>` against that provider's `models` map (splitting on the
        first `/`, so a slashed model name stays intact) and fails the run with "Model not found" if
        the name is absent. Either branch below registers `model` there, so `model` is the only place
        a config names it; without that, every config has to repeat it under
        kilo_config.provider.<provider>.models, which downstream overrides cannot add to (the config
        merge is struct-mode, so a new model key is rejected).
        """
        config = self._deep_merge({}, copy.deepcopy(self.config.kilo_config))

        if self.config.model_server:
            # A generic openai-compatible provider aimed at the Gym model server. setdefault
            # throughout so a config that wants to override any of this still can; only baseURL is
            # forced, since it is resolved per run and a stale one would silently bypass Gym.
            # Evaluation rollouts do not need an auxiliary model call to title the session (it also
            # consumes scripted replies in conformance probes); kilo_config can turn it back on.
            config.setdefault("agent", {}).setdefault("title", {}).setdefault("disable", True)
            provider = config.setdefault("provider", {}).setdefault("nemo", {})
            provider.setdefault("npm", "@ai-sdk/openai-compatible")
            options = provider.setdefault("options", {})
            options.setdefault("apiKey", "EMPTY")  # pragma: allowlist secret
            options["baseURL"] = model_base_url
            # The model server returns a completion only once it is finished, so Kilo's default 60s
            # stream watchdog (keep-alives do not reset it) would cancel long calls. Match the run budget.
            options.setdefault("chunkTimeout", self.config.timeout * 1000)
            model = provider.setdefault("models", {}).setdefault(self.config.model, {})
            model.setdefault("name", self.config.model)
            model.setdefault("limit", {"context": self.config.context_window, "output": self.config.max_output_tokens})
            if self.config.reasoning_field:
                model.setdefault("interleaved", {"field": self.config.reasoning_field})
        else:
            provider_id, _, model_name = self.config.model.partition("/")
            provider = (config.get("provider") or {}).get(provider_id)
            if model_name and isinstance(provider, dict):
                provider.setdefault("models", {}).setdefault(model_name, {})
        return config

    def _write_kilo_config(self, work_dir: Path, model_base_url: str = "") -> None:
        config = self._build_kilo_config(model_base_url)
        if not config:
            return
        (work_dir / "kilo.json").write_text(json.dumps(config, indent=2))

    def _env(self, data_home: str, config_home: str, model_base_url: str = "", db: str = ":memory:") -> dict[str, str]:
        # Per-run isolation. KILO_NO_DAEMON=1 forces a fresh embedded server (kilo's daemon client is
        # enabled iff this is unset), so concurrent rollouts share no daemon db/state. KILO_DB is
        # per run: a file in the run dir when observations are read from it, else :memory:. XDG_DATA_HOME/XDG_CONFIG_HOME point
        # inside the run dir so global ~/.local/share/kilo and ~/.config/kilo never leak in. Indexing
        # is handled by --pure (see _build_command), not here: KILO_DISABLE_CODEBASE_INDEXING only
        # triggers on the literal value "vscode-no-workspace", not a truthy one, so it is unusable.
        env = {
            **os.environ,
            "KILO_NO_DAEMON": "1",
            "KILO_DB": db,
            "XDG_DATA_HOME": data_home,
            "XDG_CONFIG_HOME": config_home,
        }
        # A Gym model server wins over openai_base_url, so a config carrying both does not point the
        # subprocess environment at a provider the `nemo` provider is meant to replace.
        base_url = model_base_url or self.config.openai_base_url
        api_key = "EMPTY" if model_base_url else self.config.openai_api_key  # pragma: allowlist secret
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        env.update({k: v for k, v in self.config.env.items() if v})
        return env

    def _build_command(self, project_dir: Path, prompt: str) -> list[str]:
        cmd = [
            *self.config.command_parts,
            "run",
            "--auto",
            # --pure disables external plugins, so codebase indexing never starts; built-in
            # bash/edit/webfetch tools and the custom provider are unaffected.
            "--pure",
            "--format",
            "json",
            "-m",
            self._effective_model(),
            "--dir",
            str(project_dir),
        ]
        if self.config.thinking:
            cmd.append("--thinking")
        cmd.extend(self.config.extra_args)
        # `--` terminates option parsing so a prompt beginning with '-' is passed as the message.
        cmd += ["--", prompt]
        return cmd

    @staticmethod
    def _kill_process_group(proc: "asyncio.subprocess.Process") -> None:
        # `kilo` is an npm shim whose child is the real binary; killing only the shim orphans the
        # child holding the stdout pipe, so kill the whole process group (start_new_session=True).
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()

    async def _run_kilo(
        self, instruction: str, system_prompt: Optional[str], rollout_id: Optional[str] = None
    ) -> tuple[str, str, int | None, bool, AgentObservationBundle | None, NeMoGymResponseUsage | None]:
        """Run one headless kilo run.

        Returns (stdout, stderr, exit_code, timed_out, observations, usage). Observations and the database's
        usage are read only for rollout-prefixed calls, the only ones that report them.
        """
        # Share one installation across concurrent local calls. Cancelling a caller must not release
        # another caller to start a second installer against the same Node/npm paths.
        if self._local_setup_task is None:
            self._local_setup_task = asyncio.create_task(asyncio.to_thread(ensure_kilo, self.config.kilo_version))
        setup_task = self._local_setup_task
        try:
            await asyncio.shield(setup_task)
        except Exception:
            # A failed installation can be retried; a cancelled caller leaves the shared task running.
            if self._local_setup_task is setup_task:
                self._local_setup_task = None
            raise
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._repo_dir(work_dir)
        data_home = work_dir / ".kilo-data"
        config_home = work_dir / ".kilo-config"
        data_home.mkdir(parents=True, exist_ok=True)
        config_home.mkdir(parents=True, exist_ok=True)
        model_base_url = self._resolve_model_base_url(rollout_id)
        self._write_kilo_config(project_dir, model_base_url)
        db_path = data_home / "kilo.db"
        env = self._env(
            str(data_home), str(config_home), model_base_url, db=str(db_path) if rollout_id else ":memory:"
        )
        cmd = self._build_command(project_dir, prompt)

        # File-backed output keeps the events written before a timeout or cancellation kill.
        stdout_path, stderr_path = work_dir / "stdout.jsonl", work_dir / "stderr.log"
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(project_dir),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env=env,
                    start_new_session=True,
                )
            timed_out = False
            try:
                await asyncio.wait_for(proc.wait(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                timed_out = True
                LOG.warning("kilo timed out after %ds", self.config.timeout)
                self._kill_process_group(proc)
                await proc.wait()
            except BaseException:
                # A cancelled request must not leave Kilo running.
                self._kill_process_group(proc)
                raise
            observations, usage = None, None
            if rollout_id:
                try:
                    observations, usage = await asyncio.to_thread(
                        read_kilo_observations, db_path, fallback_invocation_id=rollout_id
                    )
                except Exception:
                    LOG.warning("Kilo session database unavailable; retaining CLI output", exc_info=True)
            return (
                stdout_path.read_bytes().decode(errors="replace"),
                stderr_path.read_bytes().decode(errors="replace"),
                proc.returncode,
                timed_out,
                observations,
                usage,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _validate_sandbox_request(body: NeMoGymResponseCreateParamsNonStreaming) -> None:
        """Reject request controls and conversation shapes this CLI cannot faithfully execute."""
        unsupported = body.model_dump(exclude_defaults=True, exclude_none=True).keys() - {
            "input",
            "instructions",
            "model",
        }
        if unsupported:
            raise HTTPException(422, "Unsupported Kilo sandbox request fields: " + ", ".join(sorted(unsupported)))
        if isinstance(body.input, str):
            return
        roles = [getattr(item, "role", None) for item in body.input]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(
                422, "Kilo sandbox input must be one user message with optional system/developer prefix"
            )
        for item in body.input:
            content = getattr(item, "content", None)
            if not isinstance(content, str) and (
                not isinstance(content, list)
                or any(
                    (part.get("type") if isinstance(part, dict) else getattr(part, "type", None)) != "input_text"
                    for part in content
                )
            ):
                raise HTTPException(422, "Kilo sandbox input supports text only")

    def _response_from_events(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        stdout: str,
        *,
        timed_out: bool,
        failure: str | None,
        diagnostics: str,
    ) -> tuple[NeMoGymResponse, str | None]:
        """Build the response from Kilo's events, with the reason the run failed (None when it did not)."""
        error, length_stop = _scan_run_events(stdout)
        output_items, usage = parse_kilo_events(stdout)
        if error is not None:
            problem = f"Kilo reported an execution error: {error}; {diagnostics}"
        elif failure is not None:
            problem = f"{failure}: {diagnostics}"
        elif timed_out and not output_items and not any(usage.values()):
            # Kilo retries a model it cannot reach until the deadline, without events. A timeout here
            # means an unreachable or failed model server, not the model's own partial work.
            problem = (
                "Kilo timed out before any model call completed; check that Kilo can reach the model server: "
                f"{diagnostics}"
            )
        elif not output_items and not (timed_out or length_stop):
            problem = "Kilo completed without any trajectory events"
        else:
            problem = None
        response = self._build_response(
            body, output_items, usage, self.config.model, incomplete=timed_out or length_stop
        )
        return response, problem

    async def _sandbox_response(
        self, state: KiloSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, *, rollout_id: str
    ) -> NeMoGymResponse:
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        # The legacy extractor recognizes system messages only. Normalize a developer prefix on this copy.
        items = [
            item.model_copy(update={"role": "system"}) if getattr(item, "role", None) == "developer" else item
            for item in items
        ]
        user_message, input_system = _extract_instruction(items)
        prompt = "\n\n".join(
            p for p in [self.config.system_prompt, body.instructions, input_system, user_message] if p
        )
        command = self._build_command(Path(state.session.workdir), prompt)
        command[:1] = state.kilo_argv
        # --pure only skips plugins named in kilo.json; here that would skip the task-environment plugin.
        command.remove("--pure")
        config = self._sandbox_config(state, rollout_id=rollout_id)
        async with self.sem:
            state.started_at = perf_counter()
            artifacts = await state.session.execute(
                stage_activation=lambda: state.stage_activation(
                    command=command,
                    config=json.dumps(config),
                ),
                collect=state.collect_artifacts,
                timeout=self.config.timeout,
                close_timeout=self.config.session_close_timeout_seconds,
            )
        response, problem = self._response_from_events(
            body,
            artifacts.stdout,
            timed_out=bool(state.session.cleanup and state.session.cleanup["timed_out"]),
            failure=_runtime_failure(artifacts, state.session.cleanup),
            diagnostics=artifacts.log_tail or artifacts.stderr[-4096:],
        )
        if problem is not None:
            # Agent sessions must not hand a failed run to the verifier, where it would score as a zero.
            raise RuntimeError(state.redact_diagnostics(problem))
        response.metadata = {
            "harness_execution": "sandbox",
            **(
                {"harness_hostname": artifacts.runner_hostname, "harness_pid": str(artifacts.runner_pid)}
                if artifacts.runner_hostname is not None and artifacts.runner_pid is not None
                else {}
            ),
            "kilo_version": str(self.config.kilo_version),
        }
        if artifacts.usage is not None:
            response.usage = artifacts.usage.model_copy(deep=True)
        else:
            response.usage.input_tokens_details.cached_tokens = None
            response.usage.output_tokens_details.reasoning_tokens = None
        return response

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is not None:
            self._validate_sandbox_request(body)
            if body.model is not None and body.model != self.config.model:
                raise HTTPException(422, "Kilo request model must match the configured model")
            state = self._require_agent_session(session_id)
            if not isinstance(state, KiloSandboxSession):
                raise TypeError("Expected Kilo sandbox session")
            rollout_id = request.path_params.get("rollout_id")
            if rollout_id != state.request.episode_id.capture_key:
                raise HTTPException(409, "Kilo agent session requires its attempt-qualified rollout route")
            if state.session.closing:
                raise HTTPException(409, "Kilo session is closing")
            if state.task is None:
                state.activation_request = body.model_copy(deep=True)
                state.task = asyncio.create_task(
                    self._sandbox_response(state, state.activation_request, rollout_id=rollout_id)
                )
            elif body != state.activation_request:
                raise HTTPException(409, "Kilo sandbox sessions support one activation; retry the same request")
            return (await asyncio.shield(state.task)).model_copy(deep=True)
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        # run() reaches this handler through the /ng-rollout/<id> self-call route, which carries the
        # correlation id kilo's model calls are tagged with. Absent on a direct /v1/responses call.
        rollout_id = request.path_params.get("rollout_id")
        stdout, stderr, exit_code, timed_out, observations, db_usage = await self._run_kilo(
            user_message, system_prompt, rollout_id
        )
        response, problem = self._response_from_events(
            body,
            stdout,
            timed_out=timed_out,
            failure=None if timed_out or exit_code == 0 else f"Kilo exited {exit_code}",
            diagnostics=stderr[-4096:],
        )
        if problem is not None:
            # The legacy /run path still verifies (the conformance convention), so a failed run is
            # marked failed rather than passed off as a completed answer.
            LOG.warning("Kilo run failed: %s", problem[:500])
            response.status = "failed"
            response.error = ResponseError(code="server_error", message=problem[:2000])
        if db_usage is not None:
            response.usage = db_usage
        if not isinstance(rollout_id, str):
            return response
        if observations is None:
            observations = AgentObservationBundle(
                source="kilocode",
                records=[AgentInvocation(invocation_id=rollout_id)],
                gaps=[ObservationGap(code="agent_artifact_unavailable")],
            )
        for record in observations.records:
            if isinstance(record, AgentInvocation) and record.parent_invocation_id is None:
                if problem is not None:
                    record.status = "failed"
                elif response.status == "incomplete":
                    record.status = "incomplete"
        observations.gaps.append(ObservationGap(code="no_sandbox_runtime"))
        if timed_out:
            observations.gaps.append(ObservationGap(code="agent_run_timeout"))
        return response.model_copy(update={_INTERNAL_OBSERVATIONS_KEY: observations.model_dump(mode="json")})

    def _build_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        output_items: list[NeMoGymResponseOutputItem],
        usage: dict[str, int],
        model_name: str,
        *,
        incomplete: bool = False,
    ) -> NeMoGymResponse:
        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("Kilo produced no assistant message. Padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            status="incomplete" if incomplete else "completed",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def run(self, request: Request, body: KiloCodeAgentRunRequest) -> KiloCodeAgentVerifyResponse:
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "Kilo sessions require EnvironmentServer /run")
        if self.config.resources_server is None:
            raise HTTPException(
                422, "Kilo /run requires resources_server; use EnvironmentServer /run for sandbox sessions"
            )
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            raw_observations = agent_resp_json.pop(_INTERNAL_OBSERVATIONS_KEY, None)
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
            )

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return KiloCodeAgentVerifyResponse.model_validate(
                verify_json
                | {"turns_used": turns, "finished_naturally": naturally}
                | ({"ng_agent_observations": observations} if observations is not None else {})
            )


if __name__ == "__main__":
    KiloCodeAgent.run_webserver()
