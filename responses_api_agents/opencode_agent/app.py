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
import tempfile
from asyncio import Semaphore
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from shlex import quote
from time import monotonic, time
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    SandboxObservation,
    TrajectoryRecord,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, create_provider
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import (
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from responses_api_agents.opencode_agent.artifacts import (
    _parse_opencode_session,
    parse_opencode_export,
    parse_opencode_observations,
    parse_opencode_session,
)
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession
from responses_api_agents.opencode_agent.setup_opencode import ensure_opencode


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"
_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"
_NATIVE_SESSION_KEY = "nemo_gym_opencode_native_session"


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


class OpenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = None
    execution_mode: Literal["local", "sandbox", "legacy_sandbox"] = "local"
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "opencode"
    model: str = "openai/gpt-4o-mini"
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/opencode_agent/workspaces"
    repo_dir: Optional[str] = None
    thinking: bool = True
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    opencode_config: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 262144
    max_output_tokens: int = 131072
    opencode_version: Optional[str] = None

    # Native sandbox setup and lifecycle. Resources owns the sandbox itself.
    remote_opencode_install_script_path: str | None = None
    remote_opencode_binary_path: str | None = None
    remote_opencode_musl_binary_path: str | None = None
    session_lifetime_seconds: float = Field(default=21600, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    session_close_retry_window_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)

    # Temporary legacy_sandbox compatibility; unused by native sandbox sessions.
    opencode_max_context_window: int = 262144
    sandbox_provider: str = "sandbox"
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_timeout: float = 10800
    debug: bool = False

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )


class OpenCodeAgent(SimpleResponsesAPIAgent):
    """Run OpenCode locally or in a Resources-owned native sandbox session."""

    config: OpenCodeAgentConfig
    _native_sessions: dict[str, OpenCodeSandboxSession] = PrivateAttr(default_factory=dict)
    _closed_native_sessions: OrderedDict[str, tuple[EpisodeId, AgentCloseSessionResponse, float]] = PrivateAttr(
        default_factory=OrderedDict
    )
    _native_session_locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _native_session_expiry_tasks: dict[str, asyncio.Task[None]] = PrivateAttr(default_factory=dict)
    _native_session_tombstones: OrderedDict[str, tuple[EpisodeId, float]] = PrivateAttr(default_factory=OrderedDict)
    _local_runtime_ready: bool = PrivateAttr(default=False)
    _legacy_agent: SimpleResponsesAPIAgent | None = PrivateAttr(default=None)
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.sem = Semaphore(self.config.concurrency)

    def _ensure_local_runtime(self) -> None:
        if self.config.execution_mode != "local":
            raise RuntimeError("Host OpenCode execution requires execution_mode=local")
        if self._local_runtime_ready:
            return
        ensure_opencode(self.config.opencode_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("opencode command %r is not on PATH yet", self.config.command)
        self._local_runtime_ready = True

    def _legacy(self) -> SimpleResponsesAPIAgent:
        if self.config.execution_mode != "legacy_sandbox":
            raise RuntimeError("The legacy sandbox bridge requires execution_mode=legacy_sandbox")
        if self._legacy_agent is None:
            from responses_api_agents.opencode_agent.legacy import LegacyOpenCodeAgent, LegacyOpenCodeAgentConfig

            values = self.config.model_dump()
            if values["opencode_version"] is None:
                # None preserves the local installer's default; the bridge has its own pinned default.
                values.pop("opencode_version")
            config = LegacyOpenCodeAgentConfig(
                **{key: value for key, value in values.items() if key in LegacyOpenCodeAgentConfig.model_fields}
            )
            self._legacy_agent = LegacyOpenCodeAgent(config=config, server_client=self.server_client)
        return self._legacy_agent

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenCodeAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"opencode_{uuid4().hex[:8]}"
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
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    def _effective_model(self) -> str:
        return f"nemo/{self.config.model}" if self.config.model_server else self.config.model

    def _build_opencode_config(self, rollout_id: Optional[str] = None) -> dict[str, Any]:
        config = self._deep_merge({}, copy.deepcopy(self.config.opencode_config))
        if self.config.model_server:
            providers = config.setdefault("provider", {})
            nemo = providers.setdefault("nemo", {"npm": "@ai-sdk/openai-compatible"})
            nemo.setdefault("options", {}).update(
                {"baseURL": self._resolve_model_base_url(rollout_id), "apiKey": "EMPTY"}  # pragma: allowlist secret
            )
            model = nemo.setdefault("models", {}).get(self.config.model, {})
            self._deep_merge(
                model,
                {
                    "name": self.config.model,
                    "interleaved": {"field": "reasoning"},
                    "limit": {"context": self.config.context_window, "output": self.config.max_output_tokens},
                },
            )
            nemo["models"] = {self.config.model: model}
        return config

    def _write_opencode_config(self, work_dir: Path, rollout_id: Optional[str] = None) -> None:
        config = self._build_opencode_config(rollout_id)
        if not config:
            return
        (work_dir / "opencode.json").write_text(json.dumps(config, indent=2))

    def _env(self, data_home: str, rollout_id: Optional[str] = None) -> dict[str, str]:
        env = {**os.environ, "XDG_DATA_HOME": data_home}
        base_url = (
            self._resolve_model_base_url(rollout_id) if self.config.model_server else self.config.openai_base_url
        )
        api_key = "EMPTY" if self.config.model_server else self.config.openai_api_key  # pragma: allowlist secret
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        env.update({k: v for k, v in self.config.env.items() if v})
        if self.config.model_server is not None:
            env["OPENAI_BASE_URL"] = base_url
            env["OPENAI_API_KEY"] = "EMPTY"  # pragma: allowlist secret
        return env

    async def _run_opencode(
        self,
        instruction: str,
        system_prompt: Optional[str],
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
        trajectory: Optional[TrajectoryRecord] = None,
    ) -> tuple[list[Any], dict[str, int], str, AgentObservationBundle]:
        """Run one headless OpenCode session and read its persisted artifact."""
        self._ensure_local_runtime()
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._repo_dir(work_dir)
        data_home = work_dir / ".opencode-data"
        data_home.mkdir(parents=True, exist_ok=True)
        self._write_opencode_config(project_dir, rollout_id)
        env = self._env(str(data_home), rollout_id)

        cmd = [*self.config.command_parts, "run", "-m", self._effective_model(), "--dir", str(project_dir)]
        if self.config.thinking:
            cmd.append("--thinking")
        cmd.extend(self.config.extra_args)
        cmd.append(prompt)

        try:
            timed_out = False
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                _, stderr = await proc.communicate()
                timed_out = True
                LOG.warning("opencode timed out after %ds", self.config.timeout)

            if proc.returncode not in (0, None):
                LOG.warning("opencode exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            db_path = data_home / "opencode" / "opencode.db"
            invocation_id = rollout_id or f"opencode-{uuid4().hex}"
            output_items, usage = (
                ([], {"input_tokens": 0, "output_tokens": 0}) if timed_out else parse_opencode_session(db_path)
            )
            observations = AgentObservationBundle(source="opencode")
            if collect_observations:
                try:
                    observations = _parse_opencode_session(db_path, invocation_id, trajectory)
                except Exception:
                    LOG.exception("failed to read OpenCode session artifact")
                    if trajectory is not None:
                        trajectory.gaps.append(ObservationGap(code="turns_unavailable"))
                    observations = AgentObservationBundle(
                        source="opencode",
                        records=[AgentInvocation(invocation_id=invocation_id)],
                        gaps=[
                            ObservationGap(code="agent_artifact_unavailable"),
                            ObservationGap(code="agent_transcript_unavailable"),
                            ObservationGap(code="model_call_ownership_unavailable"),
                        ],
                    )
            run_status = "incomplete" if timed_out else "completed" if proc.returncode == 0 else "failed"
            for invocation in observations.records:
                if not isinstance(invocation, AgentInvocation):
                    continue
                if invocation.parent_invocation_id is None:
                    invocation.status = run_status
            if timed_out:
                observations.gaps.append(ObservationGap(code="agent_run_timeout"))
            return output_items, usage, self.config.model, observations
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
    ) -> AgentEpisode:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        prompt = user_message if system_prompt is None else f"{system_prompt}\n\n{user_message}"
        trajectory = (
            TrajectoryRecord(task_id="unscoped", rollout_id=rollout_id or "unscoped") if collect_observations else None
        )

        output_items, usage, model_name, observations = await self._run_opencode(
            user_message,
            system_prompt,
            rollout_id=rollout_id,
            collect_observations=collect_observations,
            trajectory=trajectory,
        )
        if collect_observations:
            observations.gaps.append(ObservationGap(code="no_sandbox_runtime"))

        if collect_observations:
            root = next(
                (
                    record
                    for record in observations.records
                    if isinstance(record, AgentInvocation) and record.parent_invocation_id is None
                ),
                None,
            )
            if root is not None and not any(
                getattr(item, "role", None) in {"user", "system", "developer"} for item in root.conversation
            ):
                root.conversation = [NeMoGymEasyInputMessage(role="user", content=prompt), *root.conversation]

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("OpenCode produced no assistant message. Padding empty output")
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

        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
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
        if trajectory is not None:
            response = response.model_copy(update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")})
        return AgentEpisode(response=response, observations=observations)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._native_session_marker(request)
        if self.config.execution_mode == "sandbox":
            if session_id is None:
                raise HTTPException(409, "Native OpenCode requires a seeded agent session")
            state = self._native_sessions.get(session_id)
            if state is None or request.path_params.get("rollout_id") != state.seed.episode_id.capture_key:
                raise HTTPException(409, "OpenCode activation does not match the seeded session and rollout")
            if state.activated or state.closing:
                raise HTTPException(409, "Native OpenCode sessions allow one activation")
            prompt, system = self._native_input(body)
            state.activated = True
            state.task = asyncio.create_task(self._native_response(state, body, prompt=prompt, system=system))
            try:
                return await asyncio.shield(state.task)
            except asyncio.CancelledError:
                if not state.task.done() and not state.task.cancelling():
                    state.task.cancel()
                raise
        if session_id is not None:
            raise HTTPException(409, "Native OpenCode session markers cannot enter local or legacy execution")
        if self.config.execution_mode == "legacy_sandbox":
            return await self._legacy().responses(request, body)
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        episode = await self._create_episode(
            body,
            rollout_id=rollout_id,
            collect_observations=isinstance(rollout_id, str),
        )
        if not isinstance(rollout_id, str):
            return episode.response
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def run(self, request: Request, body: OpenCodeAgentRunRequest) -> OpenCodeAgentVerifyResponse:
        if self._native_session_marker(request) is not None or self.config.execution_mode == "sandbox":
            raise HTTPException(409, "Native OpenCode sessions must use EnvironmentServer /run")
        if self.config.execution_mode == "legacy_sandbox":
            return await self._legacy().run(request, body)
        if self.config.resources_server is None:
            raise HTTPException(422, "Local OpenCode /run requires resources_server")
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

            rollout_id = self.rollout_id_from_run(body)
            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            raw_trajectory = agent_resp_json.pop(_INTERNAL_TRAJECTORY_KEY, None)
            trajectory = (
                scope_opencode_trajectory(TrajectoryRecord.model_validate(raw_trajectory), body, rollout_id)
                if isinstance(raw_trajectory, dict) and rollout_id is not None
                else None
            )
            raw_observations = (
                agent_resp_json.pop(_INTERNAL_OBSERVATIONS_KEY, None) if rollout_id is not None else None
            )
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

            return OpenCodeAgentVerifyResponse.model_validate(
                verify_json
                | {"turns_used": turns, "finished_naturally": naturally}
                | ({"ng_agent_observations": observations} if observations is not None else {})
                | ({"ng_trajectory": trajectory.model_dump(mode="json")} if trajectory is not None else {})
            )

    def _native_session_marker(self, request: Request) -> str | None:
        try:
            session = request.session
        except (AssertionError, AttributeError):
            return None
        if not isinstance(session, Mapping) or _NATIVE_SESSION_KEY not in session:
            return None
        marker = session[_NATIVE_SESSION_KEY]
        if not isinstance(marker, str) or not marker:
            raise HTTPException(409, "Invalid native OpenCode session marker")
        return marker

    async def seed_agent_session(self, request: Request, body: AgentSeedSessionRequest) -> AgentSeedSessionResponse:
        """Install OpenCode once under the EnvironmentServer's caller-assigned identity."""
        if self.config.execution_mode != "sandbox":
            raise HTTPException(422, "Native OpenCode sessions require execution_mode=sandbox")
        self._expire_native_receipts()
        session_id = body.agent_session_id
        marker = self._native_session_marker(request)
        if marker is not None and marker != session_id and marker in self._native_sessions:
            raise HTTPException(409, "OpenCode request is already bound to another session")
        lock = self._native_session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            if session_id in self._native_session_tombstones:
                raise HTTPException(409, "OpenCode session is already closed")
            state = self._native_sessions.get(session_id)
            if state is not None:
                if state.seed != body:
                    raise HTTPException(409, "OpenCode session ID is bound to different seed inputs")
                if state.closing:
                    raise HTTPException(409, "OpenCode session is closing")
            else:
                state = await self._initialize_agent_session_state(session_id, body)
                self._native_sessions[session_id] = state
                self._native_session_expiry_tasks[session_id] = asyncio.create_task(
                    self._expire_native_session(session_id, body.episode_id)
                )
            request.session[_NATIVE_SESSION_KEY] = session_id
            return AgentSeedSessionResponse(agent_session_id=session_id)

    async def _expire_native_session(self, session_id: str, episode_id: EpisodeId) -> None:
        try:
            await asyncio.sleep(self.config.session_lifetime_seconds)
            lock = self._native_session_locks.setdefault(session_id, asyncio.Lock())
            async with lock:
                await self._close_native_session(session_id, episode_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Keep failed cleanup state and its closing marker so activation cannot resume.
            LOG.exception("Could not clean up expired OpenCode session %s", session_id)
        finally:
            if self._native_session_expiry_tasks.get(session_id) is asyncio.current_task():
                self._native_session_expiry_tasks.pop(session_id, None)

    async def _initialize_agent_session_state(
        self, session_id: str, body: AgentSeedSessionRequest
    ) -> OpenCodeSandboxSession:
        if self.config.model_server is None:
            raise HTTPException(422, "Native OpenCode requires model_server")
        if self.config.num_workers not in (None, 1):
            raise HTTPException(422, "Native OpenCode sessions require num_workers=1")
        if body.sandbox_access is None or not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
            raise HTTPException(422, "Native OpenCode requires Resources-owned direct SandboxAccess")
        workdir = body.sandbox_access.workdir
        normalized_workdir = PurePosixPath(workdir)
        if (
            not normalized_workdir.is_absolute()
            or "\x00" in workdir
            or ".." in normalized_workdir.parts
            or normalized_workdir in (PurePosixPath("/"), PurePosixPath("/tmp"))
            or str(normalized_workdir).startswith("/tmp/nemo-gym-opencode")
        ):
            raise HTTPException(
                422, "Native OpenCode workdir must be absolute and separate from adapter runtime/session storage"
            )
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise HTTPException(422, "Native OpenCode uses its own tools; required HTTP/MCP tools are unsupported")
        if not isinstance(self.config.opencode_version, str) or not re.fullmatch(
            r"\d+\.\d+\.\d+", self.config.opencode_version
        ):
            raise HTTPException(422, "Native OpenCode requires an exact opencode_version")
        if self.config.context_window <= 0 or self.config.timeout <= 0 or self.config.setup_timeout <= 0:
            raise HTTPException(422, "OpenCode context window and execution timeout must be positive")
        unsupported = self.config.opencode_config.keys() - {"permission", "tools"}
        if unsupported:
            raise HTTPException(
                422, f"Native OpenCode config supports permission/tools only; unsupported: {sorted(unsupported)}"
            )
        if self.config.remote_opencode_install_script_path and not self.config.remote_opencode_binary_path:
            raise HTTPException(422, "A staged OpenCode installer requires remote_opencode_binary_path")
        if self.config.remote_opencode_musl_binary_path and not self.config.remote_opencode_install_script_path:
            raise HTTPException(422, "A staged musl binary requires a compatible staged installer")
        connection = body.sandbox_access.connection
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        # Caller-assigned IDs are wire identifiers, never filesystem paths.
        directory = f"/tmp/nemo-gym-opencode-sessions/{uuid4().hex}"
        runtime = f"/tmp/nemo-gym-opencode-runtime-{self.config.opencode_version}"
        prepared_directory = False
        try:
            # Resolve inside the sandbox: host-side lexical checks cannot detect task symlinks.
            validate_paths = (
                "from pathlib import Path; import sys; "
                "workdir,*roots=[Path(p).resolve() for p in sys.argv[1:]]; "
                "assert workdir.is_dir(), 'OpenCode task workdir is missing'; "
                "assert all(workdir != root and workdir not in root.parents and root not in workdir.parents "
                "for root in roots), 'OpenCode runtime/session storage overlaps the task workdir'"
            )
            command = (
                f"python3 -I -c {quote(validate_paths)} {quote(workdir)} "
                f"{quote(str(PurePosixPath(directory).parent))} {quote(runtime)} && mkdir -p {quote(directory)}"
            )
            result = await sandbox.exec(command, timeout_s=30)
            self._check_native_setup(command, result)
            prepared_directory = True
            installer = "install_opencode_runtime.sh"
            await sandbox.upload(Path(__file__).with_name(installer), f"{directory}/{installer}")
            command = "bash " + " ".join(
                quote(arg)
                for arg in (
                    f"{directory}/{installer}",
                    runtime,
                    self.config.opencode_version,
                    self.config.remote_opencode_binary_path or "",
                    self.config.remote_opencode_install_script_path or "",
                    self.config.remote_opencode_musl_binary_path or "",
                )
            )
            result = await sandbox.exec(command, cwd=workdir, timeout_s=self.config.setup_timeout)
            self._check_native_setup(command, result)
            await sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), f"{directory}/sandbox_runner.py")
        except BaseException:
            try:
                if prepared_directory:
                    await sandbox.exec(f"rm -rf -- {quote(directory)}", timeout_s=30)
            finally:
                await sandbox.disconnect()
            raise
        return OpenCodeSandboxSession(body, sandbox, directory, runtime)

    @staticmethod
    def _check_native_setup(command: str, result: SandboxExecResult) -> None:
        if result.return_code != 0 or result.error_type is not None:
            raise RuntimeError(
                f"OpenCode setup failed: command={command!r}, exit={result.return_code}, "
                f"error={result.error_type}; stderr={result.stderr}; stdout={result.stdout}"
            )

    def _expire_native_receipts(self) -> None:
        now = monotonic()
        while self._closed_native_sessions and next(iter(self._closed_native_sessions.values()))[2] <= now:
            self._closed_native_sessions.popitem(last=False)
        while self._native_session_tombstones and next(iter(self._native_session_tombstones.values()))[1] <= now:
            session_id, _ = self._native_session_tombstones.popitem(last=False)
            if session_id not in self._native_sessions:
                self._native_session_locks.pop(session_id, None)

    async def close_agent_session(self, request: Request, body: AgentCloseSessionRequest) -> AgentCloseSessionResponse:
        """Close by caller identity even when a lost seed response omitted the cookie."""
        self._expire_native_receipts()
        session_id = body.agent_session_id
        marker = self._native_session_marker(request)
        if marker is not None and marker != session_id:
            raise HTTPException(409, "OpenCode close cookie does not match the requested session")
        lock = self._native_session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            result = await self._close_native_session(session_id, body.episode_id)
            # Keep a tombstone cookie so this client cannot enter local or legacy execution.
            request.session[_NATIVE_SESSION_KEY] = session_id
            return result

    async def _close_native_session(self, session_id: str, episode_id: EpisodeId) -> AgentCloseSessionResponse:
        receipt = self._closed_native_sessions.get(session_id)
        if receipt is not None:
            if episode_id != receipt[0]:
                raise HTTPException(409, "OpenCode close does not match the seeded episode")
            return receipt[1]
        tombstone = self._native_session_tombstones.get(session_id)
        if tombstone is not None:
            raise HTTPException(409, "OpenCode close receipt expired")
        state = self._native_sessions.get(session_id)
        observations = None
        if state is not None:
            if state.seed.episode_id != episode_id:
                raise HTTPException(409, "OpenCode close does not match the seeded episode")
            await state.close(self.config.session_close_timeout_seconds)
            observations = state.observations or AgentObservationBundle(
                source="opencode", gaps=[ObservationGap(code="agent_activation_interrupted")]
            )
        result = AgentCloseSessionResponse(agent_session_id=session_id, agent_observations=observations)
        self._native_sessions.pop(session_id, None)
        expiry_task = self._native_session_expiry_tasks.pop(session_id, None)
        if expiry_task is not None and expiry_task is not asyncio.current_task():
            expiry_task.cancel()
        now = monotonic()
        self._closed_native_sessions[session_id] = (
            episode_id,
            result,
            now + self.config.session_close_retry_window_seconds,
        )
        # A close that arrives before seed must block the delayed seed through its lifetime.
        self._native_session_tombstones[session_id] = (
            episode_id,
            now + self.config.session_lifetime_seconds + self.config.session_close_retry_window_seconds,
        )
        return result

    def _native_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        unsupported = (
            "max_output_tokens",
            "temperature",
            "top_p",
            "reasoning",
            "max_tool_calls",
            "previous_response_id",
            "prompt",
            "text",
            "context_management",
            "conversation",
            "moderation",
            "top_logprobs",
            "truncation",
            "include",
            "store",
            "service_tier",
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
            "stream_options",
            "user",
        )
        values = body.model_dump(mode="json")
        for key in unsupported:
            if values.get(key) is not None:
                raise HTTPException(
                    422, f"Native OpenCode does not support request field {key}; configure Gym model limits/sampling"
                )
        if body.tools or body.tool_choice != "auto" or not body.parallel_tool_calls or body.background:
            raise HTTPException(422, "OpenCode owns tool selection and execution policy")
        if (body.metadata or {}).get("chat_template_kwargs") is not None:
            raise HTTPException(422, "Configure chat_template_kwargs on the Gym model server")
        if body.model not in (None, "", "dummy_model", self.config.model_server.name):
            raise HTTPException(422, "Native OpenCode uses the configured Gym model_server")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(
                422, "Native OpenCode accepts one user text prompt and optional system/developer message"
            )
        texts = []
        for item in items:
            if isinstance(item.content, str):
                texts.append(item.content)
            else:
                parts = [part if isinstance(part, dict) else part.model_dump() for part in item.content]
                if any(part.get("type") != "input_text" for part in parts):
                    raise HTTPException(422, "Native OpenCode supports text input only")
                texts.append("\n".join(part["text"] for part in parts))
        if not texts[-1].strip():
            raise HTTPException(422, "Native OpenCode requires a nonempty user prompt")
        return texts[-1], "\n\n".join(
            text for text in [self.config.system_prompt, body.instructions, *texts[:-1]] if text
        )

    def _native_output(
        self, export: dict[str, Any], observations: AgentObservationBundle
    ) -> list[NeMoGymResponseOutputItem]:
        output = []
        for message in export.get("messages", []):
            if message.get("info", {}).get("role") != "assistant":
                continue
            for part in message.get("parts", []):
                if part.get("type") in (
                    "step-start",
                    "step-finish",
                    "patch",
                    "snapshot",
                    "compaction",
                    "agent",
                    "retry",
                ):
                    continue
                try:
                    items = parse_opencode_export({"messages": [{"info": message["info"], "parts": [part]}]})
                    if part.get("type") == "tool":
                        state = part.get("state") or {}
                        for item in items:
                            item.status = "completed" if state.get("status") == "completed" else "incomplete"
                            if isinstance(item, NeMoGymFunctionCallOutput):
                                item.output = state.get("output") or state.get("error") or ""
                    output.extend(items)
                except Exception:
                    # A malformed or newer artifact must not discard already captured turns.
                    observations.gaps.append(
                        ObservationGap(code="agent_artifact_record_unparseable", detail=str(part.get("type")))
                    )
        return output

    @staticmethod
    def _native_usage(export: dict[str, Any]) -> NeMoGymResponseUsage | None:
        # OpenCode 1.17.11 getUsage subtracts cache from input and reasoning from output.
        # Restore inclusive OpenAI counters across the root and subagent model turns.
        infos = export.get("usage_messages", [message["info"] for message in export.get("messages", [])])
        usages = []
        missing_usage = False
        for info in infos:
            if info.get("role") != "assistant":
                continue
            tokens = info.get("tokens")
            if not isinstance(tokens, dict) or not tokens:
                missing_usage = True
                continue
            cache = tokens.get("cache")
            cache = cache if isinstance(cache, dict) else {}

            def count(value: object) -> int:
                return value if type(value) is int and value >= 0 else 0

            cache_read = count(cache.get("read"))
            reasoning = count(tokens.get("reasoning"))
            input_tokens = int(tokens.get("input") or 0) + cache_read + count(cache.get("write"))
            output_tokens = int(tokens.get("output") or 0) + reasoning
            # Pinned 1.17.11 Session.getUsage defaults absent/invalid optional counters
            # to zero before persistence. Only positive values establish measured details;
            # a stored zero cannot prove a backend-reported zero.
            usages.append(
                NeMoGymResponseUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cache_read or None),
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning or None),
                )
            )
        total = NeMoGymResponseUsage.sum_from_list(usages) if usages else None
        if total is not None and missing_usage:
            total.input_tokens_details.cached_tokens = None
            total.output_tokens_details.reasoning_tokens = None
        return total

    async def _native_response(
        self, state: OpenCodeSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, *, prompt: str, system: str
    ) -> NeMoGymResponse:
        base_url = self.resolve_model_base_url(self.config.model_server.name, state.seed.episode_id.capture_key)
        config = {
            "model": "nemo_gym/dummy_model",
            "small_model": "nemo_gym/dummy_model",
            "enabled_providers": ["nemo_gym"],
            "autoupdate": False,
            "share": "disabled",
            "provider": {
                "nemo_gym": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": "dummy_key",
                        "timeout": False,
                    },  # pragma: allowlist secret
                    "models": {
                        "dummy_model": {
                            "limit": {
                                "context": self.config.context_window,
                                "input": self.config.context_window,
                                "output": self.config.context_window,
                            }
                        }
                    },
                }
            },
            **self.config.opencode_config,
        }
        # System instructions are separate from the user's task and applied to every OpenCode model turn.
        if system:
            # OpenCode reads instruction files as text; JSON quoting must not become prompt content.
            with tempfile.TemporaryDirectory(prefix="opencode-instructions-") as directory:
                path = Path(directory) / "instructions.md"
                path.write_text(system)
                await state.sandbox.upload(path, f"{state.directory}/instructions.md")
            config["instructions"] = [f"{state.directory}/instructions.md"]
        payload = {
            "directory": state.directory,
            "cwd": state.seed.sandbox_access.workdir,
            "command": [f"{state.runtime}/opencode", "run", "--format", "json", "--thinking", "--title", "NeMo Gym"],
            "prompt": prompt,
            "env": {
                "HOME": f"{state.directory}/home",
                "XDG_DATA_HOME": f"{state.directory}/data",
                "XDG_CONFIG_HOME": f"{state.directory}/config",
                "XDG_CACHE_HOME": f"{state.directory}/cache",
                "XDG_STATE_HOME": f"{state.directory}/state",
                "OPENCODE_CONFIG_CONTENT": json.dumps(config),
                "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
                "OPENCODE_DISABLE_AUTOUPDATE": "true",
                "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "1000000000",
            },
            "timeout": self.config.timeout,
            "cleanup_timeout": self.config.session_close_timeout_seconds / 3,
        }
        error = None
        export = {}
        cancelled = False
        try:
            export = json.loads(
                await state.execute(
                    payload,
                    timeout=self.config.timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
            )
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            error = str(exc)
            try:
                export = json.loads(await state.read_text("export.json"))
            except Exception:
                pass
        finally:
            try:
                with tempfile.TemporaryDirectory(prefix="opencode-observations-") as directory:
                    path = Path(directory) / "observations.db"
                    await state.sandbox.download(f"{state.directory}/observations.db", path)
                    state.observations = parse_opencode_observations(
                        path, state.seed.episode_id.capture_key, require_terminal_finish=True
                    )
            except Exception:
                state.observations = AgentObservationBundle(
                    source="opencode",
                    gaps=[
                        ObservationGap(code="agent_artifact_unavailable"),
                        ObservationGap(code="observation_capture_failed"),
                    ],
                )
            if cancelled:
                for record in state.observations.records:
                    if isinstance(record, AgentInvocation) and record.parent_invocation_id is None:
                        record.status = "incomplete"
                        record.error_type = "cancelled"
        output = []
        usage = None
        if export.get("messages"):
            try:
                output = self._native_output(export, state.observations)
                usage = self._native_usage(export)
                if (
                    usage is None
                    or usage.input_tokens_details.cached_tokens is None
                    or usage.output_tokens_details.reasoning_tokens is None
                ):
                    state.observations.gaps.append(
                        ObservationGap(
                            code="token_usage_detail_unavailable",
                            detail="Persisted optional counters are absent, invalid, or runtime-defaulted zeros",
                        )
                    )
            except Exception as exc:
                error = error or f"OpenCode output parse failed: {exc}"
        result = state.result
        if result is not None:
            error = error or result.error
            if result.return_code != 0 and not result.timed_out:
                error = error or f"OpenCode exited with code {result.return_code}"
        assistants = [
            message["info"] for message in export.get("messages", []) if message["info"].get("role") == "assistant"
        ]
        for info in assistants:
            if info.get("error"):
                error = error or json.dumps(info["error"])
        if not assistants:
            error = error or "OpenCode produced no assistant result"
        incomplete = result is not None and result.timed_out
        if assistants and not incomplete:
            last = assistants[-1]
            finish = last.get("finish")
            if finish in {"length", "content-filter", "tool-calls"}:
                incomplete = True
            elif finish != "stop" or not last.get("time", {}).get("completed"):
                error = error or f"OpenCode ended without a successful terminal assistant result (finish={finish!r})"
        status = "failed" if error else "incomplete" if incomplete else "completed"
        # Artifact message completion records a model turn, not the entire invocation.
        for record in state.observations.records:
            if isinstance(record, AgentInvocation) and record.parent_invocation_id is None:
                record.status = status
                record.error_type = "server_error" if error else "timeout" if result and result.timed_out else None
        if result is not None:
            state.observations.records.append(
                SandboxObservation(
                    role="agent",
                    provider=state.seed.sandbox_access.connection.provider_config_ref,
                    sandbox_id=str(state.seed.sandbox_access.connection.descriptor.get("sandbox_id", "unknown")),
                    outcome="timeout" if result.timed_out else "failed" if error else "completed",
                    exit_code=result.return_code,
                )
            )
        state.observations.gaps.append(
            ObservationGap(
                code="model_usage_reconciliation_unavailable",
                detail="Usage comes from OpenCode persisted assistant messages; compare against captured Gym model calls.",
            )
        )
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model_server.name,
            object="response",
            output=output,
            usage=usage,
            status=status,
            error={"code": "server_error", "message": error} if error else None,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            metadata={
                "harness_execution": "sandbox",
                "opencode_version": self.config.opencode_version,
                "harness_hostname": result.hostname if result else "unknown",
                "harness_pid": str(result.pid) if result else "unknown",
            },
        )


if __name__ == "__main__":
    OpenCodeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeAgent.run_webserver()  # noqa: F401
