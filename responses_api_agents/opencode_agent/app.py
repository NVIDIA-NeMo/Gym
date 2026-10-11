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
from asyncio import Semaphore
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from shlex import quote
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agent_utils.sandbox_session import SandboxSession
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
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
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
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxSpec, create_provider
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
    parse_opencode_session,
)
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_agent.runtime import (
    OBSERVABILITY_PATCH,
    OPENCODE_VERSION,
    apply_observability_patch,
)
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession, format_sandbox_error
from responses_api_agents.opencode_agent.setup_opencode import ensure_opencode


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"
_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"


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


def _sandbox_prepare_command(workdir: str, directory: str, runtime: str) -> str:
    # Bootstrap with POSIX sh; the runtime installer itself requires Bash.
    bootstrap = """set -eu
set --
command -v python3 >/dev/null 2>&1 || set -- "$@" python3
command -v bash >/dev/null 2>&1 || set -- "$@" bash
if [ "$#" -gt 0 ]; then
    [ "$(id -u)" = 0 ] || { echo "OpenCode sandbox execution requires $*: preinstall these tools or use a root image." >&2; exit 1; }
    if command -v apk >/dev/null 2>&1; then
        apk add --no-cache "$@"
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
    else
        echo "OpenCode sandbox execution requires $*: preinstall these tools (automatic installation requires apt-get or apk)." >&2
        exit 1
    fi
fi
"""
    validate_paths = (
        "from pathlib import Path; import sys; "
        "workdir,*roots=[Path(p).resolve() for p in sys.argv[1:]]; "
        "assert workdir.is_dir(), 'OpenCode task workdir is missing'; "
        "assert all(workdir != root and workdir not in root.parents and root not in workdir.parents "
        "for root in roots), 'OpenCode runtime/session storage overlaps the task workdir'"
    )
    return (
        bootstrap + f"python3 -I -c {quote(validate_paths)} {quote(workdir)} "
        f"{quote(str(PurePosixPath(directory).parent))} {quote(runtime)} && mkdir -p {quote(directory)}"
    )


class OpenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = None
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
    timeout: int = 900
    extra_args: list[str] = []
    opencode_config: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 262144
    max_output_tokens: int = 131072
    opencode_version: str = OPENCODE_VERSION

    # Used only when Resources does not supply a borrowed sandbox.
    sandbox_provider: str | None = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_install_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

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
    """Run OpenCode locally or in a borrowed or agent-owned sandbox session."""

    ray_enabled = False

    config: OpenCodeAgentConfig
    _local_setup_task: asyncio.Task[None] | None = PrivateAttr(default=None)
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.sem = Semaphore(self.config.concurrency)

    async def _ensure_local_runtime(self) -> None:
        if self._local_setup_task is None:
            self._local_setup_task = asyncio.create_task(
                asyncio.to_thread(ensure_opencode, self.config.opencode_version)
            )
            self._local_setup_task.add_done_callback(self._clear_failed_local_setup)
        setup = self._local_setup_task
        try:
            await asyncio.shield(setup)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._local_setup_task is setup:
                self._local_setup_task = None
            raise

    def _clear_failed_local_setup(self, task: asyncio.Task[None]) -> None:
        # All shielded waiters may have gone away before the installer fails.
        failed = task.cancelled() or task.exception() is not None
        if failed and self._local_setup_task is task:
            self._local_setup_task = None

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenCodeAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _workspace_root(self) -> Path:
        """Create a fresh per-rollout workspace directory and return it.

        The full uuid plus exist_ok=False: a name collision must fail this
        rollout loudly rather than silently merge two live rollouts' trees.
        """
        root = Path(self.config.workspace_root).expanduser() / f"opencode_{uuid4().hex}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=False)
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
                    # Gym model servers emit and accept `reasoning_content`; OpenCode replays assistant
                    # history under this field, and Gym rejects an unknown `reasoning` key with a 422.
                    "interleaved": {"field": "reasoning_content"},
                    "limit": {"context": self.config.context_window, "output": self.config.max_output_tokens},
                },
            )
            nemo["models"] = {self.config.model: model}
        if self._model_call_capture_enabled():
            apply_observability_patch(config)
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
        await self._ensure_local_runtime()
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
                    observations = _parse_opencode_session(
                        db_path, invocation_id, trajectory, model_ref=self.config.model_server
                    )
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
                output_tokens_details=NeMoGymResponseOutputTokensDetails(
                    reasoning_tokens=usage.get("reasoning_tokens", 0)
                ),
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
        session_id = self._agent_session_id_from_request(request)
        if session_id is not None:
            state = self._require_agent_session(session_id)
            assert isinstance(state, OpenCodeSandboxSession)
            if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
                raise HTTPException(409, "OpenCode activation does not match the seeded session and rollout")
            prompt, system = self._session_input(body)
            if state.task is None:
                state.activation_request = body.model_copy(deep=True)
                state.task = asyncio.create_task(
                    self._session_response(state, state.activation_request, prompt=prompt, system=system)
                )
            elif body != state.activation_request:
                raise HTTPException(409, "OpenCode sandbox sessions support one activation; retry the same request")
            return (await asyncio.shield(state.task)).model_copy(deep=True)
        # Only genuinely unseeded calls reach local execution; invalid or closed
        # session markers are rejected above, never retried on the host.
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
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "OpenCode sandbox sessions must use EnvironmentServer /run")
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

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> OpenCodeSandboxSession:
        """Prepare only the harness runtime; Resources owns borrowed task setup."""
        if self.config.model_server is None:
            raise HTTPException(422, "OpenCode sandbox execution requires model_server")
        owns_sandbox = body.sandbox_access is None
        if owns_sandbox:
            if not self.config.sandbox_provider:
                raise HTTPException(422, "OpenCode requires sandbox_access or a configured sandbox_provider")
            spec = SandboxSpec(**{"workdir": "/app", **self.config.sandbox_config})
            workdir = spec.workdir
            provider_ref = self.config.sandbox_provider
        else:
            if not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
                raise HTTPException(422, "OpenCode requires direct SandboxAccess")
            workdir = body.sandbox_access.workdir
            provider_ref = body.sandbox_access.connection.provider_config_ref
        if not isinstance(workdir, str):
            raise HTTPException(422, "OpenCode sandbox workdir must be absolute")
        normalized_workdir = PurePosixPath(workdir)
        if (
            not normalized_workdir.is_absolute()
            or "\x00" in workdir
            or ".." in normalized_workdir.parts
            or normalized_workdir in (PurePosixPath("/"), PurePosixPath("/tmp"))
            or str(normalized_workdir).startswith("/tmp/nemo-gym-opencode")
        ):
            raise HTTPException(
                422, "OpenCode sandbox workdir must be absolute and separate from adapter runtime/session storage"
            )
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise HTTPException(
                422, "OpenCode sandbox execution uses its own tools; required HTTP/MCP tools are unsupported"
            )
        if not isinstance(self.config.opencode_version, str) or not re.fullmatch(
            r"\d+\.\d+\.\d+", self.config.opencode_version
        ):
            raise HTTPException(422, "OpenCode sandbox execution requires an exact opencode_version")
        if self.config.context_window <= 0 or self.config.timeout <= 0:
            raise HTTPException(422, "OpenCode context window and execution timeout must be positive")
        if not 0 < self.config.max_output_tokens <= self.config.context_window:
            raise HTTPException(
                422, "OpenCode sandbox max_output_tokens must be positive and not exceed context_window"
            )
        unsupported = self.config.opencode_config.keys() - {"permission", "tools"}
        if unsupported:
            raise HTTPException(
                422, f"OpenCode sandbox config supports permission/tools only; unsupported: {sorted(unsupported)}"
            )
        if self.config.env or self.config.extra_args or self.config.command != "opencode":
            raise HTTPException(422, "env, extra_args and command overrides are supported only by local OpenCode")
        provider = create_provider(resolve_provider_config(provider_ref, get_global_config_dict()))
        if owns_sandbox:
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(body.sandbox_access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        # Caller-assigned IDs are wire identifiers, never filesystem paths.
        directory = f"/tmp/nemo-gym-opencode-sessions/{uuid4().hex}"
        runtime = f"/tmp/nemo-gym-opencode-runtime-{self.config.opencode_version}"
        state = OpenCodeSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox, session_dir=directory, workdir=workdir, harness="OpenCode", owns_sandbox=owns_sandbox
            ),
            runtime=runtime,
            model_ref=self.config.model_server,
        )
        prepared_directory = False
        try:
            if owns_sandbox:
                await sandbox.start(spec)
                # Providers need not create SandboxSpec.workdir. Never prepare a borrowed task here.
                workspace = await sandbox.exec(f"mkdir -p -- {shlex.quote(workdir)}", cwd="/", timeout_s=30)
                if workspace.return_code != 0 or workspace.error_type:
                    raise RuntimeError(
                        f"Cannot create OpenCode sandbox workdir {workdir}: {workspace.stderr or workspace.stdout}"
                    )
            command = _sandbox_prepare_command(workdir, directory, runtime)
            result = await sandbox.exec(command, timeout_s=self.config.sandbox_install_timeout_seconds)
            self._check_sandbox_setup(command, result)
            prepared_directory = True
            installer = "install_opencode_runtime.sh"
            await sandbox.upload(Path(__file__).with_name(installer), f"{directory}/{installer}")
            command = "bash " + " ".join(
                quote(arg)
                for arg in (
                    f"{directory}/{installer}",
                    runtime,
                    self.config.opencode_version,
                )
            )
            result = await sandbox.exec(command, cwd=workdir, timeout_s=self.config.sandbox_install_timeout_seconds)
            self._check_sandbox_setup(command, result)
            await sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), f"{directory}/sandbox_runner.py")
            if self._model_call_capture_enabled():
                await sandbox.upload(OBSERVABILITY_PATCH, f"{directory}/{OBSERVABILITY_PATCH.name}")
        except BaseException as error:
            try:
                if prepared_directory or owns_sandbox:
                    await state.close(self.config.session_close_timeout_seconds)
                else:
                    await sandbox.disconnect()
            except BaseException:
                LOG.exception("OpenCode seed cleanup failed; retaining session %s", body.agent_session_id)
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    @staticmethod
    def _check_sandbox_setup(command: str, result: SandboxExecResult) -> None:
        if result.return_code != 0 or result.error_type is not None:
            raise RuntimeError(
                f"OpenCode setup failed: command={command!r}, exit={result.return_code}, "
                f"error={result.error_type}; stderr={result.stderr}; stdout={result.stdout}"
            )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Release only after capture and positive cleanup; retain failed closes for retry."""
        assert isinstance(state, OpenCodeSandboxSession)
        await state.close(self.config.session_close_timeout_seconds)
        observations = state.observations or AgentObservationBundle(
            source="opencode", gaps=[ObservationGap(code="agent_activation_interrupted")]
        )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, agent_observations=observations
        )

    def _session_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        supported = {"input", "instructions", "model"}
        for key, field in type(body).model_fields.items():
            if key in supported:
                continue
            value = getattr(body, key)
            if key in ("stream", "background") and value is False:
                continue
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(
                    422,
                    f"OpenCode sandbox execution does not support request field {key}; configure Gym model limits/sampling",
                )
        if body.model not in (None, "", "dummy_model", self.config.model_server.name):
            raise HTTPException(422, "OpenCode sandbox execution uses the configured Gym model_server")
        items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        roles = [getattr(item, "role", None) for item in items]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(
                422, "OpenCode sandbox execution accepts one user text prompt and optional system/developer message"
            )
        texts = []
        for item in items:
            if isinstance(item.content, str):
                texts.append(item.content)
            else:
                parts = [part if isinstance(part, dict) else part.model_dump() for part in item.content]
                if any(part.get("type") != "input_text" for part in parts):
                    raise HTTPException(422, "OpenCode sandbox execution supports text input only")
                texts.append("\n".join(part["text"] for part in parts))
        if not texts[-1].strip():
            raise HTTPException(422, "OpenCode sandbox execution requires a nonempty user prompt")
        return texts[-1], "\n\n".join(
            text for text in [self.config.system_prompt, body.instructions, *texts[:-1]] if text
        )

    def _session_output(
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
    def _session_usage(export: dict[str, Any]) -> NeMoGymResponseUsage | None:
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

    async def _session_response(
        self, state: OpenCodeSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, *, prompt: str, system: str
    ) -> NeMoGymResponse:
        base_url = self.resolve_model_base_url(self.config.model_server.name, state.request.episode_id.capture_key)
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
                            "interleaved": {"field": "reasoning_content"},
                            "limit": {
                                "context": self.config.context_window,
                                "input": self.config.context_window,
                                "output": self.config.max_output_tokens,
                            },
                        }
                    },
                }
            },
            **self.config.opencode_config,
        }
        # System instructions are separate from the user's task and applied to every OpenCode model turn.
        if system:
            config["instructions"] = [f"{state.session.session_dir}/instructions.md"]
        if self._model_call_capture_enabled():
            apply_observability_patch(config, plugin_path=Path(state.session.session_dir) / OBSERVABILITY_PATCH.name)
        command = [f"{state.runtime}/opencode", "run", "--format", "json", "--title", "NeMo Gym"]
        if self.config.thinking:
            command.append("--thinking")
        payload = {
            "instructions": system,
            "directory": state.session.session_dir,
            "cwd": state.session.workdir,
            "command": command,
            "prompt": prompt,
            "env": {
                "HOME": f"{state.session.session_dir}/home",
                "XDG_DATA_HOME": f"{state.session.session_dir}/data",
                "XDG_CONFIG_HOME": f"{state.session.session_dir}/config",
                "XDG_CACHE_HOME": f"{state.session.session_dir}/cache",
                "XDG_STATE_HOME": f"{state.session.session_dir}/state",
                "OPENCODE_CONFIG_CONTENT": json.dumps(config),
                "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
                "OPENCODE_DISABLE_AUTOUPDATE": "true",
                "OPENCODE_DISABLE_MODELS_FETCH": "true",
                "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": str(self.config.max_output_tokens),
            },
        }
        error = None
        export = {}
        failure = None
        try:
            async with self.sem:
                await state.execute(
                    payload,
                    timeout=self.config.timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as exc:
            failure = exc
        if state.observations is None:
            state.observations = AgentObservationBundle(
                source="opencode", gaps=[ObservationGap(code="observation_capture_failed")]
            )
        if state.session.artifacts is not None:
            try:
                parsed = json.loads(state.session.artifacts)
                if not isinstance(parsed, dict) or not isinstance(parsed.get("messages", []), list):
                    raise ValueError("expected a session object with a messages list")
                if any(
                    not isinstance(message, dict) or not isinstance(message.get("info"), dict)
                    for message in parsed.get("messages", [])
                ):
                    raise ValueError("expected message info objects")
                export = parsed
            except (ValueError, TypeError) as exc:
                error = f"OpenCode output parse failed: {exc}"
        output = []
        usage = None
        if export.get("messages"):
            try:
                output = self._session_output(export, state.observations)
                usage = self._session_usage(export)
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
        result = state.session.cleanup
        runtime = state.runtime_info
        if runtime is None:
            state.observations.gaps.append(ObservationGap(code="runtime_info_unavailable"))
        if result is not None and result["return_code"] is None:
            state.observations.gaps.append(ObservationGap(code="worker_exit_code_unavailable"))
        if result is not None:
            error = error or result["error"]
            if result["return_code"] not in (0, None) and not result["timed_out"]:
                error = error or f"OpenCode exited with code {result['return_code']}"
        assistants = [
            message["info"] for message in export.get("messages", []) if message["info"].get("role") == "assistant"
        ]
        for info in assistants:
            if info.get("error"):
                error = error or json.dumps(info["error"])
        if not assistants:
            error = error or "OpenCode produced no assistant result"
        incomplete = result is not None and result["timed_out"]
        if assistants and not incomplete:
            last = assistants[-1]
            finish = last.get("finish")
            if finish in {"length", "content-filter", "tool-calls"}:
                incomplete = True
            elif finish != "stop" or not last.get("time", {}).get("completed"):
                error = error or f"OpenCode ended without a successful terminal assistant result (finish={finish!r})"
        status = "failed" if error or failure else "incomplete" if incomplete else "completed"
        # Artifact message completion records a model turn, not the entire invocation.
        for record in state.observations.records:
            if isinstance(record, AgentInvocation) and record.parent_invocation_id is None:
                record.status = "incomplete" if isinstance(failure, asyncio.CancelledError) else status
                record.error_type = (
                    "cancelled"
                    if isinstance(failure, asyncio.CancelledError)
                    else "server_error"
                    if error or failure
                    else "timeout"
                    if result and result["timed_out"]
                    else None
                )
        if result is not None:
            state.observations.records.append(
                SandboxObservation(
                    role="agent",
                    provider=(
                        self.config.sandbox_provider
                        if state.session.owns_sandbox
                        else state.request.sandbox_access.connection.provider_config_ref
                    ),
                    sandbox_id=(
                        None
                        if state.session.owns_sandbox
                        else str(state.request.sandbox_access.connection.descriptor.get("sandbox_id", "unknown"))
                    ),
                    outcome="timeout" if result["timed_out"] else "failed" if error or failure else "completed",
                    exit_code=result["return_code"],
                )
            )
        state.observations.gaps.append(
            ObservationGap(
                code="model_usage_reconciliation_unavailable",
                detail="Usage comes from OpenCode persisted assistant messages; compare against captured Gym model calls.",
            )
        )
        if failure is not None:
            raise failure
        if error:
            raise HTTPException(502, format_sandbox_error(error, stderr=state.stderr))
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model_server.name,
            object="response",
            output=output,
            usage=usage,
            status=status,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            metadata={
                "harness_execution": "sandbox",
                "opencode_version": self.config.opencode_version,
                **({"harness_hostname": runtime.hostname, "harness_pid": str(runtime.pid)} if runtime else {}),
            },
        )


if __name__ == "__main__":
    OpenCodeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeAgent.run_webserver()  # noqa: F401
