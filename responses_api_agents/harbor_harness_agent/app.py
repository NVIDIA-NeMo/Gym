# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run any Harbor agent in a task sandbox that a Resources Server owns.

The Environment Server seeds a session with the Resources Server's ``SandboxAccess``. Seeding wraps that sandbox as a
Harbor environment, creates the configured Harbor agent, and runs its ``setup`` (which installs installed-agent
harnesses inside the sandbox). One ``/v1/responses`` activation runs the agent on the task instruction and returns its
ATIF trajectory as a Gym response. Closing the session disconnects from the sandbox without stopping it; the
Resources Server verifies and stops it afterwards.
"""

import asyncio
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, HTTPException, Request
from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trajectories import Trajectory
from harbor.models.trial.config import AgentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from pydantic import ConfigDict, Field, PositiveFloat, field_validator

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.adapters.harbor import (
    DEFAULT_EXEC_TIMEOUT_SECONDS,
    HARBOR_AGENT_TIMEOUT_METADATA_KEY,
    HARBOR_AGENT_USER_METADATA_KEY,
    HarborSandboxEnvironment,
)
from nemo_gym.sandbox.config import resolve_provider_config
from responses_api_agents.harbor_harness_agent.atif import convert_atif_to_gym_responses


LOG = logging.getLogger(__name__)


class HarborHarnessAgentConfig(BaseResponsesAPIAgentConfig):
    harbor_agent: AgentConfig = Field(
        description="Harbor trial AgentConfig: `name` or `import_path`, `model_name`, `kwargs`, `env`, timeouts."
    )
    model_server: Optional[ModelServerRef] = Field(
        default=None,
        description="Gym model server for the agent's model calls. Unset, the agent calls its own provider directly.",
    )
    model_base_url_kwarg: Optional[str] = Field(
        default="api_base",
        description="Agent constructor kwarg that receives the model server base URL (Terminus-2 uses `api_base`).",
    )
    model_base_url_env: list[str] = Field(
        default_factory=list,
        description="Environment variables set to the model server base URL, for installed agents such as OpenCode.",
    )
    agent_timeout_multiplier: PositiveFloat = 1.0
    agent_setup_timeout_multiplier: PositiveFloat = 1.0
    logs_dir: Path = Field(
        default=Path("results/harbor_harness_agent"),
        description="Host directory for each episode's agent logs and ATIF trajectory, keyed by rollout capture key.",
    )
    exec_shell: Optional[str] = "bash -c"
    default_exec_timeout_seconds: PositiveFloat = DEFAULT_EXEC_TIMEOUT_SECONDS

    @field_validator("logs_dir", mode="after")
    @classmethod
    def resolve_logs_dir(cls, logs_dir: Path) -> Path:
        # Server processes run from their own directory; config paths are relative to the Gym repository.
        logs_dir = logs_dir.expanduser()
        return logs_dir if logs_dir.is_absolute() else PARENT_DIR / logs_dir


@dataclass
class HarborHarnessSession(AgentSessionState):
    sandbox: AsyncSandbox
    environment: HarborSandboxEnvironment
    agent: BaseAgent
    trial_paths: TrialPaths
    activation_request: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    activation: Optional[asyncio.Task] = None
    observations: Optional[AgentObservationBundle] = None
    gaps: list[ObservationGap] = field(default_factory=list)


class HarborHarnessAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: HarborHarnessAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> HarborHarnessSession:
        if body.sandbox_access is None or not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
            raise HTTPException(422, "Harbor agents require direct SandboxAccess from the Resources Server")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise HTTPException(422, "Harbor agents bring their own tools and cannot honor required tool accesses")

        connection = body.sandbox_access.connection
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise

        capture_key = body.episode_id.capture_key
        trial_dir = (self.config.logs_dir / _path_component(capture_key)).resolve()
        shutil.rmtree(trial_dir, ignore_errors=True)
        trial_paths = TrialPaths(trial_dir)
        trial_paths.mkdir()
        # The agent never sees the task definition: Resources has already staged the task into the sandbox.
        environment_dir = trial_dir / "environment"
        environment_dir.mkdir()
        environment = HarborSandboxEnvironment(
            sandbox,
            environment_dir=environment_dir,
            environment_name=body.task_id.task_id,
            session_id=f"{_path_component(capture_key)}__agent",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(workdir=body.sandbox_access.workdir),
            exec_shell=self.config.exec_shell,
            default_exec_timeout_seconds=self.config.default_exec_timeout_seconds,
            logger=LOG,
        )
        agent = AgentFactory.create_agent_from_config(
            self._agent_config(capture_key), logs_dir=trial_paths.agent_dir, logger=LOG
        )
        agent.session_id = f"{_path_component(capture_key)}__agent"
        state = HarborHarnessSession(
            request=body, sandbox=sandbox, environment=environment, agent=agent, trial_paths=trial_paths
        )
        try:
            # AGENTS.md: sandboxed harnesses install their runtime during session setup, before activation.
            async with asyncio.timeout(self._setup_timeout_seconds()):
                with environment.scoped_exec_env(agent.extra_env):
                    await agent.setup(environment)
        except BaseException as error:
            try:
                await sandbox.disconnect()
            except BaseException:
                LOG.exception("Harbor agent seed cleanup failed; retaining session %s", body.agent_session_id)
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Stop the activation and disconnect, leaving the borrowed sandbox to its owner for verification."""
        if not isinstance(state, HarborHarnessSession):
            raise HTTPException(409, "Invalid Harbor agent session state")
        if state.activation is not None and not state.activation.done():
            state.activation.cancel()
            state.gaps.append(ObservationGap(code="agent_activation_interrupted"))
            try:
                await state.activation
            except BaseException:
                pass
        await state.sandbox.disconnect()
        observations = state.observations or AgentObservationBundle(source=self._source(state), gaps=state.gaps)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, agent_observations=observations
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "Harbor agents run only inside an agent session seeded with SandboxAccess")
        state = self._require_agent_session(session_id)
        if not isinstance(state, HarborHarnessSession):
            raise HTTPException(409, "Invalid Harbor agent session state")
        if state.request.episode_id.capture_key != request.path_params.get("rollout_id"):
            raise HTTPException(409, "Harbor activation does not match the seeded session and rollout route")
        if state.activation is None:
            state.activation_request = body.model_copy(deep=True)
            state.activation = asyncio.create_task(self._activate(state, body))
        elif body != state.activation_request:
            raise HTTPException(409, "Harbor sessions support one activation; retry the same request")
        # The activation belongs to the session; a disconnected caller cannot cancel it.
        return (await asyncio.shield(state.activation)).model_copy(deep=True)

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        raise HTTPException(409, "Harbor agents run through an Environment Server session, not /run")

    async def _activate(
        self, state: HarborHarnessSession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        params = body.model_dump(mode="json", exclude_unset=True)
        metadata = dict(params.get("metadata") or {})
        instruction = _instruction(params.get("input"))
        timeout_seconds = self._agent_timeout_seconds(metadata.get(HARBOR_AGENT_TIMEOUT_METADATA_KEY))
        user = metadata.get(HARBOR_AGENT_USER_METADATA_KEY) or None
        context = AgentContext()
        exit_reason = None
        started = time.time()
        try:
            with state.environment.with_default_user(user), state.environment.scoped_exec_env(state.agent.extra_env):
                await asyncio.wait_for(
                    state.agent.run(instruction=instruction, environment=state.environment, context=context),
                    timeout=timeout_seconds,
                )
        except TimeoutError:
            # Harbor still verifies a timed-out agent; return what it did so far.
            exit_reason = f"AgentTimeoutError: agent exceeded {timeout_seconds} seconds"
            state.gaps.append(ObservationGap(code="agent_timeout", detail=exit_reason))
        except NonZeroAgentExitCodeError as error:
            exit_reason = f"NonZeroAgentExitCodeError: {error}"[:2000]
            state.gaps.append(ObservationGap(code="agent_nonzero_exit", detail=exit_reason))

        trajectory_path = await self._collect_trajectory(state, context)
        warnings: list[str] = []
        output: list[dict] = []
        if trajectory_path is None:
            state.gaps.append(ObservationGap(code="agent_transcript_unavailable"))
        else:
            trajectory = Trajectory.model_validate_json(trajectory_path.read_text())
            output = convert_atif_to_gym_responses(trajectory, warnings)
        state.gaps.extend(ObservationGap(code="atif_conversion_lossy", detail=warning) for warning in warnings)
        state.observations = AgentObservationBundle(source=self._source(state), gaps=state.gaps)

        input_tokens, output_tokens = context.n_input_tokens or 0, context.n_output_tokens or 0
        return NeMoGymResponse(
            id=f"harbor-{_path_component(state.request.episode_id.capture_key)}",
            created_at=started,
            model=self.config.harbor_agent.model_name or "unknown",
            object="response",
            output=output,
            parallel_tool_calls=False,
            temperature=params.get("temperature"),
            tool_choice="auto",
            tools=[],
            top_p=params.get("top_p"),
            status="completed" if exit_reason is None else "incomplete",
            usage={
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": context.n_cache_tokens or 0, "cache_write_tokens": 0},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input_tokens + output_tokens,
            },
            metadata={
                "harbor_agent": state.agent.name(),
                "harbor_trial_dir": str(state.trial_paths.trial_dir),
                **({"harbor_agent_exit": exit_reason} if exit_reason else {}),
            },
        )

    async def _collect_trajectory(self, state: HarborHarnessSession, context: AgentContext) -> Optional[Path]:
        """Download the agent's in-sandbox logs and let the agent convert them to ATIF, as a Harbor trial does."""
        try:
            await state.environment.download_dir(str(EnvironmentPaths.agent_dir), state.trial_paths.agent_dir)
        except Exception as error:
            state.gaps.append(ObservationGap(code="agent_logs_unavailable", detail=f"{type(error).__name__}: {error}"))
        if context.is_empty():
            try:
                state.agent.populate_context_post_run(context)
            except Exception as error:
                state.gaps.append(
                    ObservationGap(code="agent_context_unavailable", detail=f"{type(error).__name__}: {error}")
                )
        trajectory_path = state.agent.logs_dir / "trajectory.json"
        return trajectory_path if trajectory_path.is_file() else None

    def _agent_config(self, capture_key: str) -> AgentConfig:
        config = self.config.harbor_agent.model_copy(deep=True)
        if self.config.model_server is not None:
            base_url = self.resolve_model_base_url(self.config.model_server.name, capture_key)
            if self.config.model_base_url_kwarg:
                config.kwargs = {**config.kwargs, self.config.model_base_url_kwarg: base_url}
            config.env = {**config.env, **dict.fromkeys(self.config.model_base_url_env, base_url)}
        return config

    def _setup_timeout_seconds(self) -> float:
        base = self.config.harbor_agent.override_setup_timeout_sec
        if base is None:
            base = AgentFactory.get_agent_class_from_config(self.config.harbor_agent).DEFAULT_SETUP_TIMEOUT_SEC
        return base * self.config.agent_setup_timeout_multiplier

    def _agent_timeout_seconds(self, task_timeout: Any) -> Optional[float]:
        base = self.config.harbor_agent.override_timeout_sec or (float(task_timeout) if task_timeout else None)
        if base is None:
            return None
        return (
            min(base, self.config.harbor_agent.max_timeout_sec or float("inf")) * self.config.agent_timeout_multiplier
        )

    @staticmethod
    def _source(state: HarborHarnessSession) -> str:
        return f"harbor:{state.agent.name()}"


def _instruction(input_value: Any) -> str:
    """Join the text of the task's input messages; Harbor agents take one instruction string."""
    if isinstance(input_value, str):
        return input_value
    parts: list[str] = []
    for item in input_value or []:
        content = item.get("content", "") if isinstance(item, dict) else ""
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(part.get("text", "") for part in content if isinstance(part, dict) and part.get("text"))
    return "\n\n".join(part for part in parts if part)


def _path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "episode"


if __name__ == "__main__":
    HarborHarnessAgent.run_webserver()
