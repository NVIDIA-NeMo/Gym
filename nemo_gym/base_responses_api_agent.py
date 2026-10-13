# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from time import monotonic
from typing import Any, Optional
from warnings import warn

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.global_config import (
    OBSERVABILITY_ENABLED_KEY_NAME,
    TOKEN_ID_CAPTURE_BLOCK,
    get_first_server_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body, rollout_context
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.server_utils import (
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
    apply_rollout_prefix,
    rollout_path_prefix,
)
from nemo_gym.telemetry.endpoints import traced_endpoint, traced_rollout_endpoint
from nemo_gym.telemetry.span_groups import GymSpanGroup
from nemo_gym.tool_access import ToolAccess


# Session cookie key that binds later activation and close requests to their agent session.
AGENT_SESSION_COOKIE_KEY = "agent_session_id"


class AgentSeedSessionRequest(BaseModel):
    """Idempotently initialize agent-server state under a caller-assigned identifier.

    Repeating the same identifier and episode must return the existing session.
    Closing an unknown identifier must prevent a racing seed within the retry window.
    The shared implementation retains close responses for session_close_retry_window_seconds;
    callers must use unique IDs and finish retries within that window. External resources
    should use provider TTLs when available; there is no active-session expiry timer.
    """

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str = Field(min_length=1)
    episode_id: EpisodeId
    task_id: TaskId
    tool_accesses: list[ToolAccess] = Field(default_factory=list)
    sandbox_access: SandboxAccess | None = None

    @field_validator("tool_accesses")
    @classmethod
    def require_unique_tool_names(cls, tool_accesses: list[ToolAccess]) -> list[ToolAccess]:
        names = [access.name for access in tool_accesses]
        if len(names) != len(set(names)):
            raise ValueError("tool access names must be unique within an agent session")
        return tool_accesses


class AgentSeedSessionResponse(BaseModel):
    """Confirm the caller-assigned agent session identifier."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str


class AgentCloseSessionRequest(BaseModel):
    """Close agent-server state."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str
    episode_id: EpisodeId


class AgentCloseSessionResponse(BaseModel):
    """Confirm closure and return captured observations."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str
    agent_observations: AgentObservationBundle | None = None
    resources_cookies: dict[str, str] | None = Field(
        default=None,
        description=(
            "The Resources Server cookie jar after the agent's tool calls. "
            "None or an empty jar leaves the Environment Server's jar from the Resources seed unchanged."
        ),
    )


class BaseResponsesAPIAgentConfig(BaseRunServerInstanceConfig):
    session_close_retry_window_seconds: float = Field(
        default=300,
        gt=0,
        allow_inf_nan=False,
        description="Retain close responses for this many seconds after cleanup; cover the caller's retry horizon.",
    )
    skip_verification: bool = False
    skip_verification_reward: float = 0.0
    # Whether this agent's rollouts participate in training token capture.
    # Native agents already receive token ids inline and normally leave this disabled.
    # Opaque external harnesses enable it because their returned output has no token ids.
    # The run-level ``token_id_capture.enabled`` setting gates the capture infrastructure.
    # The run-level ``token_id_capture.all_agents`` setting overrides this agent-level choice.
    token_id_capture: bool = False
    tool_accesses: list[ToolAccess] = Field(default_factory=list)

    @field_validator("tool_accesses")
    @classmethod
    def require_unique_tool_names(cls, tool_accesses: list[ToolAccess]) -> list[ToolAccess]:
        names = [access.name for access in tool_accesses]
        if len(names) != len(set(names)):
            raise ValueError("configured tool access names must be unique")
        return tool_accesses


class BaseResponsesAPIAgent(BaseServer):
    config: BaseResponsesAPIAgentConfig


@dataclass
class AgentSessionState:
    """Harness-owned session state with the immutable caller-assigned seed binding."""

    request: AgentSeedSessionRequest


class AgentSessionSetupError(Exception):
    """Retain incomplete setup for close while propagating the original setup error."""

    def __init__(self, state: AgentSessionState, *, error: BaseException) -> None:
        super().__init__(str(error))
        self.state: AgentSessionState = state
        self.error: BaseException = error


@dataclass
class _AgentSessionRecord:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state: AgentSessionState | None = None
    closing: bool = False
    episode_id: EpisodeId | None = None
    close_response: AgentCloseSessionResponse | None = None
    expires_at: float = float("inf")


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, AggregateMetricsMixin, SimpleServer):
    config: BaseResponsesAPIAgentConfig
    _session_records: dict[str, _AgentSessionRecord] = PrivateAttr(default_factory=dict)
    _closed_session_records: OrderedDict[str, _AgentSessionRecord] = PrivateAttr(default_factory=OrderedDict)

    @asynccontextmanager
    async def _locked_agent_session(self, session_id: str) -> AsyncIterator[_AgentSessionRecord]:
        now = monotonic()
        while self._closed_session_records:
            key, record = next(iter(self._closed_session_records.items()))
            if record.expires_at > now:
                break
            self._closed_session_records.pop(key)
            self._session_records.pop(key)
        while True:
            record = self._session_records.setdefault(session_id, _AgentSessionRecord())
            async with record.lock:
                # A waiter can outlive a failed seed or expired receipt. Never use its old lock
                # to access a replacement record for the same ID.
                if self._session_records.get(session_id) is not record:
                    continue
                try:
                    yield record
                finally:
                    if record.state is None and record.close_response is None:
                        self._session_records.pop(session_id)
                return

    @staticmethod
    def _agent_session_id_from_request(request: Request | None) -> str | None:
        if request is None:
            return None
        try:
            session = request.session
        except (AssertionError, AttributeError):
            return None
        if not isinstance(session, Mapping) or AGENT_SESSION_COOKIE_KEY not in session:
            return None
        marker = session[AGENT_SESSION_COOKIE_KEY]
        if not isinstance(marker, str) or not marker:
            raise HTTPException(409, "Invalid agent session marker")
        return marker

    def _require_agent_session(self, agent_session_id: str) -> AgentSessionState:
        record = self._session_records.get(agent_session_id)
        if record is None or record.state is None or record.closing:
            raise HTTPException(409, "Unknown or closing agent_session_id")
        return record.state

    def effective_tool_accesses(self, request: AgentSeedSessionRequest) -> list[ToolAccess]:
        """Overlay episode-scoped tool access onto configured declarations by name."""
        accesses = {access.name: access for access in self.config.tool_accesses}
        accesses.update((access.name, access) for access in request.tool_accesses)
        return list(accesses.values())

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        agent_attributes = {"nemo.gym.server.name": self.config.name}
        traced_responses = traced_endpoint(GymSpanGroup.AGENT, "gym.agent.responses", self.responses, agent_attributes)
        app.post("/v1/responses")(traced_responses)
        # A self-call made with ``url_path_for_run`` lands on a prefixed twin.
        # ``responses`` recovers the rollout id from the path.
        # The same handler serves prefixed and unprefixed calls.
        app.post(f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/v1/responses")(traced_responses)
        app.post(f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/{TOKEN_CAPTURE_PATH_SEGMENT}/v1/responses")(traced_responses)

        # Traced *inside* rollout_context, not outside it. The span reads
        # `current_rollout_id()` when it starts, so wrapping the other way round would
        # start the span before the ContextVar is set and every rollout span would be
        # missing its `nemo.gym.rollout.id` — which is exactly what a first run on real
        # hardware showed.
        run = traced_rollout_endpoint(self.run, agent_attributes)

        @wraps(run)
        async def run_with_rollout_context(*args: Any, **kwargs: Any) -> BaseVerifyResponse:
            body = kwargs.get("body")
            if body is None:
                body = next((arg for arg in args if isinstance(arg, BaseRunRequest)), None)
            with rollout_context(self.rollout_id_from_run(body)):
                return await run(*args, **kwargs)

        app.post("/run")(run_with_rollout_context)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        app.post("/v1/agent_sessions")(self.seed_agent_session)
        app.post("/v1/agent_sessions/close")(self.close_agent_session)

        return app

    async def seed_agent_session(
        self,
        request: Request,
        body: AgentSeedSessionRequest,
    ) -> AgentSeedSessionResponse:
        """Seed once per caller ID; identical retries reuse the same harness state."""
        if self.config.num_workers not in (None, 1):
            raise ValueError("Agent sessions require num_workers=1")
        current = self._agent_session_id_from_request(request)
        if current is not None and current != body.agent_session_id:
            raise HTTPException(409, "agent_session_id does not match the session cookie")
        async with self._locked_agent_session(body.agent_session_id) as record:
            if record.close_response is not None:
                raise HTTPException(409, "Agent session is already closed")
            if record.closing:
                raise HTTPException(409, "Agent session is closing")
            if record.state is None:
                if current is not None:
                    raise HTTPException(409, "Agent session cookie has expired")
                record.episode_id = body.episode_id
                try:
                    record.state = await self._seed_agent_session_state(body.model_copy(deep=True))
                except AgentSessionSetupError as error:
                    record.state = error.state
                    record.closing = True
                    raise error.error from None
            elif record.state.request != body:
                raise HTTPException(409, "agent_session_id is already bound to another seed request")
            request.session[AGENT_SESSION_COOKIE_KEY] = body.agent_session_id
            return AgentSeedSessionResponse(agent_session_id=body.agent_session_id)

    async def close_agent_session(
        self,
        request: Request,
        body: AgentCloseSessionRequest,
    ) -> AgentCloseSessionResponse:
        """Retain successful close responses for a bounded retry window, including observations.

        Failed cleanup keeps state for retry. Closing an unknown ID prevents a delayed seed
        within the same window. No timer cancels active sessions: the episode owner and sandbox
        provider retain responsibility for normal and crash cleanup.
        """
        if type(self)._close_agent_session_state is SimpleResponsesAPIAgent._close_agent_session_state:
            raise NotImplementedError("This agent does not implement episode sessions")
        current = self._agent_session_id_from_request(request)
        if current is not None and current != body.agent_session_id:
            raise HTTPException(409, "agent_session_id does not match the session cookie")
        async with self._locked_agent_session(body.agent_session_id) as record:
            if record.episode_id is not None and record.episode_id != body.episode_id:
                raise HTTPException(409, "episode_id does not match the seeded agent session")
            if record.close_response is not None:
                return record.close_response.model_copy(deep=True)
            if record.state is None:
                if current is not None:
                    raise HTTPException(409, "Agent close receipt has expired")
                result = AgentCloseSessionResponse(agent_session_id=body.agent_session_id)
            else:
                record.closing = True
                result = await self._close_agent_session_state(record.state)
            record.state = None
            record.episode_id = body.episode_id
            record.close_response = result.model_copy(deep=True)
            record.expires_at = monotonic() + self.config.session_close_retry_window_seconds
            self._closed_session_records[body.agent_session_id] = record
            # Keep the marker so a stale /responses request cannot fall back to the non-session path.
            request.session[AGENT_SESSION_COOKIE_KEY] = body.agent_session_id
            return result

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        """Validate grants and initialize harness state.

        If setup fails and cleanup cannot finish, raise AgentSessionSetupError with the
        partial state and original error. The base retains it for close, never activation.
        """
        raise NotImplementedError("This agent does not implement episode sessions")

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Release harness state, or raise without losing the handle needed for another close."""
        raise NotImplementedError("This agent does not implement episode sessions")

    def _capture_correlation_enabled(self) -> bool:
        """Return whether this agent needs rollout correlation.

        Evaluation uses ``/ng-rollout/<id>/...`` for every agent.
        Training capture uses ``/ng-rollout/<id>/training-token-capture/...``.
        Training capture requires ``token_id_capture.enabled``.
        It also requires the static agent flag or run-level ``all_agents``.
        Missing global configuration disables correlation.
        """
        return self._model_call_capture_enabled() or self._token_id_capture_enabled()

    def _model_call_capture_enabled(self) -> bool:
        """Whether evaluation model-call observability is enabled."""
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return False
        return bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False))

    def _token_id_capture_enabled(self) -> bool:
        """Whether this agent explicitly opted into training-token capture."""
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return False
        block = global_config.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        if not isinstance(block, Mapping) or not block.get("enabled", False):
            return False
        return bool(block.get("all_agents", False)) or bool(
            getattr(getattr(self, "config", None), "token_id_capture", False)
        )

    def rollout_id_from_run(self, body: Any) -> Optional[str]:
        """Return the capture id for a run request.

        Return ``None`` when capture is disabled.
        Return ``None`` when the body has no usable identity.
        """
        if not self._capture_correlation_enabled():
            return None
        return maybe_rollout_id_from_run_body(body)

    def url_path_for_run(self, url_path: str, body: Any) -> str:
        """Apply this run's capture path to a downstream URL path.

        Evaluation uses ``/ng-rollout/<id>/...``.
        Training capture uses ``/ng-rollout/<id>/training-token-capture/...``.
        Calls without a rollout id remain unchanged.
        """
        return (
            f"{rollout_path_prefix(self.rollout_id_from_run(body), token_capture=self._token_id_capture_enabled())}"
            f"{url_path}"
        )

    def base_url_for_run(self, base_url: str, body: Any) -> str:
        """Apply this run's capture path to a model-server root URL.

        Append the API-version suffix after this method returns.
        """
        return apply_rollout_prefix(
            base_url,
            self.rollout_id_from_run(body),
            token_capture=self._token_id_capture_enabled(),
        )

    def url_path_for_request(self, url_path: str, request: Optional[Request]) -> str:
        """Carry an inbound capture path onto a downstream URL path.

        Prefixed self-calls expose the rollout id as a path parameter.
        Training-capture requests preserve their dedicated path segment.
        Unprefixed requests remain unchanged.
        """
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        request_path = getattr(getattr(request, "url", None), "path", "")
        token_capture = f"/{TOKEN_CAPTURE_PATH_SEGMENT}/" in request_path
        return f"{rollout_path_prefix(rollout_id, token_capture=token_capture)}{url_path}"

    def resolve_model_base_url(self, model_server_name: str, rollout_id: Optional[str] = None) -> str:
        """Resolve a model-server URL with an optional rollout prefix."""
        server_config = get_first_server_config_dict(self.server_client.global_config_dict, model_server_name)
        base_url = self.server_client._build_server_base_url(server_config)
        return f"{apply_rollout_prefix(base_url, rollout_id, token_capture=self._token_id_capture_enabled())}/v1"

    # TODO: right now there is no validation on the TypedDict NeMoGymResponseCreateParamsNonStreaming
    # We should explicitly add validation at this server level or we should explicitly not validate so that there is flexibility in this API.
    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    @abstractmethod
    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        pass

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Default: same RewardProfiler aggregation as resources server. Override to proxy."""
        if self.config.skip_verification:
            warn(
                "Skipping aggregate metrics because skip_verification=True; "
                "use disable_aggregation=True to avoid writing aggregate metric files.",
                RuntimeWarning,
                stacklevel=2,
            )
            return AggregateMetrics()

        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
            compute_repeat_metrics_fn=self.compute_repeat_metrics,
        )
