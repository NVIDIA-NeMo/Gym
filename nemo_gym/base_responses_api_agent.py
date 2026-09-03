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
from collections.abc import Awaitable, Callable, Mapping
from contextlib import nullcontext
from functools import wraps
from typing import Any, ClassVar, Optional
from warnings import warn

import orjson
from fastapi import Body, FastAPI, Request
from pydantic import PrivateAttr

from nemo_gym._checkpoint.agent import (
    AGENT_EXECUTION_GENERATION_HEADER,
    AgentBoundaryRecord,
    AgentCheckpointParticipant,
    AgentExecution,
    install_agent_checkpoint,
)
from nemo_gym._checkpoint.control import ControlCapabilities, checkpoint_control_auth_token
from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT
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
from nemo_gym.rollout_correlation import (
    RolloutContextMiddleware,
    capture_key_for,
    checkpoint_parent_context,
    current_attempt_index,
    current_logical_rollout_id,
    execution_identity_from_run_body,
    maybe_rollout_id_from_run_body,
    rollout_context,
)
from nemo_gym.server_utils import (
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
    apply_rollout_prefix,
    rollout_path_prefix,
)
from nemo_gym.telemetry.endpoints import traced_endpoint, traced_rollout_endpoint
from nemo_gym.telemetry.span_groups import GymSpanGroup


class BaseResponsesAPIAgentConfig(BaseRunServerInstanceConfig):
    skip_verification: bool = False
    skip_verification_reward: float = 0.0
    # Whether this agent's rollouts participate in training token capture.
    # Native agents already receive token ids inline and normally leave this disabled.
    # Opaque external harnesses enable it because their returned output has no token ids.
    # The run-level ``token_id_capture.enabled`` setting gates the capture infrastructure.
    # The run-level ``token_id_capture.all_agents`` setting overrides this agent-level choice.
    token_id_capture: bool = False


class BaseResponsesAPIAgent(BaseServer):
    config: BaseResponsesAPIAgentConfig


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, AggregateMetricsMixin, SimpleServer):
    config: BaseResponsesAPIAgentConfig

    _CONTROL_COMPONENT = "responses_api_agents"
    checkpoint_continuation_supported: ClassVar[bool] = False
    _checkpoint_participant: Optional[AgentCheckpointParticipant] = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)
        app.add_middleware(RolloutContextMiddleware)
        self.setup_control_plane(app)
        self.setup_agent_checkpoint(app)

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
            logical_rollout_id, attempt_index = execution_identity_from_run_body(body)
            capture_key = maybe_rollout_id_from_run_body(body)
            with rollout_context(
                capture_key,
                attempt_index=attempt_index,
                logical_rollout_id=logical_rollout_id,
            ):
                if logical_rollout_id is None or attempt_index is None or self._checkpoint_participant is None:
                    return await run(*args, **kwargs)
                execution = await self._checkpoint_participant.begin(
                    logical_rollout_id,
                    attempt_index,
                    task=asyncio.current_task(),
                )
                if execution.terminal_result is not None:
                    return execution.terminal_result
                token = self._checkpoint_participant.bind(execution)
                continuation = self._checkpoint_participant.continuation(execution)
                parent_context = (
                    checkpoint_parent_context(
                        capture_key_for(continuation.rollout_id, continuation.attempt_index),
                        continuation.last_committed_model_call_id,
                    )
                    if continuation is not None and continuation.last_committed_model_call_id is not None
                    else nullcontext()
                )
                try:
                    with parent_context:
                        result = await run(*args, **kwargs)
                except asyncio.CancelledError:
                    await self._checkpoint_participant.finish(execution, outcome="cancelled")
                    raise
                except BaseException:
                    await self._checkpoint_participant.finish(execution, outcome="failed")
                    raise
                finally:
                    self._checkpoint_participant.unbind(token)
                await self._checkpoint_participant.finish(execution, outcome="completed", result=result)
                return result

        app.post("/run")(run_with_rollout_context)
        app.post("/aggregate_metrics")(self.aggregate_metrics)

        return app

    def checkpoint_participant(self) -> AgentCheckpointParticipant:
        if self._checkpoint_participant is None:
            self._checkpoint_participant = AgentCheckpointParticipant(self.config.name)
        return self._checkpoint_participant

    def checkpoint_control_auth_token(self) -> Optional[str]:
        global_config = getattr(self.server_client, "global_config_dict", None)
        return checkpoint_control_auth_token(global_config)

    def setup_agent_checkpoint(self, app: FastAPI) -> None:
        auth_token = self.checkpoint_control_auth_token()
        if auth_token is None or not self.checkpoint_continuation_supported:
            return
        install_agent_checkpoint(
            app,
            participant=self.checkpoint_participant(),
            fence=self.checkpoint_fence(),
            auth_token=auth_token,
        )

    def control_capabilities(self) -> ControlCapabilities:
        capabilities = super().control_capabilities()
        if self.checkpoint_control_auth_token() is not None and self.checkpoint_continuation_supported:
            capabilities.checkpoint_mode = "export_restore"
            capabilities.concurrency_contract = "serialized_per_session"
        return capabilities

    def checkpoint_execution(self, request: Optional[Request] = None) -> Optional[AgentExecution]:
        if self._checkpoint_participant is None:
            return None
        current = self._checkpoint_participant.current_execution()
        if current is not None:
            return current
        logical_rollout_id = current_logical_rollout_id()
        attempt_index = current_attempt_index()
        if logical_rollout_id is None or attempt_index is None or request is None:
            return None
        generation_raw = request.headers.get(AGENT_EXECUTION_GENERATION_HEADER)
        if generation_raw is None:
            return None
        try:
            generation = int(generation_raw)
        except ValueError:
            return None
        return self._checkpoint_participant.resolve(
            logical_rollout_id,
            attempt_index,
            generation=generation,
        )

    def checkpoint_execution_headers(self) -> Optional[dict[str, str]]:
        execution = self.checkpoint_execution()
        if execution is None:
            return None
        return {AGENT_EXECUTION_GENERATION_HEADER: str(execution.generation)}

    def checkpoint_continuation(
        self,
        body: Any,
        request: Optional[Request] = None,
    ) -> Optional[AgentBoundaryRecord]:
        """Return the restored boundary for the current replacement attempt."""
        execution = self.checkpoint_execution(request)
        if execution is None:
            return None
        return self._checkpoint_participant.continuation(execution)

    async def retry_checkpoint_refusal(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        request: Optional[Request] = None,
    ) -> Any:
        execution = self.checkpoint_execution(request)
        while True:
            response = await operation()
            if await _checkpoint_refusal_code(response) is None:
                return response
            if execution is None or self._checkpoint_participant is None:
                return response
            await self._checkpoint_participant.park(execution)

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
        )


async def _checkpoint_refusal_code(response: Any) -> Optional[str]:
    status = getattr(response, "status", None)
    if not isinstance(status, int) or status != 409:
        return None
    try:
        payload = orjson.loads(await response.read())
    except (TypeError, ValueError):
        return None
    error = payload.get("error") if isinstance(payload, Mapping) else None
    code = error.get("code") if isinstance(error, Mapping) else None
    if code in {"checkpoint_parked", "resources_admission_closed"}:
        return code
    return None
