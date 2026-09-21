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
from time import monotonic
from typing import Any, ClassVar, Optional
from uuid import uuid4
from warnings import warn

import orjson
from fastapi import Body, FastAPI, Request
from pydantic import PrivateAttr

from nemo_gym._checkpoint.agent import (
    AGENT_EXECUTION_GENERATION_HEADER,
    COMPLETED_RESULT_ACKNOWLEDGEMENT_FEATURE,
    DISCARD_RESTORED_CONTINUATION_FEATURE,
    AgentBoundaryRecord,
    AgentCheckpointParticipant,
    AgentExecution,
    install_agent_checkpoint,
)
from nemo_gym._checkpoint.artifacts import (
    AGENT_CONTINUATION_INDEX_FEATURE,
    AGENT_RESOURCE_DEPENDENCY_INDEX_FEATURE,
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
    MODEL_CALL_CAPTURE_OUTCOME_HEADER,
    MODEL_CALL_ID_HEADER,
    ModelCallCaptureOutcome,
    ModelCallCaptureResult,
    RolloutContextMiddleware,
    checkpoint_model_call_capture,
    checkpoint_parent_context,
    current_attempt_index,
    current_logical_rollout_id,
    current_rollout_id,
    execution_identity_from_run_body,
    maybe_rollout_id_from_run_body,
    model_request_correlation,
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
    # Opt in only when an unfinished verifier is safe to replay after recovery.
    checkpoint_replayable_verify: bool = False
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
    checkpoint_resource_dependencies_supported: ClassVar[bool] = False
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
        app.post(f"/{TOKEN_CAPTURE_PATH_SEGMENT}/v1/responses")(traced_responses)
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
                        continuation.last_committed_model_capture_key,
                        continuation.last_committed_model_call_id,
                    )
                    if (
                        continuation is not None
                        and continuation.last_committed_model_capture_key is not None
                        and continuation.last_committed_model_call_id is not None
                    )
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
        if auth_token is None:
            return
        install_agent_checkpoint(
            app,
            participant=self.checkpoint_participant(),
            fence=self.checkpoint_fence(),
            auth_token=auth_token,
        )

    def control_capabilities(self) -> ControlCapabilities:
        capabilities = super().control_capabilities()
        if self.checkpoint_control_auth_token() is not None:
            capabilities.checkpoint_mode = "export_restore"
            capabilities.concurrency_contract = "serialized_per_session"
            capabilities.features = [COMPLETED_RESULT_ACKNOWLEDGEMENT_FEATURE]
            if self.checkpoint_continuation_supported:
                capabilities.features.extend(
                    [
                        AGENT_CONTINUATION_INDEX_FEATURE,
                        DISCARD_RESTORED_CONTINUATION_FEATURE,
                    ]
                )
                if self.checkpoint_resource_dependencies_supported:
                    capabilities.features.append(AGENT_RESOURCE_DEPENDENCY_INDEX_FEATURE)
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
        checkpointable_model_wait: bool = False,
    ) -> Any:
        execution = self.checkpoint_execution(request)
        refusal_count = 0
        logical_call_id = uuid4().hex if checkpointable_model_wait else None
        request_attempt_number = 0
        while True:
            checkpoint_resumed_during_model_wait = False
            request_attempt_number += 1
            request_attempt_id = uuid4().hex if logical_call_id is not None else None
            request_started: Optional[float] = None

            def log_request_started() -> None:
                nonlocal request_started
                if logical_call_id is None or request_attempt_id is None:
                    return
                request_started = monotonic()
                print(
                    "[agent_model_request_lifecycle] "
                    + orjson.dumps(
                        {
                            "stage": "agent_http_request_started",
                            "participant": (
                                self._checkpoint_participant.instance_name
                                if self._checkpoint_participant is not None
                                else None
                            ),
                            "rollout_id": execution.rollout_id if execution is not None else None,
                            "attempt_index": execution.attempt_index if execution is not None else None,
                            "logical_call_id": logical_call_id,
                            "request_attempt_id": request_attempt_id,
                            "request_attempt_number": request_attempt_number,
                        }
                    ).decode(),
                    flush=True,
                )

            correlation_context = (
                model_request_correlation(logical_call_id, request_attempt_id)
                if logical_call_id is not None and request_attempt_id is not None
                else nullcontext()
            )
            if checkpointable_model_wait and execution is not None and self._checkpoint_participant is not None:
                await self._checkpoint_participant.begin_model_wait(execution)
                try:
                    log_request_started()
                    with correlation_context:
                        response = await operation()
                finally:
                    checkpoint_resumed_during_model_wait = await self._checkpoint_participant.end_model_wait(execution)
            else:
                log_request_started()
                with correlation_context:
                    response = await operation()
            if logical_call_id is not None and request_attempt_id is not None:
                response_status = getattr(response, "status", None)
                if not isinstance(response_status, int):
                    response_status = None
                print(
                    "[agent_model_request_lifecycle] "
                    + orjson.dumps(
                        {
                            "stage": "agent_http_response_received",
                            "participant": (
                                self._checkpoint_participant.instance_name
                                if self._checkpoint_participant is not None
                                else None
                            ),
                            "rollout_id": execution.rollout_id if execution is not None else None,
                            "attempt_index": execution.attempt_index if execution is not None else None,
                            "logical_call_id": logical_call_id,
                            "request_attempt_id": request_attempt_id,
                            "request_attempt_number": request_attempt_number,
                            "status": response_status,
                            "elapsed_s": (
                                round(monotonic() - request_started, 3) if request_started is not None else None
                            ),
                        }
                    ).decode(),
                    flush=True,
                )
            refusal_code, refusal_reason = await _checkpoint_refusal(response)
            if refusal_code is None:
                return response
            if execution is None or self._checkpoint_participant is None:
                return response
            refusal_count += 1
            boundary = execution.boundary
            print(
                "[agent_checkpoint_refusal] "
                + orjson.dumps(
                    {
                        "rollout_id": execution.rollout_id,
                        "attempt_index": execution.attempt_index,
                        "generation": execution.generation,
                        "refusal_code": refusal_code,
                        "refusal_reason": refusal_reason,
                        "refusal_count": refusal_count,
                        "logical_call_id": logical_call_id,
                        "request_attempt_id": request_attempt_id,
                        "request_attempt_number": request_attempt_number,
                        "state": execution.state.value,
                        "boundary_index": boundary.boundary_index if boundary is not None else None,
                        "model_wait_depth": execution.model_wait_depth,
                    }
                ).decode(),
                flush=True,
            )
            if refusal_count == 3 or refusal_count % 10 == 0:
                last_reached_stage = {
                    "admission_already_closed": "policy_request_received",
                    "checkpoint_closed_before_generation_started": "pre_generation_setup",
                    "resources_admission_closed": "resources_request_received",
                }.get(refusal_reason, "unknown")
                print(
                    "[checkpoint_starvation_suspected] "
                    + orjson.dumps(
                        {
                            "participant": self._checkpoint_participant.instance_name,
                            "rollout_id": execution.rollout_id,
                            "attempt_index": execution.attempt_index,
                            "logical_call_id": logical_call_id,
                            "request_attempt_id": request_attempt_id,
                            "refusals": refusal_count,
                            "reason": refusal_reason,
                            "last_reached_stage": last_reached_stage,
                        }
                    ).decode(),
                    flush=True,
                )
            retry_mode = "checkpoint_resumed"
            if not checkpoint_resumed_during_model_wait:
                await self._checkpoint_participant.park(execution)
                retry_mode = "park_resume"
            print(
                "[agent_checkpoint_refusal_retry] "
                + orjson.dumps(
                    {
                        "rollout_id": execution.rollout_id,
                        "attempt_index": execution.attempt_index,
                        "generation": execution.generation,
                        "refusal_count": refusal_count,
                        "logical_call_id": logical_call_id,
                        "previous_request_attempt_id": request_attempt_id,
                        "next_request_attempt_number": request_attempt_number + 1,
                        "retry_mode": retry_mode,
                        "state": execution.state.value,
                    }
                ).decode(),
                flush=True,
            )

    async def checkpointable_external_wait(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        request: Optional[Request] = None,
    ) -> Any:
        """Run a replayable external operation without consuming it across a cut."""
        execution = self.checkpoint_execution(request)
        if execution is None or self._checkpoint_participant is None:
            return await operation()
        await self._checkpoint_participant.begin_external_wait(execution)
        try:
            return await operation()
        finally:
            await self._checkpoint_participant.end_external_wait(execution)

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

    def model_call_capture_result(self, headers: Mapping[str, Any] | None) -> ModelCallCaptureResult | None:
        """Validate model-call capture evidence when the response carries it.

        Training-token capture and turn-level checkpoint participation are
        independent capabilities. Typed capture outcomes are authoritative
        whenever present and mandatory for training-token capture. Turn-level
        checkpointing without token capture retains the legacy model-call ID
        header contract used by the model ledger.
        """
        capture_header_present = headers is not None and MODEL_CALL_CAPTURE_OUTCOME_HEADER in headers
        if capture_header_present or self._token_id_capture_enabled():
            return checkpoint_model_call_capture(headers)
        if self._checkpoint_participant is None or headers is None:
            return None
        model_call_id = headers.get(MODEL_CALL_ID_HEADER)
        if not isinstance(model_call_id, str) or not model_call_id:
            return None
        return ModelCallCaptureResult(
            outcome=ModelCallCaptureOutcome.CAPTURED,
            model_call_id=model_call_id,
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
        if rollout_id is None:
            rollout_id = current_rollout_id()
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


async def _checkpoint_refusal(response: Any) -> tuple[Optional[str], Optional[str]]:
    status = getattr(response, "status", None)
    if not isinstance(status, int) or status != 409:
        return None, None
    try:
        payload = orjson.loads(await response.read())
    except (TypeError, ValueError):
        return None, None
    error = payload.get("error") if isinstance(payload, Mapping) else None
    code = error.get("code") if isinstance(error, Mapping) else None
    if code not in {"checkpoint_parked", "resources_admission_closed"}:
        return None, None
    detail = error.get("detail") if isinstance(error, Mapping) else None
    if code == "resources_admission_closed":
        reason = "resources_admission_closed"
    elif isinstance(detail, str) and "admitted before checkpoint close" in detail:
        reason = "checkpoint_closed_before_generation_started"
    elif isinstance(detail, str) and "admission is" in detail:
        reason = "admission_already_closed"
    else:
        reason = "checkpoint_parked"
    return code, reason
