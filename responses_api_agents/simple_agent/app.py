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
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from aiohttp import ClientResponseError
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, TypeAdapter, ValidationError

from nemo_gym._checkpoint.agent import (
    Activation,
    ActivationOutOfOrderError,
    LegacyRun,
    RestoredAgentSession,
    require_rollout,
)
from nemo_gym._checkpoint.errors import RolloutIdRequiredError
from nemo_gym._checkpoint.steps import CHECKPOINT_CLOSE_FOLLOWS_HEADER, StepType, seed_restarts, seed_verify_step_type
from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.base_environment_server import is_retryable_dependency_status
from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
)
from nemo_gym.base_responses_api_agent import (
    AGENT_ACTIVATION_HEADER,
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config import ROLLOUT_PATH_PREFIX, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.failure_kinds import MODEL_REQUEST_ERROR
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInput,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from nemo_gym.server_utils import get_response_json, raise_for_status, rollout_path_prefix
from nemo_gym.server_utils import request as http_request
from nemo_gym.session_routing import SESSION_OWNER_HEADER
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPToolAccess
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


LOG = logging.getLogger(__name__)

_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"
RESOURCE_ONLY_METADATA_KEYS = frozenset(
    {"baseline_response", "baseline_model", "baseline_dataset_file", "baseline_row_index"}
)

# Legacy /run episodes are checkpointed under their logical rollout ID so a replacement attempt finds them.
_LEGACY_SESSION_PREFIX = "run:"
_INPUT_ITEMS_ADAPTER = TypeAdapter(NeMoGymResponseInput)


class SimpleAgentLoopState(BaseModel):
    """Loop position at a boundary: before a model call, or after a model response whose tools are pending."""

    model_config = ConfigDict(extra="forbid")

    step: int
    pending_tools: bool
    new_outputs: list[dict[str, Any]]
    usage: dict[str, Any] | None
    last_model_response: dict[str, Any] | None
    model_server_cookies: dict[str, str]
    resources_server_cookies: dict[str, str]
    # Each turn without its question, which repeats the request's input and the loop's outputs before the turn.
    turns: list[dict[str, Any]]
    # For each turn, how many of the loop's outputs its question holds after the request's input.
    # A restore rebuilds each question from the input the caller sends again and the restored outputs.
    question_outputs: list[int]
    tool_records: list[dict[str, Any]]
    model_calls: list[dict[str, Any]]
    gaps: list[dict[str, Any]]


def _to_json(item: Any) -> Any:
    return item.model_dump(mode="json") if isinstance(item, BaseModel) else item


def _cookie_values(cookies: Any) -> dict[str, str]:
    """Flatten a plain dict or an aiohttp cookie jar of morsels into name-to-value pairs."""
    return {str(name): str(getattr(value, "value", value)) for name, value in (cookies or {}).items()}


@dataclass
class SimpleAgentSessionState(AgentSessionState):
    tool_access: DirectHTTPToolAccess | None
    # None means the session holds no resources cookie jar, so close leaves the Environment Server's jar unchanged.
    resources_cookies: dict[str, str] | None
    observations: AgentObservationBundle | None = None
    activations: int = 0
    # With checkpointing, the last activation that finished and the reply it gave:
    # an environment server restored from a checkpoint before that reply invokes it again and gets the same reply.
    # The saved reply's response is kept as the object the activation returned,
    # and serialized only to return or export it;
    # a restored reply holds the JSON it was exported as.
    completed_activation: int = 0
    completed_reply: dict[str, Any] | None = None


def _dump_reply(reply: dict[str, Any] | None) -> dict[str, Any] | None:
    """A completed reply as a checkpoint records it."""
    if reply is None:
        return None
    return {"response": _to_json(reply["response"]), "cookies": reply["cookies"]}


@dataclass
class _LegacyLoopReply:
    """The reply of a legacy /run's turn loop, the /run's one activation, kept while the /run runs.

    A checkpoint after the loop replied, and before /run recorded its next boundary, exports it with the session.
    The replacement /run continues at the loop step and gets this reply instead of running the loop again.
    """

    completed_activation: int = 0
    completed_reply: dict[str, Any] | None = None


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = None
    model_server: ModelServerRef
    max_steps: int = None
    max_model_retries: int = Field(
        default=2,
        ge=0,
        description="Additional attempts for model HTTP 408, 425, 429 and 5xx responses, with bounded exponential backoff.",
    )
    execute_tools: bool = Field(
        default=True,
        description=(
            "Whether to execute model-requested tools. Disabling tool execution is supported only for agent-session "
            "requests, where unresolved function calls are returned to the Environment Server."
        ),
    )


class SimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SimpleAgent(SimpleResponsesAPIAgent):
    checkpoint_mode = "exported"
    config: SimpleAgentConfig
    # With checkpointing, each legacy /run's loop reply, by its session key.
    # An entry leaves with its /run, or with a retire.
    _legacy_replies: dict[str, _LegacyLoopReply] = PrivateAttr(default_factory=dict)

    @staticmethod
    def _strip_resource_only_metadata(
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        # Keep judge-only baseline data out of policy requests to avoid unnecessary payload.
        metadata = dict(body.metadata or {})
        for key in RESOURCE_ONLY_METADATA_KEYS:
            metadata.pop(key, None)
        return body.model_copy(update={"metadata": metadata or None})

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> SimpleAgentSessionState:
        return self._new_session_state(body)

    def _new_session_state(self, body: AgentSeedSessionRequest) -> SimpleAgentSessionState:
        """Validate a seed request's grants and build its session state; a restore rebuilds sessions the same way."""
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("Simple Agent does not support sandbox access")

        accesses = self.effective_tool_accesses(body)
        unsupported = [access.name for access in accesses if isinstance(access, MCPToolAccess) and access.required]
        if unsupported:
            raise AgentSeedRejectedError(
                f"Simple Agent does not support required MCP tool access: {', '.join(unsupported)}"
            )

        direct_accesses = [access for access in accesses if isinstance(access, DirectHTTPToolAccess)]
        if len(direct_accesses) > 1:
            raise AgentSeedRejectedError("Simple Agent supports at most one direct HTTP tool access per session")
        direct_access = direct_accesses[0] if direct_accesses else None
        return SimpleAgentSessionState(
            request=body,
            tool_access=direct_access,
            resources_cookies=dict(direct_access.cookies) if direct_access is not None else None,
        )

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        return {session_key: self._export_agent_session(session_key) for session_key in session_keys}

    def _export_agent_session(self, session_key: str) -> dict[str, JsonValue]:
        if session_key.startswith(_LEGACY_SESSION_PREFIX):
            # A legacy /run keeps everything else it needs in its own boundaries.
            loop = self._legacy_replies.get(session_key)
            if loop is None or loop.completed_reply is None:
                return {}
            return {
                "completed_activation": loop.completed_activation,
                "completed_reply": _dump_reply(loop.completed_reply),
            }
        state = self._session_state(session_key)
        if not isinstance(state, SimpleAgentSessionState):
            raise TypeError(f"no open Simple Agent session {session_key!r}")
        return {
            "request": state.request.model_dump(mode="json"),
            "resources_cookies": state.resources_cookies,
            "observations": state.observations.model_dump(mode="json") if state.observations is not None else None,
            # Activations started so far, so a restored session neither reuses an invocation ID nor renames one.
            "activations": state.activations,
            "completed_activation": state.completed_activation,
            "completed_reply": _dump_reply(state.completed_reply),
        }

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        restored: dict[str, SimpleAgentSessionState] = {}
        legacy: dict[str, _LegacyLoopReply] = {}
        for session in sessions:
            if session.session_key.startswith(_LEGACY_SESSION_PREFIX):
                if session.session.get("completed_reply") is not None:
                    legacy[session.session_key] = _LegacyLoopReply(
                        session.session["completed_activation"], session.session["completed_reply"]
                    )
                continue
            request = AgentSeedSessionRequest.model_validate(
                session.session["request"] | {"episode_id": session.episode_id.model_dump(mode="json")}
            )
            state = self._new_session_state(request)
            cookies = session.session["resources_cookies"]
            state.resources_cookies = dict(cookies) if cookies is not None else None
            observations = session.session.get("observations")
            state.observations = AgentObservationBundle.model_validate(observations) if observations else None
            state.activations = session.session["activations"]
            state.completed_activation = session.session["completed_activation"]
            state.completed_reply = session.session["completed_reply"]
            restored[session.session_key] = state
        for session_key, state in restored.items():
            self._install_restored_session(session_key, state)
        self._legacy_replies.update(legacy)

    async def retire_agent_session(self, session_key: str) -> None:
        self._legacy_replies.pop(session_key, None)
        self._free_session(session_key)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, SimpleAgentSessionState):
            raise TypeError("Expected Simple Agent session state")
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=state.observations,
            resources_cookies=state.resources_cookies,
        )

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
        resources_server_cookies: Any = None,
        tool_access: DirectHTTPToolAccess | None = None,
        resources_prefix: str = "",
        in_session: bool = False,
        execute_tools: bool = True,
        invocation_id: str = "root",
        task_id: str = "unscoped",
        rollout_id: str = "unscoped",
        collect_trajectory: bool = False,
        activation: Activation | None = None,
    ) -> tuple[NeMoGymResponse, TrajectoryRecord | None, Any, Any]:
        tool_records: list[TrajectoryToolCall] = []
        model_calls: list[ModelCallRef] = []
        turns: list[TrajectoryTurn] = []
        question_outputs: list[int] = []
        trajectory_gaps: list[ObservationGap] = []
        body = body.model_copy(deep=True)
        body = self._strip_resource_only_metadata(body)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        usage = None
        step = 0
        invocation_status = "completed"
        invocation_error_type = None
        model_server_cookies = None
        # A restored boundary after a model response resumes with that response's tool calls.
        pending_tools = False
        model_response: NeMoGymResponse | None = None

        async def commit_boundary() -> None:
            if activation is None:
                return
            # Pin the loop position now and build the state only if a checkpoint exports it.
            # The lists only grow,
            # so their current lengths pin this boundary even while the loop has moved on to its next policy call.
            frozen = (step, pending_tools, usage, model_response, model_server_cookies, resources_server_cookies)
            lists = (new_outputs, turns, question_outputs, tool_records, model_calls, trajectory_gaps)
            lengths = tuple(len(items) for items in lists)

            def snapshot() -> dict[str, Any]:
                at_step, tools_pending, at_usage, response, model_cookies, resources_cookies = frozen
                outputs, at_turns, at_questions, at_tools, at_calls, at_gaps = (
                    items[:n] for items, n in zip(lists, lengths)
                )
                # The fields of SimpleAgentLoopState, built as JSON directly: a restore validates them.
                return {
                    "step": at_step,
                    "pending_tools": tools_pending,
                    "new_outputs": [_to_json(item) for item in outputs],
                    "usage": _to_json(at_usage) if at_usage is not None else None,
                    "last_model_response": response.model_dump(mode="json") if response is not None else None,
                    "model_server_cookies": _cookie_values(model_cookies),
                    "resources_server_cookies": _cookie_values(resources_cookies),
                    "turns": [turn.model_dump(mode="json", exclude={"question"}) for turn in at_turns],
                    "question_outputs": at_questions,
                    "tool_records": [record.model_dump(mode="json") for record in at_tools],
                    "model_calls": [call.model_dump(mode="json") for call in at_calls],
                    "gaps": [gap.model_dump(mode="json") for gap in at_gaps],
                }

            await activation.boundary(snapshot)

        if activation is not None and activation.continuation is not None:
            restored = SimpleAgentLoopState.model_validate(activation.continuation)
            step = restored.step
            pending_tools = restored.pending_tools
            new_outputs = _INPUT_ITEMS_ADAPTER.validate_python(restored.new_outputs)
            usage = NeMoGymResponseUsage.model_validate(restored.usage) if restored.usage is not None else None
            if restored.last_model_response is not None:
                model_response = NeMoGymResponse.model_validate(restored.last_model_response)
            model_server_cookies = restored.model_server_cookies
            resources_server_cookies = restored.resources_server_cookies
            turns = [
                TrajectoryTurn.model_validate(turn | {"question": body.input + new_outputs[:count]})
                for turn, count in zip(restored.turns, restored.question_outputs)
            ]
            question_outputs = list(restored.question_outputs)
            tool_records = [TrajectoryToolCall.model_validate(record) for record in restored.tool_records]
            model_calls = [ModelCallRef.model_validate(call) for call in restored.model_calls]
            trajectory_gaps = [ObservationGap.model_validate(gap) for gap in restored.gaps]
        # The boundary before the first model call, or the restored one:
        # a checkpoint may park the activation before it does anything, and export this boundary meanwhile.
        await commit_boundary()

        while True:
            if not pending_tools:
                step += 1
                new_body = body.model_copy(update={"input": body.input + new_outputs})
                question_length = len(new_outputs)
                if collect_trajectory:
                    turn_timestamp = time()

                async def call_model() -> tuple[Any, Any]:
                    retry = 0
                    while True:
                        response = await self.server_client.post(
                            server_name=self.config.model_server.name,
                            url_path=model_url_path,
                            json=new_body,
                            cookies=model_server_cookies,
                            **({"headers": {"x-session-id": invocation_id}} if collect_trajectory else {}),
                        )
                        try:
                            await raise_for_status(response)
                        except ClientResponseError as error:
                            if retry == self.config.max_model_retries or not is_retryable_dependency_status(
                                error.status
                            ):
                                raise
                            LOG.warning(
                                "Retrying model HTTP %s (%s/%s) model_server=%s rollout_id=%s",
                                error.status,
                                retry + 1,
                                self.config.max_model_retries,
                                self.config.model_server.name,
                                rollout_id,
                            )
                            await asyncio.sleep(0.5 * 2 ** min(retry, 4))
                            retry += 1
                            continue
                        return await get_response_json(response), response.cookies

                try:
                    model_response_json, model_server_cookies = await self._call_policy(activation, call_model)
                except ClientResponseError as error:
                    if not in_session:
                        raise
                    # Return the cause and partial work; the Environment Server owns grading and retry policy.
                    content = getattr(error, "response_content", b"")
                    detail = content.decode(errors="replace") if isinstance(content, bytes) else str(content)
                    invocation_status = "failed"
                    invocation_error_type = MODEL_REQUEST_ERROR
                    model_response = NeMoGymResponse(
                        id=f"resp_{uuid4().hex}",
                        created_at=time(),
                        model=body.model or self.config.model_server.name,
                        object="response",
                        output=[],
                        status="failed",
                        error={
                            "code": "server_error",
                            "message": f"Model HTTP {error.status}: {detail or error.message}",
                        },
                        tools=body.tools,
                        tool_choice=body.tool_choice,
                        parallel_tool_calls=body.parallel_tool_calls,
                        metadata={
                            **(body.metadata or {}),
                            AGENT_OUTCOME_METADATA_KEY: AgentOutcome(
                                failure_kind=MODEL_REQUEST_ERROR, status_code=error.status
                            ).model_dump_json(),
                        },
                    )
                    # No successful model response means known zero aggregate usage, not invented per-call usage.
                    if step == 1:
                        usage = NeMoGymResponseUsage.sum_from_list([])
                    break
                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as e:
                    raise RuntimeError(
                        f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                    ) from e

                output = model_response.output
                new_outputs.extend(output)
                if collect_trajectory:
                    turn_model_calls = []
                    if model_response.id:
                        model_call_ref = ModelCallRef(
                            model_ref=self.config.model_server, response_id=model_response.id
                        )
                        model_calls.append(model_call_ref)
                        turn_model_calls.append(model_call_ref)
                    else:
                        trajectory_gaps.append(
                            ObservationGap(
                                code="model_call_reference_unavailable",
                                invocation_id=invocation_id,
                                detail=f"turn:{step}",
                            )
                        )
                    reasoning = [item.model_dump(mode="json") for item in output if item.type == "reasoning"] or None
                    answer = [item for item in output if item.type != "reasoning"]
                    turns.append(
                        TrajectoryTurn(
                            invocation_id=invocation_id,
                            task_id=task_id,
                            rollout_id=rollout_id,
                            turn_no=step,
                            timestamp=turn_timestamp,
                            question=new_body.input,
                            answer=answer,
                            reasoning_content=reasoning,
                            step_count=len(tool_records),
                            model_calls=turn_model_calls,
                        )
                    )
                    question_outputs.append(question_length)

                usage = accumulate_response_usage(usage, model_response.usage)
                model_response.usage = None

                if model_response.incomplete_details:
                    invocation_status = "incomplete"
                    break

                all_fn_calls: list[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
                all_output_messages: list[NeMoGymResponseOutputMessage] = [
                    o for o in output if o.type == "message" and o.role == "assistant"
                ]
                if not all_fn_calls:
                    if not all_output_messages:
                        invocation_status = "incomplete"
                        termination_message = (
                            "Ending trajectory: model returned no assistant message or tool calls "
                            "(reasoning-only or empty output) without reported truncation. "
                            "This is the stop-token case (finish_reason='stop'), not length truncation "
                            "(finish_reason='length', handled separately via incomplete_details). "
                            "This indicates either a badly trained model requiring training-level fixes "
                            "or a bug in the inference engine."
                        )
                        termination_reason = "incomplete_reasoning" if output else "empty_output"
                        model_response.status = "incomplete"
                        model_response.metadata = {
                            **(model_response.metadata or {}),
                            "ng_termination_reason": termination_reason,
                            "ng_termination_message": termination_message,
                        }
                        LOG.warning(
                            "%s model_server=%s response_id=%s rollout_id=%s step=%s",
                            termination_message,
                            self.config.model_server.name,
                            model_response.id,
                            rollout_id,
                            step,
                        )
                    break

                if not execute_tools:
                    break

                pending_tools = True
                await commit_boundary()

            all_fn_calls = [o for o in model_response.output if o.type == "function_call"]
            for output_function_call in all_fn_calls:
                if collect_trajectory:
                    started_at = time()
                    started_monotonic = perf_counter()
                try:
                    parsed_arguments = json.loads(output_function_call.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    tool_output = json.dumps({"error": f"Invalid tool call arguments: {e!r}"})
                    if collect_trajectory:
                        error_type = type(e).__name__
                        tool_status = "failed"
                else:
                    # Resource-server errors are valid model-visible tool outputs.
                    if tool_access is not None:
                        api_response = await http_request(
                            method="POST",
                            url=f"{_tools_base_url(tool_access, resources_prefix)}/{output_function_call.name}",
                            json=parsed_arguments,
                            cookies=resources_server_cookies,
                            headers=dict(tool_access.headers),
                            _internal=True,
                        )
                    else:
                        if in_session:
                            # An Environment Server episode reaches Resources only through its grants; the
                            # configured resources_server has no session for this episode.
                            raise RuntimeError(
                                f"Model called tool {output_function_call.name!r}, but this agent session has no "
                                "direct HTTP tool access"
                            )
                        if self.config.resources_server is None:
                            raise RuntimeError(
                                "Simple Agent received a tool call without direct HTTP tool access "
                                "or a legacy resources_server configuration"
                            )
                        api_response = await self.server_client.post(
                            server_name=self.config.resources_server.name,
                            url_path=f"{resources_prefix}/{output_function_call.name}",
                            json=parsed_arguments,
                            cookies=resources_server_cookies,
                        )
                    tool_output = (await api_response.content.read()).decode()
                    resources_server_cookies = dict(resources_server_cookies or {})
                    resources_server_cookies.update(_cookies(api_response))
                    if collect_trajectory:
                        completed = 200 <= api_response.status < 400
                        tool_status = "completed" if completed else "failed"
                        error_type = None if completed else f"http_{api_response.status}"

                if collect_trajectory:
                    tool_records.append(
                        TrajectoryToolCall(
                            invocation_id=invocation_id,
                            tool_call_id=output_function_call.call_id,
                            tool_name=output_function_call.name,
                            started_at=started_at,
                            completed_at=max(started_at, time()),
                            duration_ms=(perf_counter() - started_monotonic) * 1000,
                            timing_source="executor",
                            status=tool_status,
                            error_type=error_type,
                            output=tool_output,
                        )
                    )

                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=output_function_call.call_id,
                        output=tool_output,
                    )
                )

            if collect_trajectory and all_fn_calls:
                turns[-1].step_count = len(tool_records)
            pending_tools = False

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                invocation_status = "incomplete"
                break
            await commit_boundary()

        model_response.output = new_outputs
        model_response.usage = usage
        trajectory = None
        if collect_trajectory:
            invocation = AgentInvocation(
                invocation_id=invocation_id,
                status=invocation_status,
                error_type=invocation_error_type,
                model_calls=model_calls,
                conversation=[*body.input, *new_outputs],
            )
            trajectory = TrajectoryRecord(
                task_id=task_id,
                rollout_id=rollout_id,
                invocations=[invocation],
                turns=turns,
                tool_calls=tool_records,
                gaps=trajectory_gaps,
            )
        return model_response, trajectory, model_server_cookies, resources_server_cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        collect_trajectory = self._model_call_capture_enabled() and isinstance(rollout_id, str)
        agent_session_id = self._agent_session_id_from_request(request)
        state = self._require_agent_session(agent_session_id) if agent_session_id is not None else None
        if state is not None and not isinstance(state, SimpleAgentSessionState):
            raise TypeError("Expected Simple Agent session state")
        if state is None and not self.config.execute_tools:
            raise ValueError(
                "Simple Agent execute_tools=false is supported only for agent-session requests; "
                "seed an agent session before calling /v1/responses"
            )
        participant = self.checkpoint_participant
        target = self._activation_target(state, rollout_id)
        index = None
        if target is None:
            # Without checkpointing nothing is tracked.
            # With it, a direct call is a restart.
            activation_scope = participant.restart_call(rollout_id) if participant is not None else nullcontext()
        else:
            session_key, episode_id, loop = target
            participant.check_session(session_key, episode_id)
            index = self._activation_index(request)
            while index is not None:
                if index == loop.completed_activation:
                    # The caller's checkpoint precedes the reply this activation already gave: give it again.
                    participant.answered_from_saved_reply(session_key, episode_id)
                    return self._saved_reply(loop, session_key)
                if index != loop.completed_activation + 1:
                    raise ActivationOutOfOrderError(
                        f"activation {index} of agent session {session_key!r} is out of order: "
                        f"activation {loop.completed_activation} is the last one completed"
                    )
                # A caller that sends the activation again while the first request still runs gets its reply.
                if not await participant.wait_activation(session_key, index):
                    break
            activation_scope = participant.activation(session_key, episode_id, index=index)
        async with activation_scope as activation:
            invocation_id = "root"
            if state is not None:
                # A session spans several activations, and its observations keep one invocation per activation.
                # The first keeps "root" so single-activation sessions report what they did before.
                # An activation continued from a restored boundary was already counted, so it keeps its ID.
                if activation is None or activation.continuation is None:
                    state.activations = index if index is not None else state.activations + 1
                if state.activations > 1:
                    invocation_id = f"activation-{state.activations}"
            model_response, trajectory, model_server_cookies, resources_server_cookies = await self._create_episode(
                body,
                model_url_path=self.url_path_for_request("/v1/responses", request),
                resources_server_cookies=state.resources_cookies if state is not None else request.cookies,
                tool_access=state.tool_access if state is not None else None,
                # With checkpointing on, the Resources Server refuses a tool call that does not name its attempt.
                # Each call names the attempt of the activation it serves,
                # which after a restore is the replacement, not the attempt that seeded the session.
                resources_prefix=rollout_path_prefix(rollout_id) if participant is not None else "",
                in_session=state is not None,
                execute_tools=self.config.execute_tools,
                invocation_id=invocation_id,
                rollout_id=rollout_id or "unscoped",
                collect_trajectory=collect_trajectory,
                activation=activation,
            )
            if state is not None:
                # Only a session with a direct tool grant holds the resources cookie jar.
                if state.tool_access is not None:
                    state.resources_cookies = dict(resources_server_cookies or {})
                if trajectory is not None:
                    # A session returns agent evidence at close, where the Environment Server records it.
                    previous = state.observations
                    state.observations = AgentObservationBundle(
                        source="simple_agent",
                        records=[
                            *(previous.records if previous else []),
                            *trajectory.invocations,
                            *trajectory.tool_calls,
                        ],
                        gaps=[*(previous.gaps if previous else []), *trajectory.gaps],
                    )

            # Legacy self-dispatch propagates resources cookies for its later verification call.
            if state is None:
                downstream_cookies = (*resources_server_cookies.items(), *model_server_cookies.items())
            else:
                downstream_cookies = (model_server_cookies or {}).items()
            for k, v in downstream_cookies:
                response.set_cookie(k, v)
            # A session caller that sends the activation index takes the session's evidence at close,
            # so its reply carries no trajectory, and neither does the saved reply.
            # Other callers, and a legacy /run, read the trajectory from the reply.
            if trajectory is not None and (state is None or index is None):
                model_response = model_response.model_copy(
                    update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")}
                )
            if target is not None:
                loop = target[2]
                # Recorded before the activation ends, so a checkpoint exports the session with its reply.
                if state is not None:
                    loop.completed_activation = state.activations
                else:
                    loop.completed_activation = index if index is not None else loop.completed_activation + 1
                # Only a caller that sends the index can ask for the reply again.
                loop.completed_reply = (
                    {"response": model_response, "cookies": _cookie_values(dict(downstream_cookies))}
                    if index is not None
                    else None
                )
        return model_response

    @staticmethod
    def _saved_reply(loop: SimpleAgentSessionState | _LegacyLoopReply, session_key: str) -> JSONResponse:
        """The saved reply of the session's last completed activation, without running anything again."""
        if loop.completed_reply is None:
            raise ActivationOutOfOrderError(f"agent session {session_key!r} has completed no activation to reply with")
        reply = JSONResponse(content=_to_json(loop.completed_reply["response"]))
        for name, value in loop.completed_reply["cookies"].items():
            reply.set_cookie(name, value)
        return reply

    @staticmethod
    async def _call_policy(activation: Activation | None, call: Callable[[], Awaitable[Any]]) -> Any:
        """Issue a policy call from the latest boundary; a checkpoint may hold its reply.

        ``call`` must read the whole reply, body included.
        The model server's checkpoint gate can send a held reply's headers and hold its body until resume,
        so a session that left its boundary on the headers alone would wait on the body and hold up prepare.
        """
        if activation is None:
            return await call()
        async with activation.awaiting_model():
            return await call()

    def _activation_target(
        self, state: SimpleAgentSessionState | None, capture_key: str | None
    ) -> tuple[str, EpisodeId, SimpleAgentSessionState | _LegacyLoopReply] | None:
        """The checkpointed session a request activates, the attempt it runs as, and where its reply is kept.

        ``None`` without checkpointing, and for a direct call, which has neither an agent session nor a legacy /run.
        With checkpointing, an activation of an agent session must name its attempt in its rollout path.
        """
        participant = self.checkpoint_participant
        if participant is None:
            return None
        path_episode = EpisodeId.from_capture_key(capture_key) if capture_key is not None else None
        if state is not None:
            # The session's own episode cannot stand in for a missing rollout path.
            # A restore rewrites it to the replacement attempt,
            # so a late call from the replaced attempt's harness, with only the session's cookie,
            # would pass as the replacement and could get its saved reply or start its next activation.
            if path_episode is None:
                raise RolloutIdRequiredError(
                    f"checkpointing requires the /ng-rollout/<id> prefix on every activation of agent session "
                    f"{state.request.agent_session_id!r}"
                )
            # The rollout path names the caller's attempt, which the session's must match.
            return state.request.agent_session_id, path_episode, state
        if path_episode is None:
            return None
        # Only a legacy /run opens this session.
        session_key = f"{_LEGACY_SESSION_PREFIX}{path_episode.rollout_id}"
        if not participant.has_session(session_key):
            return None
        return session_key, path_episode, self._legacy_replies.setdefault(session_key, _LegacyLoopReply())

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        if not self.config.execute_tools:
            raise ValueError(
                "Simple Agent execute_tools=false is supported only for agent-session requests; "
                "the legacy /run route requires execute_tools=true"
            )
        if self.config.resources_server is None:
            raise ValueError("resources_server is required when invoking the legacy Simple Agent /run route")
        participant = self.checkpoint_participant
        capture_key = self.rollout_id_from_run(body)
        if participant is None:
            return await self._run(request, body, legacy_run=None)
        episode_id = EpisodeId.from_capture_key(require_rollout(capture_key))
        session_key = f"{_LEGACY_SESSION_PREFIX}{episode_id.rollout_id}"
        async with participant.legacy_run(session_key, episode_id) as legacy_run:
            try:
                return await self._run(request, body, legacy_run=legacy_run)
            finally:
                # A /run that ends can no longer be continued at its loop step.
                self._legacy_replies.pop(session_key, None)

    async def _run(
        self, request: Request, body: SimpleAgentRunRequest, *, legacy_run: LegacyRun | None
    ) -> SimpleAgentVerifyResponse:
        # Steps: seed → turn loop → verify.
        # With checkpointing, a boundary before each step names the next step and what it needs,
        # so a replacement attempt resumes at that step.
        continuation = (legacy_run.continuation if legacy_run is not None else None) or {}
        stage = continuation.get("next", "seed")
        cookies: Any = continuation.get("cookies", request.cookies)
        # Whether the resources server's /verify may be replayed, as its seed reply reported.
        resources_verify_step_type: StepType = continuation.get("resources_verify_step_type", "wait")

        async def boundary(state: dict[str, Any]) -> None:
            if legacy_run is None:
                return
            loop = self._legacy_replies.get(legacy_run.key)
            if state["next"] in ("verify", "return") and loop is not None:
                # From verify on, this boundary holds the loop's reply, so the session no longer exports its own copy.
                # Dropped with no await before the boundary is recorded, so no export sees both or neither.
                # The completed activation stays, so a late duplicate of the loop request is refused, not run again.
                loop.completed_reply = None
            await legacy_run.boundary(state)

        def step(step_type: StepType) -> AbstractAsyncContextManager[None]:
            return legacy_run.step(step_type) if legacy_run is not None else nullcontext()

        if stage == "seed":
            await boundary({"next": "seed"})
            # A seed creates a new session for a new cookie, so running it again after a crash is safe.
            async with step("replay"):
                seed_session_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/seed_session",
                    json=body.model_dump(),
                    cookies=cookies,
                )
                await raise_for_status(seed_session_response)
            cookies = _cookie_values(seed_session_response.cookies)
            resources_verify_step_type = seed_verify_step_type(seed_session_response.headers)
            if legacy_run is not None and seed_restarts(seed_session_response.headers):
                # The resources server cannot capture its session, so this episode starts over after a crash.
                await legacy_run.mark_restart()
            stage = "loop"

        if stage == "loop":
            await boundary(
                {
                    "next": "loop",
                    "cookies": _cookie_values(cookies),
                    "resources_verify_step_type": resources_verify_step_type,
                }
            )
            # The turn loop parks at its own boundaries and continues from them after a restore.
            # It is the /run's one activation: a restore after its reply invokes it again and gets the same reply.
            # Its activation is tracked by this worker, which holds the episode, so the call comes back here.
            owner = getattr(request.app.state, "nemo_gym_routing_id", None) if legacy_run is not None else None
            headers = {AGENT_ACTIVATION_HEADER: "1"} if legacy_run is not None else {}
            if owner is not None:
                headers[SESSION_OWNER_HEADER] = owner
            async with step("replay"):
                response = await self.server_client.post(
                    server_name=self.config.name,
                    url_path=self.url_path_for_run("/v1/responses", body),
                    json=self._strip_resource_only_metadata(body.responses_create_params),
                    cookies=cookies,
                    headers=headers,
                )
                await raise_for_status(response)
                model_response_json = await get_response_json(response)
            cookies = _cookie_values(response.cookies)
        elif stage in ("verify", "return"):
            model_response_json = continuation["response"]

        trajectory = None
        expected_rollout_id = self.rollout_id_from_run(body)
        # The boundaries keep the response with its trajectory,
        # so a /run restored at verify or return still reports it.
        boundary_response_json = model_response_json
        model_response_json = dict(model_response_json)
        raw_trajectory = (
            model_response_json.pop(_INTERNAL_TRAJECTORY_KEY, None) if expected_rollout_id is not None else None
        )
        if isinstance(raw_trajectory, dict):
            trajectory = TrajectoryRecord.model_validate(raw_trajectory)
            extra = body.model_extra or {}
            task_id = next(
                (
                    str(extra[key])
                    for key in ("task_id", "problem_id", "instance_id", "_ng_task_index")
                    if extra.get(key) is not None
                ),
                "unknown",
            )
            rollout_id = expected_rollout_id or trajectory.rollout_id
            trajectory = trajectory.model_copy(
                update={
                    "task_id": task_id,
                    "rollout_id": rollout_id,
                    "turns": [
                        turn.model_copy(update={"task_id": task_id, "rollout_id": rollout_id})
                        for turn in trajectory.turns
                    ],
                }
            )

        if stage == "return":
            # A /run restored at its return step returns the result it recorded, whether or not it verified.
            result = continuation["result"]
        elif self.config.skip_verification:
            result = body.model_dump() | {
                "response": model_response_json,
                "reward": float(self.config.skip_verification_reward),
                "verification_skipped": True,
            }
            # The same boundary a verified result gets.
            # A checkpoint that closed while the turn loop's reply was on its way back parks the /run here,
            # so the commit still holds the episode, and a restore returns this result without running the loop again.
            await boundary(
                {
                    "next": "return",
                    "response": boundary_response_json,
                    "result": result,
                    "cookies": _cookie_values(cookies),
                    "resources_verify_step_type": resources_verify_step_type,
                }
            )
        else:
            await boundary(
                {
                    "next": "verify",
                    "response": boundary_response_json,
                    "cookies": _cookie_values(cookies),
                    "resources_verify_step_type": resources_verify_step_type,
                }
            )
            verify_payload = body.model_dump() | {"response": model_response_json}
            async with step(resources_verify_step_type):
                verify_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=verify_payload,
                    cookies=cookies,
                    # A checkpoint does not wait for a replayable verification, so it may keep the boundary before it.
                    # The resources server then keeps the session until the close below, so a restore can verify again.
                    headers=(
                        {CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"}
                        if self._closes_after_verify(legacy_run, resources_verify_step_type)
                        else {}
                    ),
                )
                await raise_for_status(verify_response)
                result = await get_response_json(verify_response)
            # Record the result so a restore never runs a state-changing verification twice.
            await boundary(
                {
                    "next": "return",
                    "response": boundary_response_json,
                    "result": result,
                    "cookies": _cookie_values(cookies),
                    "resources_verify_step_type": resources_verify_step_type,
                }
            )
        if self._closes_after_verify(legacy_run, resources_verify_step_type) and not self.config.skip_verification:
            await self._close_legacy_resources_session(body, cookies)
        if trajectory is not None:
            resolved = result.get("resolved")
            if isinstance(resolved, bool) and trajectory.turns:
                trajectory.turns[-1].resolved = resolved
            else:
                trajectory.gaps.append(ObservationGap(code="resolution_unavailable", invocation_id="root"))
            result["ng_trajectory"] = trajectory.model_dump(mode="json")
        return SimpleAgentVerifyResponse.model_validate(result)

    @staticmethod
    def _closes_after_verify(legacy_run: LegacyRun | None, resources_verify_step_type: StepType) -> bool:
        """Whether a legacy /run closes its resources session after a verification a checkpoint does not wait for."""
        return legacy_run is not None and resources_verify_step_type == "replay"

    async def _close_legacy_resources_session(self, body: SimpleAgentRunRequest, cookies: Any) -> None:
        """Close the resources session a legacy /run seeded, keeping the run's verdict if the close fails.

        The session cookie names the session.
        A legacy seed assigns no resources session ID, so the close names the /run's own session key.
        """
        episode_id = EpisodeId.from_capture_key(require_rollout(self.rollout_id_from_run(body)))
        try:
            close_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close_session",
                json=ResourcesCloseSessionRequest(
                    resources_session_id=f"{_LEGACY_SESSION_PREFIX}{episode_id.capture_key}", episode_id=episode_id
                ).model_dump(mode="json"),
                cookies=cookies,
            )
            await raise_for_status(close_response)
        except Exception:
            LOG.warning("could not close the resources session of %s", episode_id.capture_key, exc_info=True)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)
        if self.config.resources_server is None:
            raise ValueError("resources_server is required to proxy aggregate metrics")

        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


def _cookies(response: Any) -> dict[str, str]:
    return _cookie_values(response.cookies)


_ROLLOUT_PREFIX_SUFFIX = re.compile(rf"/{re.escape(ROLLOUT_PATH_PREFIX)}/[^/]+$")


def _tools_base_url(tool_access: DirectHTTPToolAccess, resources_prefix: str) -> str:
    """The grant's base URL, under ``resources_prefix`` in place of the rollout prefix the grant carries, if any."""
    base_url = str(tool_access.base_url).rstrip("/")
    if not resources_prefix:
        return base_url
    return f"{_ROLLOUT_PREFIX_SUFFIX.sub('', base_url)}{resources_prefix}"


if __name__ == "__main__":
    SimpleAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = SimpleAgent.run_webserver()  # noqa: F401
