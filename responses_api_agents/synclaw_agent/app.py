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
import contextlib
import json
import logging
import time
import uuid
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Annotated, Any, Literal

from aiohttp import ClientError, ClientResponseError
from fastapi import HTTPException, Request, Response
from pydantic import BeforeValidator, Field, PositiveInt, model_validator
from synclaw.rollout.budget import (
    ContextBudgetSnapshot,
    ContextTokenEstimator,
    budget_summary_fields,
    initialize_budget_feedback,
)
from synclaw.rollout.loop import (
    AssistantTurn,
    ControlledTermination,
    RolloutState,
    RolloutTask,
    ToolExecution,
    run_rollout,
)
from synclaw.rollout.policy import (
    DEFAULT_CONTEXT_CRITICAL_REMAINING_FRACTION,
    DEFAULT_CONTEXT_SAFETY_MARGIN_FRACTION,
    DEFAULT_CONTEXT_WARNING_REMAINING_FRACTION,
    DEFAULT_MAX_LOW_SIMILARITY_SUBMISSIONS,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TURNS,
    DEFAULT_SAME_PATHWAY_FINISH_AFTER,
    DEFAULT_STOP_CONDITION,
    DEFAULT_TURN_WARNING_REMAINING,
    PreparedToolCall,
    RolloutPolicyConfig,
    StopCondition,
    parse_stop_condition,
    task_budget_config,
    tool_argument_error,
)
from synclaw.runners.model_runner import (
    responses_finish_reason,
    responses_to_chat_message,
)
from synclaw.tools.contracts import (
    TOOL_NAMES,
    filter_responses_tool_schemas,
    normalize_excluded_tools,
    task_excluded_tools,
)

from nemo_gym.agents import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import NeMoGymAsyncOpenAI, ServerClient, get_response_json, raise_for_status
from nemo_gym.config import (
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    ResourcesServerRef,
    get_first_server_config_dict,
)
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseUsage, accumulate_response_usage
from nemo_gym.trajectory import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from resources_servers.synclaw.schema import (
    SYNCLAW_SUBMISSION_KEY,
    SynClawRunRequest,
    SynClawSeedSessionRequest,
    SynClawSeedSessionResponse,
    SynClawVerifyRequest,
    SynClawVerifyResponse,
)


logger = logging.getLogger(__name__)
_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"


class SynClawAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Each rollout has at most one model call in flight, so collection concurrency already
    # bounds model calls. Set this only when several collectors share one agent server.
    max_concurrent_model_requests: PositiveInt | None = Field(
        None, description="Cap on in-flight model calls across this server's rollouts; None means no cap."
    )

    temperature: float | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] = Field(
        "high",
        description=(
            "Responses `reasoning.effort`. Which values work depends on the endpoint: OpenAI's top effort is "
            "`xhigh`, while the patched vllm-router accepts low, medium, high and max."
        ),
    )
    max_turns: PositiveInt | None = DEFAULT_MAX_TURNS
    max_tool_calls: int = Field(DEFAULT_MAX_TOOL_CALLS, ge=0, description="Tool-call cap; 0 means unlimited.")
    max_tokens: PositiveInt | None = DEFAULT_MAX_TOKENS
    max_model_len: PositiveInt | None = Field(
        DEFAULT_MAX_MODEL_LEN,
        description=(
            "Context ceiling that tasks may lower. Must not exceed the endpoint's reported limit. "
            "null disables context clipping unless the task supplies a limit."
        ),
    )
    context_safety_margin_fraction: float = Field(DEFAULT_CONTEXT_SAFETY_MARGIN_FRACTION, ge=0, lt=1)
    budget_feedback: bool = True
    context_warning_remaining_fraction: float = Field(DEFAULT_CONTEXT_WARNING_REMAINING_FRACTION, ge=0, le=1)
    context_critical_remaining_fraction: float = Field(DEFAULT_CONTEXT_CRITICAL_REMAINING_FRACTION, ge=0, le=1)
    stop_condition: Annotated[StopCondition, BeforeValidator(parse_stop_condition)] = Field(
        DEFAULT_STOP_CONDITION, description="Run-level stopping rule: exact, model, or a similarity threshold."
    )
    turn_warning_remaining: int = DEFAULT_TURN_WARNING_REMAINING
    same_pathway_finish_after: int = DEFAULT_SAME_PATHWAY_FINISH_AFTER
    max_low_similarity_submissions: int | None = DEFAULT_MAX_LOW_SIMILARITY_SUBMISSIONS
    excluded_tools: frozenset[str] = frozenset()

    def rollout_policy(self) -> RolloutPolicyConfig:
        return RolloutPolicyConfig(
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            max_tokens=self.max_tokens,
            max_model_len=self.max_model_len,
            context_safety_margin_fraction=self.context_safety_margin_fraction,
            budget_feedback=self.budget_feedback,
            context_warning_remaining_fraction=self.context_warning_remaining_fraction,
            context_critical_remaining_fraction=self.context_critical_remaining_fraction,
            stop_condition=self.stop_condition,
            turn_warning_remaining=self.turn_warning_remaining,
            same_pathway_finish_after=self.same_pathway_finish_after,
            max_low_similarity_submissions=self.max_low_similarity_submissions,
            excluded_tools=self.excluded_tools,
        )

    @model_validator(mode="after")
    def validate_rollout_policy(self):
        self.rollout_policy()
        return self


class SynClawAgent(SimpleResponsesAPIAgent):
    config: SynClawAgentConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Shared by trajectories, held only during inference so tools can overlap it.
        limit = self.config.max_concurrent_model_requests
        self._model_semaphore = asyncio.Semaphore(limit) if limit is not None else None
        self._policy = self.config.rollout_policy()
        self._served_limit_lock = asyncio.Lock()
        self._served_limit_checked = False
        self._served_max_model_len: int | None = None

    def _excluded_tools(self, body: SynClawRunRequest) -> set[str]:
        try:
            return task_excluded_tools(body.model_extra or {}, normalize_excluded_tools(self.config.excluded_tools))
        except RuntimeError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def _task_policy(self, body: SynClawRunRequest) -> RolloutPolicyConfig:
        try:
            return task_budget_config(
                body.model_extra or {}, replace(self._policy, excluded_tools=frozenset(self._excluded_tools(body)))
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    async def _require_served_max_model_len(self, max_model_len: int | None) -> None:
        """Check the resolved task limit against cached endpoint metadata."""
        if max_model_len is None:
            return
        async with self._served_limit_lock:
            if not self._served_limit_checked:
                self._served_max_model_len = await self._fetch_served_max_model_len()
                self._served_limit_checked = True
        served = self._served_max_model_len
        if served is not None and served < max_model_len:
            raise HTTPException(
                status_code=422,
                detail=f"The model endpoint serves max_model_len={served}, but this task requires {max_model_len}. "
                f"Lower the task or agent max_model_len to at most {served}, or serve a larger context.",
            )

    async def _fetch_served_max_model_len(self) -> int | None:
        """The context limit the model endpoint reports in /v1/models, or None if it reports none.

        Raises:
            HTTPException: The endpoint could not be asked (502).
        """
        global_config = self.server_client.global_config_dict
        name = self.config.model_server.name
        model_config = get_first_server_config_dict(global_config, name) if name in global_config else {}
        # vllm_model and openai_model spell their endpoint settings differently.
        base_url = model_config.get("base_url") or model_config.get("openai_base_url")
        if not base_url:
            logger.warning("Model server %s has no base_url; cannot check its max_model_len", name)
            return None
        client = NeMoGymAsyncOpenAI(
            base_url=base_url if isinstance(base_url, str) else base_url[0],
            api_key=model_config.get("api_key") or model_config.get("openai_api_key") or "",
        )
        try:
            models = (await client.create_models()).get("data") or []
        except (ClientError, TimeoutError) as err:
            # Not cached by the caller, so the next rollout looks again; running unchecked could overflow the context.
            raise HTTPException(
                status_code=502, detail=f"Could not read {client.base_url}/models to check max_model_len: {err}"
            ) from err
        model_name = model_config.get("model") or model_config.get("openai_model")
        card = next((model for model in models if model.get("id") == model_name), models[0] if models else {})
        served = card.get("max_model_len")
        if not isinstance(served, int):
            logger.warning("%s/models reports no max_model_len; cannot check it", client.base_url)
            return None
        return served

    async def responses(
        self,
        request: Request,
        response: Response,
        body: SynClawRunRequest = Body(),
    ) -> NeMoGymResponse:
        config = self._task_policy(body)
        assert isinstance(body.responses_create_params.input, list)
        # SynClaw's rollout loop keeps chat-shaped `state.messages` for its policy and trace
        # events; the transport keeps the authoritative Responses history separately.
        initial_messages = tuple(
            {"role": message.role, "content": message.content} for message in body.responses_create_params.input
        )
        # exclude_none, not exclude_unset: the seed arrives over the wire from /seed_session
        # with `phase: null` already materialized.
        seed_input = body.responses_create_params.model_dump(mode="json", exclude_none=True)["input"]
        await self._require_served_max_model_len(config.max_model_len)

        rollout_id = request.path_params.get("rollout_id")
        trajectory = None
        if self._model_call_capture_enabled() and isinstance(rollout_id, str):
            trajectory = TrajectoryRecord(
                task_id=body.task_id,
                rollout_id=rollout_id,
                invocations=[AgentInvocation(invocation_id="root", in_session=False)],
            )
            # The shared HTTP boundary retains this evidence if execution raises.
            request.state.nemo_gym_agent_trajectory = trajectory
        started = time.perf_counter()
        transport = _GymTransport(
            trajectory=trajectory,
            server_client=self.server_client,
            model_server_name=self.config.model_server.name,
            model_url_path=self.url_path_for_request("/v1/responses", request),
            input_history=seed_input,
            max_tokens=config.max_tokens,
            max_model_len=config.max_model_len,
            temperature=self.config.temperature,
            reasoning_effort=self.config.reasoning_effort,
            cookies=request.cookies,
            model_semaphore=self._model_semaphore,
        )
        tool_schemas = filter_responses_tool_schemas(excluded_tools=config.excluded_tools)
        tool_executor = _GymToolExecutor(
            trajectory=trajectory,
            server_client=self.server_client,
            resources_server_name=self.config.resources_server.name,
            cookies=request.cookies,
        )
        try:
            rollout_result = await run_rollout(
                task=RolloutTask(
                    task_id=body.task_id,
                    target_smiles=body.target_smiles,
                    initial_messages=initial_messages,
                ),
                config=config,
                state=RolloutState(
                    messages=list(initial_messages),
                    events=[],
                    budget_feedback=initialize_budget_feedback(config),
                ),
                transport=transport,
                budget_provider=transport.budget_snapshot,
                tool_executor=tool_executor,
                tool_schemas=tool_schemas,
            )
        except BaseException as error:
            if trajectory is not None:
                invocation = trajectory.invocations[0]
                invocation.status = "failed" if isinstance(error, Exception) else "incomplete"
                invocation.error_type = type(error).__name__
            raise
        finally:
            if trajectory is not None:
                invocation = trajectory.invocations[0]
                invocation.duration_ms = (time.perf_counter() - started) * 1000
                invocation.conversation = deepcopy(transport.input_history)
        if trajectory is not None:
            trajectory.invocations[0].status = "incomplete" if rollout_result.partial else "completed"
            trajectory.invocations[0].stop_reason = rollout_result.reason

        model_response = NeMoGymResponse(
            object="response",
            id=f"resp_{uuid.uuid4().hex}",
            created_at=int(time.time()),
            model=transport.model or "",
            tool_choice="auto",
            # Record the per-task tool subset the model actually saw, so the rollout can be
            # exported for SFT without re-deriving schemas from a possibly newer SynClaw release.
            tools=[{"strict": None, **tool} for tool in tool_schemas],
            parallel_tool_calls=False,
            # Everything the transport appended after the seeded prompt: the model's output
            # items verbatim (reasoning included), tool results, policy-injected messages.
            output=transport.new_items,
            usage=transport.usage,
            status="incomplete" if rollout_result.partial else "completed",
            incomplete_details=(
                {"reason": "max_output_tokens"} if rollout_result.termination_kind == "max_tokens" else None
            ),
        )
        model_response = model_response.model_copy(
            update={
                SYNCLAW_SUBMISSION_KEY: rollout_result.submission,
                **budget_summary_fields(rollout_result.state.events),
                "last_request_output_tokens": transport.last_request_output_tokens,
            }
        )

        if trajectory is not None:
            model_response = model_response.model_copy(
                update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")}
            )
        for key, val in (*transport.cookies.items(), *tool_executor.cookies.items()):
            response.set_cookie(key, val)

        return model_response

    async def run(
        self,
        request: Request,
        body: SynClawRunRequest,
    ) -> SynClawVerifyResponse:
        config = self._task_policy(body)
        await self._require_served_max_model_len(config.max_model_len)
        cookies = request.cookies

        seed_session_body = SynClawSeedSessionRequest(
            system_prompt=body.system_prompt,
            question=body.question,
            config=config,
        )
        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=seed_session_body.model_dump(mode="json"),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        seed_session_json = await get_response_json(seed_session_response)
        cookies = seed_session_response.cookies

        body.responses_create_params = SynClawSeedSessionResponse.model_validate(seed_session_json)

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body,
            cookies=cookies,
        )
        await raise_for_status(response)
        model_response_json = await get_response_json(response)
        cookies = response.cookies
        raw_trajectory = model_response_json.pop(_INTERNAL_TRAJECTORY_KEY, None)
        trajectory = None
        if raw_trajectory is not None and self.rollout_id_from_run(body) is not None:
            trajectory = TrajectoryRecord.model_validate(raw_trajectory)
            request.state.nemo_gym_agent_trajectory = trajectory
            for invocation in trajectory.invocations:
                invocation.verification_skipped = self.config.skip_verification

        if self.config.skip_verification:
            result = body.model_dump() | {
                "response": model_response_json,
                "reward": float(self.config.skip_verification_reward),
                "verification_skipped": True,
                "evaluation_completed": False,
            }
        else:
            verify_request = SynClawVerifyRequest.model_validate(body.model_dump() | {"response": model_response_json})
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            result = await get_response_json(verify_response)

        if trajectory is not None:
            if trajectory.turns and not self.config.skip_verification:
                trajectory.turns[-1].resolved = result["exact_reconstruction"]
            result["ng_trajectory"] = trajectory.model_dump(mode="json")
        return SynClawVerifyResponse.model_validate(result)

    async def aggregate_metrics(
        self,
        body: AggregateMetricsRequest = Body(),
    ) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)

        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


class _GymTransport:
    """Drives a Gym model server's ``/v1/responses`` endpoint for SynClaw's rollout loop.

    SynClaw's loop hardcodes chat-shaped ``state.messages``, which cannot represent reasoning
    items. This transport therefore ignores the ``messages`` argument to ``complete()`` and
    owns ``input_history``: the Responses items actually sent upstream. That list grows from
    three sources -- ``append_user``, ``append_tool``, and each response's ``output`` echoed
    back verbatim -- so the model re-reads its own reasoning every turn, matching the
    convention in ``responses_api_agents/simple_agent``. ``state.messages`` is left to serve
    SynClaw's policy and trace events.
    """

    # The Responses path reports truncation in-band as status="incomplete", but
    # openai_model's native passthrough still raises upstream context-length errors.
    _TOKEN_LIMIT_STRINGS = frozenset(
        {"context length", "ContextWindowExceededError", "combined input and output tokens"}
    )

    def __init__(
        self,
        server_client: ServerClient,
        model_server_name: str,
        *,
        model_url_path: str = "/v1/responses",
        input_history: Sequence[dict[str, Any]] = (),
        max_tokens: int | None = DEFAULT_MAX_TOKENS,
        max_model_len: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        cookies: dict[str, str] | None = None,
        model_semaphore: asyncio.Semaphore | None = None,
        trajectory: TrajectoryRecord | None = None,
    ):
        self.cookies = cookies or {}
        self.model_semaphore = model_semaphore
        self.trajectory = trajectory
        self.server_client = server_client
        self.model_server_name = model_server_name
        self.model_url_path = model_url_path
        self.max_tokens = max_tokens
        self.last_request_output_tokens: int | None = None
        self.max_model_len = max_model_len
        self.budget_estimator = ContextTokenEstimator()
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.input_history: list[dict[str, Any]] = [dict(item) for item in input_history]
        self._seed_length = len(self.input_history)
        self.usage: NeMoGymResponseUsage | None = None
        self.model: str | None = None

    @property
    def new_items(self) -> list[dict[str, Any]]:
        """Items appended after the seeded prompt, in chronological order."""
        return self.input_history[self._seed_length :]

    def _budget_payload(self, tools: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {"input": list(self.input_history), "tools": [{"strict": None, **tool} for tool in tools]}

    def budget_snapshot(self, messages: Any, tools: Sequence[dict[str, Any]]) -> ContextBudgetSnapshot:
        return self.budget_estimator.snapshot(
            self._budget_payload(tools),
            context_window_tokens=self.max_model_len,
            max_output_tokens=self.max_tokens,
        )

    def replace_last_input(self, message: dict[str, Any]) -> None:
        # Amend pending input in place so budget/repair feedback preserves the tool turn
        # and all preceding native reasoning items.
        last = self.input_history[-1] if self.input_history else {}
        if (
            message["role"] == "tool"
            and last.get("type") == "function_call_output"
            and last.get("call_id") == message.get("tool_call_id")
        ):
            self.input_history[-1] = {**last, "output": message["content"]}
            self._observe_tool_output(last["call_id"], message["content"])
        elif message["role"] == last.get("role") == "user":
            self.input_history[-1] = {**last, "content": message["content"]}
        else:
            raise ValueError("Harness feedback must update the last unsent input")

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]],
        tool_choice: Any,
        max_output_tokens: int | None,
    ) -> AssistantTurn:
        del messages  # `input_history` is authoritative; see the class docstring.

        # The loop builds tool_choice for the chat dialect; Responses wants it flattened.
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function") or {}
            tool_choice = {"type": "function", "name": function.get("name") or tool_choice.get("name")}

        budget_payload = self._budget_payload(tools)
        payload: dict[str, Any] = {
            # FunctionToolParam requires `strict`, which SynClaw's Responses schemas omit. None
            # rather than True: several schemas leave optional properties out of `required`,
            # which strict mode rejects upstream.
            **budget_payload,
            "tool_choice": tool_choice,
            "max_output_tokens": max_output_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        async with self.model_semaphore or contextlib.nullcontext():
            self.last_request_output_tokens = max_output_tokens
            turn_started = time.time()
            model_response = await self.server_client.post(
                server_name=self.model_server_name,
                url_path=self.model_url_path,
                json=payload,
                cookies=self.cookies,
                **({"headers": {"x-session-id": "root"}} if self.trajectory is not None else {}),
            )
            try:
                await raise_for_status(model_response)
            except ClientResponseError as err:
                body = (err.response_content or b"").decode(errors="replace")
                if any(substr in body for substr in self._TOKEN_LIMIT_STRINGS):
                    raise ControlledTermination("context_window_exceeded", body) from err
                else:
                    raise

            model_response_json = await get_response_json(model_response)
        self.cookies = model_response.cookies

        # Summed per turn as in simple_agent. A turn that omits a detail count (e.g. reasoning
        # tokens from vLLM < 0.28) leaves that total null rather than silently undercounted.
        if model_response_json.get("usage"):
            self.usage = accumulate_response_usage(
                self.usage, NeMoGymResponseUsage.model_validate(model_response_json["usage"])
            )

        self.model = model_response_json.get("model") or self.model
        output = [item for item in (model_response_json.get("output") or []) if isinstance(item, dict)]
        self.budget_estimator.observe(budget_payload, model_response_json.get("usage"), replay_output=output)
        if self.trajectory is not None:
            refs = []
            if model_response_json.get("id"):
                ref = ModelCallRef(
                    model_ref=ModelServerRef(type="responses_api_models", name=self.model_server_name),
                    response_id=model_response_json["id"],
                )
                refs.append(ref)
                self.trajectory.invocations[0].model_calls.append(ref)
            else:
                self.trajectory.gaps.append(
                    ObservationGap(code="model_call_reference_unavailable", invocation_id="root")
                )
            self.trajectory.turns.append(
                TrajectoryTurn(
                    invocation_id="root",
                    task_id=self.trajectory.task_id,
                    rollout_id=self.trajectory.rollout_id,
                    turn_no=len(self.trajectory.turns) + 1,
                    timestamp=turn_started,
                    question=deepcopy(payload["input"]),
                    answer=deepcopy([item for item in output if item.get("type") != "reasoning"]),
                    reasoning_content=deepcopy([item for item in output if item.get("type") == "reasoning"]) or None,
                    step_count=len(self.trajectory.tool_calls),
                    model_calls=refs,
                )
            )
        self.input_history.extend(output)
        return AssistantTurn(
            message=responses_to_chat_message(model_response_json),
            finish_reason=responses_finish_reason(model_response_json),
        )

    def append_user(self, message: dict[str, Any]) -> None:
        # The loop appends untagged {"role", "content"} dicts. NeMoGymResponse.output rejects an
        # item with no `type` discriminator, and this list is returned as the response output.
        self.input_history.append({**message, "type": "message"})

    def _has_function_call(self, call_id: str) -> bool:
        return any(
            item.get("type") == "function_call" and item.get("call_id") == call_id for item in self.input_history
        )

    def append_tool(self, call: PreparedToolCall, result_json: str) -> None:
        if not self._has_function_call(call.call_id):
            # No function_call was echoed for this id -- a Harmony-parsed text call, or SynClaw's
            # "tool-call-{n}" fallback when the provider omits one. An unpaired
            # function_call_output downconverts to a chat `tool` message with no preceding call,
            # which breaks every later turn, so materialize the call SynClaw actually dispatched.
            self.input_history.append(
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": json.dumps(call.dispatch_arguments, separators=(",", ":"), sort_keys=True),
                }
            )
        self.input_history.append({"type": "function_call_output", "call_id": call.call_id, "output": result_json})
        self._observe_tool_output(call.call_id, result_json)

    def _observe_tool_output(self, call_id: str, result_json: str) -> None:
        if self.trajectory is not None:
            # Compaction and harness feedback can amend the model-visible result.
            for observed in reversed(self.trajectory.tool_calls):
                if observed.tool_call_id == call_id:
                    observed.output = result_json
                    break


class _GymToolExecutor:
    def __init__(
        self,
        server_client: ServerClient,
        resources_server_name: str,
        *,
        cookies: dict[str, str] | None = None,
        trajectory: TrajectoryRecord | None = None,
    ):
        self.cookies = cookies or {}
        self.trajectory = trajectory
        self.server_client = server_client
        self.resources_server_name = resources_server_name

    async def execute(
        self,
        calls: Sequence[PreparedToolCall],
        *,
        target_smiles: str,
        assistant_turn: int,
    ) -> Sequence[ToolExecution]:
        del assistant_turn
        executions = []
        for call in calls:
            observed = None
            started = time.perf_counter()
            if self.trajectory is not None:
                observed = TrajectoryToolCall(
                    invocation_id="root",
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                    started_at=time.time(),
                    timing_source="executor",
                    status="incomplete",
                )
                self.trajectory.tool_calls.append(observed)
            try:
                result = (
                    tool_argument_error(call.name, call.argument_error, call.arguments)
                    if call.argument_error is not None
                    else await self._dispatch_tool_call(
                        call.name, call.dispatch_arguments, target_smiles=target_smiles
                    )
                )
            except BaseException as error:
                if observed is not None:
                    observed.status = "failed" if isinstance(error, Exception) else "cancelled"
                    observed.error_type = type(error).__name__
                raise
            else:
                if observed is not None:
                    observed.status = "failed" if result.get("error") else "completed"
                    observed.error_type = str(result["error"]) if result.get("error") else None
                    observed.output = json.dumps(result, sort_keys=True, separators=(",", ":"))
                executions.append(ToolExecution(result))
            finally:
                if observed is not None:
                    observed.completed_at = max(observed.started_at, time.time())
                    observed.duration_ms = (time.perf_counter() - started) * 1000
                    if self.trajectory.turns:
                        self.trajectory.turns[-1].step_count = len(self.trajectory.tool_calls)
        return executions

    async def _dispatch_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        target_smiles: str,
    ) -> dict[str, Any]:
        if name not in TOOL_NAMES:
            return tool_argument_error(name, f"Unknown tool name: {name}", arguments)
        parsed_arguments = {
            **arguments,
            "extra_tool_call_arguments": {
                "target_smiles": target_smiles,
            },
        }
        api_response = await self.server_client.post(
            server_name=self.resources_server_name,
            url_path=f"/{name}",
            json=parsed_arguments,
            cookies=self.cookies,
        )
        await raise_for_status(api_response)
        tool_output = await get_response_json(api_response)
        self.cookies = api_response.cookies
        return tool_output


if __name__ == "__main__":
    SynClawAgent.run_webserver()
