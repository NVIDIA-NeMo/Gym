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

"""SciCode multi-step generation agent.

SciCode problems are solved one sub-step at a time: for each sub-step the agent builds a prompt
from the problem description, the code it generated for previous sub-steps, and the current
function header, calls the model, extracts the Python code, and accumulates it. The resources
server then runs each sub-step's tests against the accumulated per-step solutions.

The step loop runs in one of two places:

- ``/run`` (behind a ``legacy_agent`` relay) reads the problem from the row and sends the solutions,
  ``step_usage`` and ``token_usage_version`` to ``/verify`` as top-level fields.
- An agent session, seeded by an environment server such as ``single_agent_turn_legacy``, reads the
  problem from the row's ``agent_context`` and runs the loop inside one ``/v1/responses`` activation.
  The returned response carries the same three fields under ``scicode``, where the resources server's
  ``/verify`` reads them.

Both send the model the same requests. The benchmark metrics belong to the resources server; this
agent's ``/aggregate_metrics`` forwards to it, so both paths report the same metrics.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import to_jsonable_python
from step_utils import (
    OUT_OF_CONTEXT,
    PREFILLED_STEPS_CODE,
    extract_python_script,
    is_context_window_error,
    process_problem_steps,
)

from nemo_gym.agents import (
    ActivationConflictError,
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    OrderedOperationLedger,
    SimpleResponsesAPIAgent,
)
from nemo_gym.agents.utils.verify_call import masked_verify_failure, verify_with_transport_retry
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import (
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    ResourcesServerRef,
)
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
    NgFailureEvent,
    accumulate_response_usage,
)
from nemo_gym.data import PromptConfig, load_prompt_config
from nemo_gym.server_runtime import BaseRunRequest
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryTurn,
)


LOG = logging.getLogger(__name__)
TOKEN_USAGE_VERSION = 1
# The response field in which a session activation reports what /run sends /verify as top-level fields.
SCICODE_RESPONSE_KEY = "scicode"


class ScicodeAgentConfig(BaseResponsesAPIAgentConfig):
    """Configuration for the SciCode multi-step agent."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Per-sub-step user prompt template (PromptConfig YAML) the agent fills each step.
    prompt_fpath: str
    # Inject each sub-step's scientific background into the prompt context.
    with_background: bool = True


class ScicodeAgentRunRequest(BaseRunRequest):
    # extra="allow" so passthrough fields (e.g. uuid) survive to the resources server.
    model_config = ConfigDict(extra="allow")
    problem_id: str
    sub_steps: List[dict]
    required_dependencies: str
    # Benchmark-specific official code for steps that provide context but are not generated/scored.
    # Keys are canonical step numbers such as "13.6".
    prefilled_steps_code: Dict[str, str] = Field(default_factory=dict)


class ScicodeSubStepPrompt(BaseModel):
    """The prompt material of one sub-step. Test cases and reference code stay with the resources server."""

    model_config = ConfigDict(extra="forbid")

    step_number: str
    step_description_prompt: str
    step_background: str
    function_header: str
    return_line: str


class ScicodeAgentContext(BaseModel):
    """The ``agent_context`` values an agent session needs to build every sub-step's prompt."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str
    sub_steps: List[ScicodeSubStepPrompt]
    required_dependencies: str
    prefilled_steps_code: Dict[str, str] = Field(default_factory=dict)


@dataclass
class ScicodeSessionState(AgentSessionState):
    context: ScicodeAgentContext
    operations: OrderedOperationLedger[NeMoGymResponse] = field(default_factory=OrderedOperationLedger)
    # The activation's model calls, returned as agent observations when the session closes.
    invocation: Optional[AgentInvocation] = None
    gaps: List[ObservationGap] = field(default_factory=list)


@dataclass
class _StepTurn:
    """One generated sub-step: its prompt, the model's response and when the call started."""

    user_content: str
    response: NeMoGymResponse
    timestamp: float


@dataclass
class _SolvedProblem:
    """What the step loop produced for one problem."""

    solutions: Dict[str, str]
    step_usage: List[dict]
    # The last generated sub-step's response, with usage summed over every generated sub-step (None when unknown).
    response: dict
    turns: List[_StepTurn]
    out_of_context: bool


def _empty_response() -> dict:
    """Minimal NeMoGymResponse for the degenerate case where no sub-step was generated."""
    return NeMoGymResponse(
        id="scicode",
        created_at=0.0,
        model="",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump()


class ScicodeAgent(SimpleResponsesAPIAgent):
    """Agent that drives the SciCode per-sub-step generation + code-accumulation loop."""

    config: ScicodeAgentConfig
    agent_context_model = ScicodeAgentContext

    def model_post_init(self, context):
        self._prompt: PromptConfig = load_prompt_config(self.config.prompt_fpath)

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """One model call; in an agent session, the whole step loop for the session's problem."""
        session_id = self._agent_session_id_from_request(request)
        if session_id is not None:
            return await self._activate(request, session_id, body)

        model_response_json, model_cookies = await self._call_model(
            self.url_path_for_request("/v1/responses", request), body, request.cookies
        )
        for k, v in model_cookies.items():
            response.set_cookie(k, v)

        return NeMoGymResponse.model_validate(model_response_json)

    async def _call_model(
        self, url_path: str, body: NeMoGymResponseCreateParamsNonStreaming | Dict[str, Any], cookies: Any
    ) -> tuple[Dict[str, Any], Any]:
        """Make one model call; return the response JSON and the model server's cookies.

        A dict body is read as its JSON, as /run's call to this agent's /v1/responses sends it, so the request
        keeps exactly the fields that body sets.
        """
        if isinstance(body, dict):
            body = NeMoGymResponseCreateParamsNonStreaming.model_validate(to_jsonable_python(body))
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=url_path,
            json=body.model_dump(exclude_unset=True, exclude_none=True),
            cookies=cookies,
        )
        await raise_for_status(model_response)
        return await get_response_json(model_response), model_response.cookies

    def _step_request_params(self, params: NeMoGymResponseCreateParamsNonStreaming) -> Dict[str, Any]:
        """The task's request settings that every sub-step's model call carries, without its input."""
        response_create_params = params.model_dump(exclude_unset=True, exclude_none=True)
        response_create_params.pop("input", None)
        response_create_params["model"] = params.model or self.config.model_server.name
        response_create_params["tools"] = params.tools
        return response_create_params

    async def _solve(
        self,
        *,
        problem_id: str,
        sub_steps: List[dict],
        required_dependencies: str,
        prefilled_steps_code: Dict[str, str],
        params: NeMoGymResponseCreateParamsNonStreaming,
        generate: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    ) -> _SolvedProblem:
        """Generate code for each sub-step, accumulating prior code as context.

        ``generate`` makes one model call and returns the response JSON. A context-window error ends generation:
        that sub-step and every later one get the out-of-context sentinel.
        """
        total = len(sub_steps)
        previous_llm_code = [None] * total
        solutions: Dict[str, str] = {}
        out_of_context = False
        last_response_json = None
        zero_usage = NeMoGymResponseUsage.sum_from_list([])
        usage = zero_usage
        usage_complete = True
        step_usage = []
        turns: List[_StepTurn] = []
        response_create_params = self._step_request_params(params)

        for cur_step in range(total):
            step_record = {
                "step_number": sub_steps[cur_step]["step_number"],
                "status": "skipped",
                "usage": zero_usage.model_dump(),
            }
            step_usage.append(step_record)
            # Prefilled steps provide context for later steps but are not scored (no solution entry).
            step_number = sub_steps[cur_step]["step_number"]
            if step_number in prefilled_steps_code:
                previous_llm_code[cur_step] = prefilled_steps_code[step_number]
                step_record["status"] = "prefilled"
                continue
            if (problem_id, cur_step) in PREFILLED_STEPS_CODE:
                previous_llm_code[cur_step] = PREFILLED_STEPS_CODE[(problem_id, cur_step)]
                step_record["status"] = "prefilled"
                continue
            if out_of_context:
                solutions[f"{problem_id}.{cur_step + 1}"] = OUT_OF_CONTEXT
                continue

            problem_steps_str, next_step_str, previous_code_str = process_problem_steps(
                sub_steps, cur_step, previous_llm_code, self.config.with_background
            )
            previous_code = (
                f"{required_dependencies}\n{previous_code_str}\n"
                if previous_code_str
                else f"{required_dependencies}\n"
            )
            user_content = self._prompt.user.format(
                problem_steps_str=problem_steps_str, next_step_str=next_step_str, dependencies=required_dependencies
            )

            turn_timestamp = time()
            try:
                last_response_json = await generate(
                    {**response_create_params, "input": [{"role": "user", "content": user_content}]}
                )
            except Exception as error:
                if is_context_window_error(error):
                    LOG.warning("SciCode step %s: context window exceeded; failing remaining steps.", cur_step)
                    out_of_context = True
                    step_record["status"] = "context_window_exceeded"
                    solutions[f"{problem_id}.{cur_step + 1}"] = OUT_OF_CONTEXT
                    continue
                raise

            model_response = NeMoGymResponse.model_validate(last_response_json)
            turns.append(_StepTurn(user_content=user_content, response=model_response, timestamp=turn_timestamp))
            step_record["status"] = "generated"
            # The top-level response retains only the final step. Preserve stop
            # evidence for every generation, even when model-call capture is off.
            step_record["response_id"] = model_response.id
            step_record["response_status"] = last_response_json.get("status")
            step_record["incomplete_details"] = last_response_json.get("incomplete_details")
            step_record["error"] = last_response_json.get("error")
            step_record["usage"] = model_response.usage.model_dump() if model_response.usage is not None else None
            usage_complete = usage_complete and model_response.usage is not None
            usage = accumulate_response_usage(usage, model_response.usage)
            if (model_response.metadata or {}).get("nemo_gym_context_window_exceeded") == "true":
                step_record["status"] = "context_window_exceeded"
                out_of_context = True
                solutions[f"{problem_id}.{cur_step + 1}"] = OUT_OF_CONTEXT
                continue
            generation = model_response.output_text
            extracted = extract_python_script(generation)
            previous_llm_code[cur_step] = extracted
            solutions[f"{problem_id}.{cur_step + 1}"] = f"{previous_code}\n{extracted}"

        # /verify requires a response; record the last sub-step's generation (empty if none ran).
        response = last_response_json if last_response_json is not None else _empty_response()
        # Keep the final generation for inspection, but account for the whole problem.
        response["usage"] = usage.model_dump() if usage_complete else None
        return _SolvedProblem(
            solutions=solutions,
            step_usage=step_usage,
            response=response,
            turns=turns,
            out_of_context=out_of_context,
        )

    def _model_call_refs(self, response: NeMoGymResponse) -> List[ModelCallRef]:
        return [ModelCallRef(model_ref=self.config.model_server, response_id=response.id)] if response.id else []

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> ScicodeSessionState:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("SciCode agent does not support sandbox access")
        required = [access.name for access in self.effective_tool_accesses(body) if access.required]
        if required:
            raise AgentSeedRejectedError(f"SciCode agent does not support required tool access: {', '.join(required)}")
        context = self.parse_agent_context(body)
        if not isinstance(context, ScicodeAgentContext):
            raise TypeError("Expected the SciCode agent context")
        return ScicodeSessionState(request=body, context=context)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, ScicodeSessionState):
            raise TypeError("Expected SciCode session state")
        await state.operations.close(timeout=30.0)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(
                source=self.config.name,
                records=[state.invocation] if state.invocation is not None else [],
                gaps=list(state.gaps),
            ),
        )

    async def _activate(
        self, request: Request, session_id: str, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """Run the session's one activation, or join it when the caller sends the same activation again."""
        state = self._require_agent_session(session_id)
        if not isinstance(state, ScicodeSessionState):
            raise TypeError("Expected SciCode session state")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Activation requires the seeded session's rollout path")
        model_url_path = self.url_path_for_request("/v1/responses", request)
        try:
            return await state.operations.execute(
                index=1,
                request=body,
                operation=lambda: self._solve_session(state, body, model_url_path=model_url_path),
            )
        except ActivationConflictError as error:
            raise HTTPException(409, str(error)) from error

    async def _solve_session(
        self, state: ScicodeSessionState, params: NeMoGymResponseCreateParamsNonStreaming, *, model_url_path: str
    ) -> NeMoGymResponse:
        """The step loop for the session's problem; the response carries what /run sends /verify."""
        context = state.context
        model_cookies: Dict[str, Any] = {}

        async def generate(step_request: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal model_cookies
            response_json, model_cookies = await self._call_model(model_url_path, step_request, model_cookies)
            return response_json

        invocation = AgentInvocation(invocation_id="root", in_session=True)
        state.invocation = invocation
        try:
            solved = await self._solve(
                problem_id=context.problem_id,
                sub_steps=[step.model_dump() for step in context.sub_steps],
                required_dependencies=context.required_dependencies,
                prefilled_steps_code=context.prefilled_steps_code,
                params=params,
                generate=generate,
            )
        except BaseException as error:
            invocation.status = "failed" if isinstance(error, Exception) else "incomplete"
            invocation.error_type = type(error).__name__
            raise
        for turn in solved.turns:
            refs = self._model_call_refs(turn.response)
            if not refs:
                state.gaps.append(ObservationGap(code="model_call_reference_unavailable", invocation_id="root"))
            invocation.model_calls.extend(refs)
        # As /run's trajectory: prefilled or skipped sub-steps are not model decisions, so no invocation without a turn.
        invocation.status = "incomplete" if solved.out_of_context else "completed"
        if not solved.turns:
            state.invocation = None
        return NeMoGymResponse.model_validate(
            solved.response
            | {
                SCICODE_RESPONSE_KEY: {
                    "solutions": solved.solutions,
                    "step_usage": solved.step_usage,
                    "token_usage_version": TOKEN_USAGE_VERSION,
                }
            }
        )

    async def run(self, request: Request, body: ScicodeAgentRunRequest):
        """Generate code for each sub-step (accumulating prior code as context), then verify."""
        cookies = request.cookies

        async def generate(step_request: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal cookies
            gen_response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=step_request,
                cookies=cookies,
            )
            await raise_for_status(gen_response)
            cookies = gen_response.cookies
            return await gen_response.json()

        solved = await self._solve(
            problem_id=body.problem_id,
            sub_steps=body.sub_steps,
            required_dependencies=body.required_dependencies,
            prefilled_steps_code=body.prefilled_steps_code,
            params=body.responses_create_params,
            generate=generate,
        )

        verify_request_data = body.model_dump()
        verify_request_data["solutions"] = solved.solutions
        verify_request_data["response"] = solved.response
        verify_request_data["token_usage_version"] = TOKEN_USAGE_VERSION
        verify_request_data["step_usage"] = solved.step_usage

        async def verify() -> dict[str, Any]:
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request_data,
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            return await verify_response.json()

        try:
            result = await verify_with_transport_retry(
                verify, server_client=self.server_client, server_name=self.config.resources_server.name
            )
        except NgFailureEvent as failure:
            # The grader failed, not the model: keep the answer and mask the row with the typed failure.
            result = masked_verify_failure(verify_request_data, failure)
        rollout_id = self.rollout_id_from_run(body)
        if self._model_call_capture_enabled() and rollout_id is not None:
            result["ng_trajectory"] = self._trajectory(body.problem_id, rollout_id, solved).model_dump(mode="json")
        return result

    def _trajectory(self, task_id: str, rollout_id: str, solved: _SolvedProblem) -> TrajectoryRecord:
        """The legacy /run's trajectory: one invocation, and one turn per generated sub-step."""
        trajectory = TrajectoryRecord(task_id=task_id, rollout_id=rollout_id)
        invocation = AgentInvocation(invocation_id="root")
        for turn in solved.turns:
            refs = self._model_call_refs(turn.response)
            if not refs:
                trajectory.gaps.append(ObservationGap(code="model_call_reference_unavailable", invocation_id="root"))
            invocation.model_calls.extend(refs)
            trajectory.turns.append(
                TrajectoryTurn(
                    invocation_id="root",
                    task_id=trajectory.task_id,
                    rollout_id=trajectory.rollout_id,
                    turn_no=len(trajectory.turns) + 1,
                    timestamp=turn.timestamp,
                    question=[{"role": "user", "content": turn.user_content}],
                    answer=turn.response.output,
                    step_count=len(trajectory.turns) + 1,
                    model_calls=refs,
                )
            )
        # Prefilled/skipped subproblems are not model decisions.
        if trajectory.turns:
            invocation.status = "incomplete" if solved.out_of_context else "completed"
            trajectory.invocations.append(invocation)
        return trajectory

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Forward to the resources server, which owns the SciCode metrics on both paths."""
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    ScicodeAgent.run_webserver()
