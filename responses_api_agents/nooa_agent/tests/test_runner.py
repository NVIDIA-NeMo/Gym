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
import json
from http.cookies import SimpleCookie
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from nooa import Agent, strategy
from nooa.errors import GenerationError, LoopDetectedError

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from nemo_gym.rollout_observability import AgentInvocation, ToolCallObservation
from nemo_gym.tool_access import DirectHTTPToolAccess
from responses_api_agents.nooa_agent import resource_tools
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import (
    InProcessNOOARunner,
    NOOARunFailure,
    NOOARunRequest,
)
from responses_api_agents.nooa_agent.sandbox_entrypoint import SandboxCheckpoint, write_checkpoint
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response
from responses_api_agents.nooa_agent.tests.test_resource_tools import FakeResponse as FakeResourceResponse


@pytest.fixture(autouse=True)
def outbound(monkeypatch):
    mock = AsyncMock(return_value=FakeResponse())
    monkeypatch.setattr(resource_tools, "request", mock)
    return mock


def grant() -> DirectHTTPToolAccess:
    return DirectHTTPToolAccess(name="resources", required=True, base_url="http://resources/")


class ValidAgent(Agent):
    def __init__(self, *, llm: Any, label: str) -> None:
        super().__init__(llm=llm)
        self.label = label

    async def analyze(self, text: str, customer_id: str) -> str: ...


class PolicyAgent(Agent):
    async def analyze(self, text: str) -> str:
        return await self.primary(text)

    async def primary(self, text: str) -> str:
        """Answer the question."""
        ...


class MethodLLMOverrideAgent(Agent):
    @strategy(llm="helper")
    async def analyze(self, text: str) -> str:
        """Answer the question."""
        ...


class BudgetAgent(PolicyAgent):
    async def analyze(self, text: str) -> str:
        await self.primary(text)
        await self.primary(text)
        return await self.primary(text)


class FailingAgent(PolicyAgent):
    async def analyze(self, text: str) -> str:
        await self.primary(text)
        raise RuntimeError("failed after a model call")


class WaitingAgent(PolicyAgent):
    ready: Any = None

    async def analyze(self, text: str) -> str:
        await self.primary(text)
        self.ready.set()
        await asyncio.Event().wait()
        return "unreachable"


class ResourceUsingAgent(Agent):
    async def analyze(self, text: str) -> str:
        """Answer the question using the available resource methods."""
        ...


class FakeAgent:
    instances = 0
    get_weather: Any

    def __init__(self, *, llm: Any, label: str) -> None:
        FakeAgent.instances += 1
        self.llm = llm
        self.label = label
        self.event_manager = FakeEventManager()

    async def analyze(self, text: str, customer_id: str) -> str:
        weather = await self.get_weather(city=customer_id)
        return f"{text}: {weather['weather']}"


adapter_requests: list[NeMoGymResponseCreateParamsNonStreaming] = []


class FakeEventManager:
    def on(self, event_type: str, handler: Any) -> Any:
        return lambda: None


async def invoke(agent: Any, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    adapter_requests.append(request)
    assert isinstance(request.input, str)
    text, customer_id = request.input.split("|", maxsplit=1)
    return await agent.analyze(text, customer_id)


class FakeContent:
    async def read(self) -> bytes:
        return json.dumps({"weather": "cold"}).encode()


class FakeResponse:
    ok = True
    status = 200
    content = FakeContent()
    cookies = SimpleCookie()


def make_runner(
    *, execution_mode: str = "embedded", context_window: int | None = None
) -> tuple[InProcessNOOARunner, MagicMock]:
    invocation = NOOAInvocationConfig.model_validate(
        {
            "agent_class": f"{__name__}:ValidAgent",
            "invocation_adapter": f"{__name__}:invoke",
            "execution_mode": execution_mode,
            "init_kwargs": {"label": "configured"},
        }
    )
    client = MagicMock()
    client.post = resource_tools.request
    runner = InProcessNOOARunner(
        invocation=invocation,
        server_client=client,
        model_server_name="policy_model",
        max_policy_calls=3,
        context_window=context_window,
    )
    runner._agent_class = FakeAgent
    return runner, client


@pytest.mark.asyncio
async def test_runner_passes_context_window_and_request_reply_cap_to_nooa() -> None:
    runner, client = make_runner(context_window=262144)

    async def inspect_limits(agent, request):
        limits = agent.llm.get_context_limits(fallback_reserve=4096)
        assert limits.context_window == 262144
        assert limits.reserved_output_tokens == 32768
        assert limits.usable_input_tokens == 229376
        return "limits checked"

    runner._invocation_adapter = inspect_limits
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hello", max_output_tokens=32768),
            model_url_path="/v1/responses",
        )
    )
    assert result.return_value == "limits checked"
    client.post.assert_not_called()


def responses_create_params(customer_id: str) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": f"Check delivery|{customer_id}",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_embedded_runner_invokes_adapter_and_attaches_resource_methods() -> None:
    runner, client = make_runner()
    request = responses_create_params("Paris")
    adapter_requests.clear()

    result = await runner.run(
        NOOARunRequest(
            tool_access=grant(),
            responses_create_params=request,
            model_url_path="/ng-rollout/rollout-1/v1/responses",
            resource_cookies={"session": "one"},
        )
    )

    assert [item.type for item in result.episode.response.output] == ["function_call", "function_call_output"]
    assert json.loads(result.episode.response.output[-1].output) == {"weather": "cold"}
    assert result.return_value == "Check delivery: cold"
    assert result.episode.observations.source == "nooa"
    assert result.episode.observations.gaps == []
    invocation = next(record for record in result.episode.observations.records if isinstance(record, AgentInvocation))
    tool = next(record for record in result.episode.observations.records if isinstance(record, ToolCallObservation))
    assert invocation.conversation[0].content == "Check delivery|Paris"
    assert tool.tool_name == "get_weather"
    assert adapter_requests == [request]
    assert client.post.await_args.kwargs["json"] == {"city": "Paris"}


@pytest.mark.asyncio
async def test_constructs_a_fresh_agent_for_every_rollout() -> None:
    runner, _ = make_runner()
    FakeAgent.instances = 0

    first = await runner.run(
        NOOARunRequest(
            tool_access=grant(),
            responses_create_params=responses_create_params("Paris"),
            model_url_path="/one/v1/responses",
        )
    )
    second = await runner.run(
        NOOARunRequest(
            tool_access=grant(),
            responses_create_params=responses_create_params("Berlin"),
            model_url_path="/two/v1/responses",
        )
    )

    assert FakeAgent.instances == 2
    assert first.episode is not second.episode
    assert first.resource_cookies is not second.resource_cookies


def test_runner_can_execute_inside_sandbox_process() -> None:
    make_runner(execution_mode="sandboxed")


def test_method_level_llm_override_fails_during_runner_construction() -> None:
    with pytest.raises(ValueError, match=r"method-level LLM overrides.*analyze"):
        policy_runner(MethodLLMOverrideAgent)


async def invoke_policy(agent: Any, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    if isinstance(request.input, str):
        text = request.input
    else:
        content = request.input[-1].content
        assert isinstance(content, str)
        text = content
    return await agent.analyze(text)


def policy_runner(agent_class: type[Agent] = PolicyAgent) -> tuple[InProcessNOOARunner, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    async def post(*, json: dict[str, Any], **_: Any) -> FakeHTTPResponse:
        calls.append(json)
        output = NeMoGymResponseFunctionToolCall(
            id=f"fc-{len(calls)}",
            call_id=f"call-{len(calls)}",
            name="return_result",
            arguments='{"result":"done"}',
        )
        return FakeHTTPResponse(model_response(output, response_id=f"response-{len(calls)}"))

    client = MagicMock()
    client.post = AsyncMock(side_effect=post)
    invocation = NOOAInvocationConfig(
        agent_class=f"{__name__}:{agent_class.__name__}",
        invocation_adapter=f"{__name__}:invoke_policy",
    )
    return InProcessNOOARunner(
        invocation=invocation,
        server_client=client,
        model_server_name="primary_model",
        max_policy_calls=3,
    ), calls


@pytest.mark.asyncio
async def test_real_strategy_budget_failure_keeps_completed_calls() -> None:
    runner, calls = policy_runner(BudgetAgent)
    runner._max_policy_calls = 2
    result = await runner.run(
        NOOARunRequest(
            tool_access=grant(),
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
            model_url_path="/v1/responses",
        )
    )
    assert result.termination_reason is NOOATerminationReason.POLICY_BUDGET_EXCEEDED
    assert len(calls) == len(result.trajectory.turns) == 2
    assert any(invocation.status == "incomplete" for invocation in result.trajectory.invocations)


@pytest.mark.asyncio
async def test_checkpoint_failure_logs_error_without_failing_the_run(caplog) -> None:
    runner, calls = policy_runner()

    def disk_full(result: object) -> None:
        raise OSError("disk full")

    runner._on_progress = disk_full
    with caplog.at_level("ERROR", logger="responses_api_agents.nooa_agent.runner"):
        result = await runner.run(
            NOOARunRequest(
                tool_access=grant(),
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
                model_url_path="/v1/responses",
            )
        )
    assert result.return_value == "done"
    assert result.termination_reason is None
    assert len(calls) == len(result.trajectory.turns) == 1
    failures = [record for record in caplog.records if record.message == "Failed to persist NOOA partial checkpoint"]
    assert failures
    assert all(
        isinstance(record.exc_info[1], OSError) and str(record.exc_info[1]) == "disk full" for record in failures
    )


@pytest.mark.asyncio
async def test_unexpected_agent_failure_carries_the_partial_episode() -> None:
    runner, calls = policy_runner(FailingAgent)
    with pytest.raises(NOOARunFailure, match="failed after a model call") as error:
        await runner.run(
            NOOARunRequest(
                tool_access=grant(),
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
                model_url_path="/v1/responses",
            )
        )
    assert len(calls) == len(error.value.result.trajectory.turns) == 1
    assert error.value.result.episode.response.output


@pytest.mark.asyncio
async def test_cancellation_preserves_evidence_without_swallowing_cancellation() -> None:
    runner, _ = policy_runner(WaitingAgent)
    WaitingAgent.ready = asyncio.Event()
    task = asyncio.create_task(
        runner.run(
            NOOARunRequest(
                tool_access=grant(),
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
                model_url_path="/v1/responses",
            )
        )
    )
    try:
        await asyncio.wait_for(WaitingAgent.ready.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        assert len(error.value.nooa_result.trajectory.turns) == 1
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        WaitingAgent.ready = None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200])
async def test_real_code_and_resource_outputs_join_to_their_own_invocations(status: int, tmp_path) -> None:
    runner, _ = policy_runner(ResourceUsingAgent)
    checkpoints = []

    def checkpoint(result):
        assert result.return_value is None
        write_checkpoint(tmp_path / "checkpoint.json", result)
        checkpoints.append(SandboxCheckpoint.model_validate_json((tmp_path / "checkpoint.json").read_text()))

    runner._on_progress = checkpoint
    model_requests = []
    code_arguments = json.dumps({"code": "print(await self.get_weather(city='Paris'))"})

    async def post(*, server_name: str, json: Any, **kwargs: Any) -> FakeHTTPResponse:
        if server_name == "resources":
            response = FakeHTTPResponse({"weather": "cold"} if status == 200 else {"error": "unavailable"})
            response.status = status
            return response
        model_requests.append(json)
        first = len(model_requests) == 1
        output = NeMoGymResponseFunctionToolCall(
            id=f"fc-{len(model_requests)}",
            call_id=f"call-{len(model_requests)}",
            name="execute_python" if first else "return_result",
            arguments=code_arguments if first else '{"result":"done"}',
        )
        return FakeHTTPResponse(model_response(output, response_id=f"response-{len(model_requests)}"))

    runner._server_client.post = AsyncMock(side_effect=post)

    async def resource_request(method: str, url: str, **kwargs: Any) -> FakeHTTPResponse:
        return await post(server_name="resources", **kwargs)

    resource_tools.request.side_effect = resource_request
    result = await runner.run(
        NOOARunRequest(
            tool_access=grant(),
            responses_create_params=responses_create_params("Paris"),
            model_url_path="/v1/responses",
        )
    )
    assert result.return_value == "done"
    # The NOOA code prefill, model response, resource completion within live
    # code, code completion, final model response, and return_result completion.
    assert [len(item.trajectory.turns) for item in checkpoints] == [0, 1, 1, 1, 2, 2]
    assert len(checkpoints[0].trajectory.tool_calls) > 0
    pending_code = next(tool for tool in checkpoints[2].trajectory.tool_calls if tool.tool_call_id == "call-1")
    assert pending_code.tool_name == "execute_python"
    assert pending_code.status == "incomplete"
    assert pending_code.output is None
    assert not any(
        item.type == "function_call_output" and item.call_id == pending_code.tool_call_id
        for item in checkpoints[2].response.output
    )
    completed_code = next(tool for tool in checkpoints[3].trajectory.tool_calls if tool.tool_call_id == "call-1")
    assert completed_code.status == "completed"
    assert all(item.partial and item.response.status == "incomplete" for item in checkpoints)
    assert len(model_requests) == 2
    code = next(tool for tool in result.trajectory.tool_calls if tool.tool_call_id == "call-1")
    resource = next(tool for tool in result.trajectory.tool_calls if tool.tool_name == "get_weather")
    assert code.tool_call_id == "call-1"
    assert resource.status == ("completed" if status == 200 else "failed")
    observed = next(
        item.output
        for item in model_requests[1].input
        if item.type == "function_call_output" and item.call_id == "call-1"
    )
    assert observed == "status: accepted"
    assert "PythonOutput(" in code.output and "cold" in code.output
    assert any(
        getattr(item, "role", None) == "user" and getattr(item, "content", None) == code.output
        for item in model_requests[1].input
    )
    assert json.loads(resource.output) == ({"weather": "cold"} if status == 200 else {"error": "unavailable"})
    assert all(tool.tool_name != "return_result" for tool in result.trajectory.tool_calls)
    model_owner = next(inv for inv in result.trajectory.invocations if inv.invocation_id == code.invocation_id)
    assert model_owner.conversation[0].role == "system"
    for tool in (code, resource):
        owner = next(
            invocation
            for invocation in result.trajectory.invocations
            if invocation.invocation_id == tool.invocation_id
        )
        assert any(
            getattr(item, "call_id", None) == tool.tool_call_id and item.type == "function_call_output"
            for item in owner.conversation
        )


@pytest.mark.asyncio
async def test_real_codeact_recovers_from_resource_http_error(outbound: AsyncMock) -> None:
    runner, _ = policy_runner(ResourceUsingAgent)
    model_requests: list[NeMoGymResponseCreateParamsNonStreaming] = []
    feedback = "Unknown city Pariss; use Paris."
    outbound.side_effect = [
        FakeResourceResponse({"error": feedback}, status=400, cookie=("tool_state", "rejected")),
        FakeResourceResponse({"weather": "cold"}, cookie=("tool_state", "accepted")),
    ]
    replies = [
        ("execute_python", json.dumps({"code": "print(await self.get_weather(city='Pariss'))"})),
        ("execute_python", json.dumps({"code": "print(await self.get_weather(city='Paris'))"})),
        ("return_result", '{"result":"cold"}'),
    ]

    async def post(*, json: NeMoGymResponseCreateParamsNonStreaming, **_: Any) -> FakeHTTPResponse:
        name, arguments = replies[len(model_requests)]
        model_requests.append(json)
        output = NeMoGymResponseFunctionToolCall(
            id=f"fc-{len(model_requests)}",
            call_id=f"call-{len(model_requests)}",
            name=name,
            arguments=arguments,
        )
        return FakeHTTPResponse(model_response(output, response_id=f"response-{len(model_requests)}"))

    runner._server_client.post = AsyncMock(side_effect=post)
    result = await runner.run(
        NOOARunRequest(
            tool_access=grant(),
            responses_create_params=responses_create_params("Paris"),
            model_url_path="/v1/responses",
            resource_cookies={"session": "one"},
        )
    )

    assert result.return_value == "cold"
    assert result.termination_reason is None
    assert len(model_requests) == len(result.trajectory.turns) == 3
    for request, expected in (
        (model_requests[1], feedback),
        (model_requests[2], "cold"),
    ):
        assert expected in request.model_dump_json(include={"input"})
    assert [call.kwargs["json"] for call in outbound.await_args_list] == [{"city": "Pariss"}, {"city": "Paris"}]
    assert [call.kwargs["cookies"] for call in outbound.await_args_list] == [
        {"session": "one"},
        {"session": "one", "tool_state": "rejected"},
    ]
    assert result.resource_cookies == {"session": "one", "tool_state": "accepted"}
    resources = [tool for tool in result.trajectory.tool_calls if tool.tool_name == "get_weather"]
    assert [tool.status for tool in resources] == ["failed", "completed"]
    assert [tool.error_type for tool in resources] == ["http_400", None]
    assert [json.loads(tool.output) for tool in resources] == [{"error": feedback}, {"weather": "cold"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["model", "resource"])
async def test_swallowed_infrastructure_failure_never_becomes_gradable(source: str) -> None:
    runner, _ = make_runner()
    failure = ConnectionError("dependency unavailable")

    async def swallow(agent, request):
        try:
            if source == "resource":
                await agent.get_weather(city="Paris")
            else:
                await agent.llm.acall([{"role": "user", "content": "question"}])
        except ConnectionError:
            pass
        return "apparently successful"

    runner._invocation_adapter = swallow
    if source == "resource":
        resource_tools.request.side_effect = failure
    else:
        runner._server_client.post = AsyncMock(side_effect=failure)
    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(
            NOOARunRequest(
                responses_create_params=responses_create_params("Paris"),
                model_url_path="/v1/responses",
                tool_access=grant(),
            )
        )
    assert caught.value.__cause__ is failure
    assert caught.value.result.termination_reason is None
    root = caught.value.result.trajectory.invocations[0]
    assert root.status == "failed" and root.error_type == "ConnectionError"


@pytest.mark.asyncio
async def test_real_codeact_output_limit_is_gradable_and_preserves_model_evidence() -> None:
    runner, _ = policy_runner()
    response = model_response(response_id="truncated")
    response["incomplete_details"] = {"reason": "max_output_tokens"}
    response["status"] = "incomplete"
    runner._server_client.post = AsyncMock(return_value=FakeHTTPResponse(response))
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
            model_url_path="/v1/responses",
        )
    )
    assert result.termination_reason is NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED
    assert result.trajectory.invocations[0].status == "incomplete"
    assert runner._server_client.post.await_count == 3
    assert len(result.trajectory.turns) == 3
    assert all(turn.model_calls[0].response_id == "truncated" for turn in result.trajectory.turns)


@pytest.mark.asyncio
@pytest.mark.parametrize("truncated,transport_failure", [(False, False), (True, False), (True, True)])
async def test_generation_error_without_output_limit_or_with_transport_failure_is_not_gradable(
    truncated: bool,
    transport_failure: bool,
) -> None:
    runner, _ = make_runner()
    response = model_response()
    if truncated:
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    responses = [FakeHTTPResponse(response)]
    failure = ConnectionError("model unavailable")
    if transport_failure:
        responses.append(failure)
    runner._server_client.post = AsyncMock(side_effect=responses)

    async def invoke(agent, request):
        from responses_api_agents.nooa_agent.gym_llm import OutputTokenLimitExceeded

        try:
            await agent.llm.acall([{"role": "user", "content": "question"}])
        except OutputTokenLimitExceeded:
            pass
        if transport_failure:
            try:
                await agent.llm.acall([{"role": "user", "content": "again"}])
            except ConnectionError:
                pass
        raise GenerationError("arbitrary generation failure")

    runner._invocation_adapter = invoke
    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(
            NOOARunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
                model_url_path="/v1/responses",
            )
        )
    assert caught.value.result.termination_reason is None
    assert isinstance(caught.value.__cause__, GenerationError)


@pytest.mark.asyncio
@pytest.mark.parametrize("main_failed", [False, True])
@pytest.mark.parametrize("summary_failed", [False, True])
async def test_summary_transport_state_is_separate_from_main(main_failed: bool, summary_failed: bool) -> None:
    from nooa.agents.summarization import _in_summary_fork

    runner, _ = make_runner()
    main_error = ConnectionError("unrecovered main call")
    responses = [
        main_error if main_failed else FakeHTTPResponse(model_response()),
        ConnectionError("contained summary call") if summary_failed else FakeHTTPResponse(model_response()),
    ]
    runner._server_client.post = AsyncMock(side_effect=responses)

    async def complete(agent, request):
        try:
            await agent.llm.acall([{"role": "user", "content": "main"}])
        except ConnectionError:
            pass
        token = _in_summary_fork.set(True)
        try:
            try:
                await agent.llm.acall([{"role": "user", "content": "summary"}])
            except ConnectionError:
                pass
        finally:
            _in_summary_fork.reset(token)
        return "completed"

    runner._invocation_adapter = complete
    request = NOOARunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
        model_url_path="/v1/responses",
    )
    if main_failed:
        with pytest.raises(NOOARunFailure) as caught:
            await runner.run(request)
        assert caught.value.__cause__ is main_error
    else:
        result = await runner.run(request)
        assert result.return_value == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_arbitrary_loop_error_is_not_a_policy_termination(wrapped: bool) -> None:
    runner, _ = make_runner()
    error = LoopDetectedError("not raised by the pinned loop guard")

    async def invoke(agent, request):
        try:
            raise error
        except LoopDetectedError as cause:
            if wrapped:
                raise GenerationError("arbitrary wrapper") from cause
            raise

    runner._invocation_adapter = invoke
    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(
            NOOARunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
                model_url_path="/v1/responses",
            )
        )
    assert caught.value.result.termination_reason is None
    assert type(caught.value.__cause__) is (GenerationError if wrapped else LoopDetectedError)


def model_http_error(status: int) -> ClientResponseError:
    return ClientResponseError(
        request_info=SimpleNamespace(real_url="http://model/v1/responses"),
        history=(),
        status=status,
        message="Original provider error: request rejected",
    )


async def invoke_model_directly(agent: Any, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    return await agent.llm.acall([{"role": "user", "content": "question"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 413, 422])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_model_rejection_preserves_original_provider_error(status: int, wrapped: bool) -> None:
    from responses_api_agents.nooa_agent.result import finalize_run_result

    runner, _ = policy_runner()
    failure = model_http_error(status)
    runner._server_client.post = AsyncMock(side_effect=failure)
    if not wrapped:
        runner._invocation_adapter = invoke_model_directly

    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
            model_url_path="/v1/responses",
        )
    )

    assert runner._server_client.post.await_count == (3 if wrapped else 1)
    assert result.termination_reason is NOOATerminationReason.MODEL_REJECTED
    assert result.termination_error == str(failure)
    response, observations = finalize_run_result(result)
    assert response.status == "failed"
    assert response.error.message == str(failure)
    assert any(gap.code == "model_rejected" and gap.detail == str(failure) for gap in observations.gaps)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 408, 425, 429, 500, 503])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_other_model_http_errors_remain_ungradable(status: int, wrapped: bool) -> None:
    runner, _ = policy_runner()
    failure = model_http_error(status)
    runner._server_client.post = AsyncMock(side_effect=failure)
    if not wrapped:
        runner._invocation_adapter = invoke_model_directly

    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(
            NOOARunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
                model_url_path="/v1/responses",
            )
        )

    assert caught.value.result.termination_reason is None
    cause = caught.value.__cause__
    assert (cause.__cause__ if wrapped else cause) is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 413, 422])
async def test_successful_model_retry_clears_rejection(status: int) -> None:
    runner, _ = policy_runner()
    runner._server_client.post.side_effect = [
        model_http_error(status),
        FakeHTTPResponse(
            model_response(
                NeMoGymResponseFunctionToolCall(
                    id="fc-retry", call_id="call-retry", name="return_result", arguments='{"result":"recovered"}'
                )
            )
        ),
    ]

    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
            model_url_path="/v1/responses",
        )
    )

    assert runner._server_client.post.await_count == 2
    assert result.return_value == "recovered"
    assert result.termination_reason is None
    assert result.termination_error is None


@pytest.mark.asyncio
@pytest.mark.parametrize("handling", ["swallowed", "fabricated", "replacement", "arbitrary_wrapper"])
async def test_only_propagated_model_rejection_is_gradable(handling: str) -> None:
    runner, _ = policy_runner()
    failure = model_http_error(400)
    runner._server_client.post = AsyncMock(side_effect=failure)

    async def invoke(agent, request):
        if handling == "fabricated":
            raise failure
        try:
            await invoke_model_directly(agent, request)
        except ClientResponseError as cause:
            if handling == "arbitrary_wrapper":
                raise GenerationError("arbitrary wrapper") from cause
            if handling == "replacement":
                raise model_http_error(400) from cause
        return "apparently successful"

    runner._invocation_adapter = invoke
    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(
            NOOARunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
                model_url_path="/v1/responses",
            )
        )
    assert caught.value.result.termination_reason is None


@pytest.mark.asyncio
async def test_resource_infrastructure_failure_vetoes_model_rejection(outbound: AsyncMock) -> None:
    runner, _ = make_runner()
    resource_failure = ConnectionError("resources unavailable")
    outbound.side_effect = resource_failure
    runner._server_client.post = AsyncMock(side_effect=model_http_error(400))

    async def invoke(agent, request):
        try:
            await agent.get_weather(city="Paris")
        except ConnectionError:
            pass
        return await invoke_model_directly(agent, request)

    runner._invocation_adapter = invoke
    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(
            NOOARunRequest(
                tool_access=grant(),
                responses_create_params=responses_create_params("Paris"),
                model_url_path="/v1/responses",
            )
        )
    assert caught.value.__cause__ is resource_failure
    assert caught.value.result.termination_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize("marked_error", ["wrapper", "model"])
async def test_cleanup_marker_anywhere_in_trusted_model_chain_vetoes_grading(marked_error: str) -> None:
    runner, _ = policy_runner()
    runner._server_client.post = AsyncMock(side_effect=model_http_error(400))
    original_adapter = runner._invocation_adapter

    async def invoke(agent, request):
        try:
            return await original_adapter(agent, request)
        except GenerationError as error:
            target = error if marked_error == "wrapper" else error.__cause__
            target.nooa_cleanup_error = ConnectionError("cleanup failed")
            raise

    runner._invocation_adapter = invoke
    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(
            NOOARunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
                model_url_path="/v1/responses",
            )
        )
    assert caught.value.result.termination_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_boundary", ["agent", "scope", "both"])
async def test_benchagent_cleanup_failure_overrides_model_rejection(
    tmp_path, monkeypatch, cleanup_boundary: str
) -> None:
    from nooa.tools._bash_session import BashSession
    from nooa.tools.shell_tools import ShellTools

    module = pytest.importorskip("nooa_bench.bench_agent")
    monkeypatch.chdir(tmp_path)
    fresh_sessions = []
    solve_errors = []
    original_solve = module.BenchAgent._solve_task
    original_session_close = BashSession.close
    original_shell_close = ShellTools.close
    agent_error = ValueError("agent cleanup failed")
    scope_error = ConnectionError("scope cleanup failed")

    async def solve(agent, *args, **kwargs):
        fresh_sessions.append(ShellTools(cwd=str(tmp_path)).session)
        try:
            return await original_solve(agent, *args, **kwargs)
        except Exception as error:
            solve_errors.append(error)
            raise

    async def session_close(session):
        await original_session_close(session)
        if cleanup_boundary in {"scope", "both"} and session in fresh_sessions:
            raise scope_error

    async def shell_close(shell):
        await original_shell_close(shell)
        if cleanup_boundary in {"agent", "both"}:
            raise agent_error

    monkeypatch.setattr(module.BenchAgent, "_solve_task", solve)
    monkeypatch.setattr(BashSession, "close", session_close)
    monkeypatch.setattr(ShellTools, "close", shell_close)
    failure = model_http_error(400)
    runner = InProcessNOOARunner(
        invocation=NOOAInvocationConfig(
            agent_class="nooa_bench.bench_agent:BenchAgent",
            invocation_adapter="responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent",
        ),
        server_client=SimpleNamespace(post=AsyncMock(side_effect=failure)),
        model_server_name="policy",
        max_policy_calls=None,
    )

    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(
            NOOARunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
                model_url_path="/v1/responses",
            )
        )

    assert caught.value.result.termination_reason is None
    assert len(solve_errors) == 1
    assert caught.value.__cause__ is solve_errors[0]
    assert solve_errors[0].__cause__ is failure
    expected = tuple(
        error
        for boundary, error in (("agent", agent_error), ("scope", scope_error))
        if cleanup_boundary in {boundary, "both"}
    )
    assert solve_errors[0].nooa_cleanup_errors == expected
    for error in expected:
        assert str(error) in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["model", "resource"])
async def test_fatal_transport_then_policy_stop_finalizes_root_as_failed(source: str) -> None:
    from responses_api_agents.nooa_agent.gym_llm import PolicyCallBudgetExceeded

    runner, _ = make_runner()
    failure = ConnectionError("transport unavailable")
    runner._max_policy_calls = 1

    async def fail_then_exhaust(agent, request):
        try:
            if source == "model":
                await agent.llm.acall([{"role": "user", "content": "first"}])
            else:
                await agent.get_weather(city="Paris")
        except ConnectionError:
            pass
        if source == "resource":
            await agent.llm.acall([{"role": "user", "content": "first"}])
        return await agent.llm.acall([{"role": "user", "content": "over budget"}])

    runner._invocation_adapter = fail_then_exhaust
    if source == "model":
        runner._server_client.post = AsyncMock(side_effect=failure)
    else:
        resource_tools.request.side_effect = failure
        runner._server_client.post = AsyncMock(return_value=FakeHTTPResponse(model_response()))
    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(
            NOOARunRequest(
                responses_create_params=responses_create_params("Paris"),
                model_url_path="/v1/responses",
                tool_access=grant(),
            )
        )
    result = caught.value.result
    assert result.termination_reason is None
    root = result.trajectory.invocations[0]
    assert root.status == "failed"
    assert root.error_type == ("PolicyCallBudgetExceeded" if source == "model" else "ConnectionError")
    assert isinstance(caught.value.__cause__, PolicyCallBudgetExceeded if source == "model" else ConnectionError)
    assert runner._server_client.post.await_count == 1
