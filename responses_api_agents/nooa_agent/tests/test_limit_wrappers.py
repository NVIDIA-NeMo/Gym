# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep actual CodeAct limits gradable without hiding outer failures."""

from unittest.mock import AsyncMock

import pytest
from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CodeActConfig, PredictConfig
from nooa.errors import GenerationError
from nooa.strategies.predict import PredictStrategy

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from responses_api_agents.nooa_agent.gym_llm import (
    InvalidPolicyOutputError,
    OutputTokenLimitExceeded,
    PolicyCallBudgetExceeded,
)
from responses_api_agents.nooa_agent.runner import NOOARunFailure, NOOARunRequest
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response
from responses_api_agents.nooa_agent.tests.test_runner import invoke_policy, policy_runner


class CustomGenerationError(GenerationError):
    pass


class OneIterationAgent(Agent):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=1, max_retries=1)))
    async def analyze(self, text: str) -> str:
        """Make progress using execute_python before returning the answer."""
        ...


class TwoIterationAgent(Agent):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2, max_retries=1)))
    async def analyze(self, text: str) -> str:
        """Make progress using execute_python before returning the answer."""
        ...


def _limit_runner(scenario: str):
    runner, _ = policy_runner()
    runner._agent_class = TwoIterationAgent if scenario == "policy_calls" else OneIterationAgent
    runner._max_policy_calls = 1
    response = model_response(
        NeMoGymResponseFunctionToolCall(
            id="partial", call_id="partial-call", name="execute_python", arguments='{"code":"print(42)"}'
        )
    )
    if scenario == "output_tokens":
        response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    runner._server_client.post = AsyncMock(return_value=FakeHTTPResponse(response))
    return runner


def _request() -> NOOARunRequest:
    return NOOARunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="work on this task"),
        model_url_path="/v1/responses",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "reason", "cause_type"),
    [
        ("iterations", NOOATerminationReason.MAX_ITERATIONS_EXCEEDED, type(None)),
        ("output_tokens", NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED, OutputTokenLimitExceeded),
        ("policy_calls", NOOATerminationReason.POLICY_BUDGET_EXCEEDED, PolicyCallBudgetExceeded),
    ],
)
async def test_actual_codeact_limit_wrappers_remain_gradable(
    scenario: str, reason: NOOATerminationReason, cause_type: type
) -> None:
    runner = _limit_runner(scenario)
    raised: list[GenerationError] = []

    async def capture(agent: Agent, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
        try:
            return await invoke_policy(agent, request)
        except GenerationError as error:
            raised.append(error)
            raise

    runner._invocation_adapter = capture
    result = await runner.run(_request())

    assert len(raised) == 1
    assert type(raised[0]) is GenerationError
    assert type(raised[0].__cause__) is cause_type
    assert result.termination_reason is reason
    assert result.episode.response.output
    assert len(result.trajectory.turns) == runner._server_client.post.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_type", [ConnectionError, GenerationError, CustomGenerationError])
@pytest.mark.parametrize("explicit_cause", [False, True])
async def test_arbitrary_wrapper_does_not_make_an_earlier_iteration_limit_gradable(
    wrapper_type: type[Exception], explicit_cause: bool
) -> None:
    runner = _limit_runner("iterations")
    outer_failure = wrapper_type("failure after iteration exhaustion")

    async def fail_after_limit(agent: Agent, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
        try:
            return await invoke_policy(agent, request)
        except GenerationError as error:
            if explicit_cause:
                raise outer_failure from error
            raise outer_failure

    runner._invocation_adapter = fail_after_limit
    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(_request())

    assert isinstance(outer_failure.__context__, GenerationError)
    assert (outer_failure.__cause__ is not None) is explicit_cause
    assert caught.value.__cause__ is outer_failure
    assert caught.value.result.termination_reason is None
    assert caught.value.result.termination_error is None
    assert caught.value.result.episode.response.output
    assert runner._server_client.post.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["iterations", "output_tokens", "policy_calls"])
async def test_codeact_cleanup_failure_preserves_outer_error_after_a_limit(
    scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _limit_runner(scenario)
    cleanup_failure = ConnectionError("sandbox cleanup connection lost")
    cleanup = AsyncMock(side_effect=cleanup_failure)
    monkeypatch.setattr(CodeActStrategy, "_close_sandbox", cleanup)

    with pytest.raises(NOOARunFailure) as caught:
        await runner.run(_request())

    cleanup.assert_awaited_once()
    assert isinstance(cleanup_failure.__context__, GenerationError)
    assert caught.value.__cause__ is cleanup_failure
    assert caught.value.result.termination_reason is None
    assert caught.value.result.termination_error is None
    assert caught.value.result.episode.response.output
    assert runner._server_client.post.await_count == 1


class StructuredRetryAgent(Agent):
    @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
    async def analyze(self, text: str) -> str:
        """Return a structured answer for the supplied text."""
        ...


@pytest.mark.asyncio
async def test_actual_predict_validation_wrapper_remains_gradable() -> None:
    runner, _ = policy_runner()
    runner._agent_class = StructuredRetryAgent
    response = model_response(
        {
            "id": "invalid",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "not json", "annotations": []}],
        }
    )
    runner._server_client.post = AsyncMock(return_value=FakeHTTPResponse(response))
    raised: list[GenerationError] = []

    async def capture(agent: Agent, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
        try:
            return await invoke_policy(agent, request)
        except GenerationError as error:
            raised.append(error)
            raise

    runner._invocation_adapter = capture
    result = await runner.run(_request())

    assert len(raised) == 1
    assert type(raised[0]) is GenerationError
    assert isinstance(raised[0].__cause__, InvalidPolicyOutputError)
    assert result.termination_reason is NOOATerminationReason.INVALID_POLICY_OUTPUT
    assert result.episode.response.output
    assert len(result.trajectory.turns) == runner._server_client.post.await_count == 1
