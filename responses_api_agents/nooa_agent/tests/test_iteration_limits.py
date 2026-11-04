# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise agent sessions with the pinned CodeAct runtime and scripted model replies."""

import asyncio
from json import dumps as json_dumps
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CodeActConfig
from nooa.errors import GenerationError

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from nemo_gym.rollouts.rollout_collection import _episode_record
from responses_api_agents.nooa_agent.app import NOOAAgent
from responses_api_agents.nooa_agent.runner import NOOARunFailure
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_app import close, open_session
from responses_api_agents.nooa_agent.tests.test_config import agent_config
from responses_api_agents.nooa_agent.tests.test_environment import _environment, _paths, _request, _Response
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


class IterationLimitedAgent(Agent):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=1, max_retries=1)))
    async def analyze(self, text: str) -> str:
        """Work on the task using execute_python."""
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
    async def followup(self) -> str:
        """Ask the model for more information."""
        ...


class CallLimitedAgent(Agent):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2, max_retries=1)))
    async def analyze(self, text: str) -> str:
        """Continue working until the host stops model calls."""
        ...


class RetryingAgent(Agent):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3, max_retries=2)))
    async def analyze(self, text: str) -> str:
        """Retry a failed generation before completing the task."""
        ...


class TextOnlyLimitedAgent(Agent):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=1, max_consecutive_text_only=1)))
    async def analyze(self, text: str) -> str:
        """Work on the task using execute_python."""
        ...


async def invoke_text(
    agent: IterationLimitedAgent | TextOnlyLimitedAgent | CallLimitedAgent | RetryingAgent,
    request: NeMoGymResponseCreateParamsNonStreaming,
) -> str:
    return await agent.analyze(str(request.input))


def iteration_agent(*, scenario: str = "iterations", sentinel: Path | None = None) -> tuple[NOOAAgent, list[object]]:
    client = MagicMock(spec=ServerClient)
    model_calls: list[object] = []

    async def post(*, server_name: str, url_path: str, json: object, **_: object) -> FakeHTTPResponse:
        assert server_name == "policy"
        assert url_path.endswith("/v1/responses")
        model_calls.append(json)
        if len(model_calls) > 1:
            if scenario == "output_then_success":
                return FakeHTTPResponse(
                    model_response(
                        NeMoGymResponseFunctionToolCall(
                            id="recovered", call_id="call-2", name="return_result", arguments='{"result":"recovered"}'
                        ),
                        response_id="recovered-response",
                    )
                )
            raise ConnectionError("unrecovered model outage")
        code = "print('partial progress')"
        if scenario == "retries":
            code = ""
        elif scenario == "model_outage":
            code = "try:\n    await self.followup()\nexcept Exception:\n    pass\nprint('partial progress')"
        elif scenario.startswith("output_"):
            assert sentinel is not None
            code = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')"
        output = [
            {"type": "reasoning", "id": "reasoning-1", "summary": [{"type": "summary_text", "text": "plan so far"}]},
            NeMoGymResponseOutputMessageForTraining(
                id="message-1",
                prompt_token_ids=[1],
                generation_token_ids=[2],
                generation_log_probs=[-0.1],
                content=[NeMoGymResponseOutputText(annotations=[], text="working on the task", logprobs=[])],
            ),
        ]
        if scenario not in {"text_only", "output_text"}:
            arguments = json_dumps({"code": code})
            if scenario == "output_partial_tool":
                arguments = arguments[:-1]
            output.append(
                NeMoGymResponseFunctionToolCall(
                    id="function-1",
                    call_id="call-1",
                    name="execute_python",
                    arguments=arguments,
                )
            )
        response = model_response(*output)
        if scenario.startswith("output_"):
            response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        return FakeHTTPResponse(response)

    client.post = AsyncMock(side_effect=post)
    agent_class = IterationLimitedAgent
    if scenario == "text_only":
        agent_class = TextOnlyLimitedAgent
    elif scenario == "calls":
        agent_class = CallLimitedAgent
    elif scenario in {"output_then_success", "output_then_outage"}:
        agent_class = RetryingAgent
    instance = NOOAAgent(
        config=agent_config(
            nooa={
                "agent_class": f"{__name__}:{agent_class.__name__}",
                "invocation_adapter": f"{__name__}:invoke_text",
            },
            max_policy_calls=1 if scenario == "calls" else None,
        ),
        server_client=client,
    )
    return instance, model_calls


LIMIT_SCENARIOS = [
    ("iterations", NOOATerminationReason.MAX_ITERATIONS_EXCEEDED),
    ("calls", NOOATerminationReason.POLICY_BUDGET_EXCEEDED),
    ("output_text", NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED),
    ("output_tool", NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED),
    ("output_partial_tool", NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED),
]


def assert_partial_evidence(response: NeMoGymResponse, *, scenario: str) -> None:
    assert response.status == "incomplete"
    assert response.error is None
    assert any(item.type == "reasoning" for item in response.output)
    assert any(item.type == "message" and item.content[0].text == "working on the task" for item in response.output)
    calls = [item for item in response.output if item.type == "function_call" and item.call_id == "call-1"]
    if scenario == "output_text":
        assert not calls
    else:
        assert len(calls) == 1
        assert calls[0].call_id == "call-1"
        if scenario == "output_partial_tool":
            assert calls[0].arguments.endswith('"')
    if scenario.startswith("output_"):
        assert response.incomplete_details.reason == "max_output_tokens"
        assert not any(item.type == "function_call_output" and item.call_id == "call-1" for item in response.output)
    else:
        assert any(
            item.type == "function_call_output" and item.call_id == "call-1" and "partial progress" in item.output
            for item in response.output
        )


@pytest.mark.parametrize(("scenario", "reason"), LIMIT_SCENARIOS)
async def test_real_codeact_response_retains_partial_evidence_without_verification(
    scenario: str, reason: NOOATerminationReason, tmp_path: Path
) -> None:
    sentinel = tmp_path / "must-not-execute"
    instance, model_calls = iteration_agent(scenario=scenario, sentinel=sentinel)
    req = await open_session(instance)
    try:
        response = await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert_partial_evidence(response, scenario=scenario)
        assert "_ng_failure_class" not in response.model_dump(mode="json")
        assert len(model_calls) == 1
        assert not sentinel.exists()
    finally:
        receipt = await close(instance)
    assert reason in {gap.code for gap in receipt.agent_observations.gaps}


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [*LIMIT_SCENARIOS, ("output_then_success", None), ("output_then_outage", None), ("model_outage", None)],
)
async def test_real_codeact_environment_verifies_partial_response_once(
    scenario: str, reason: NOOATerminationReason | None, tmp_path: Path
) -> None:
    sentinel = tmp_path / "must-not-execute"
    instance, model_calls = iteration_agent(scenario=scenario, sentinel=sentinel)
    environment, client = _environment(instance)
    original = client.post.side_effect
    verified: list[dict[str, object]] = []

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        response = await original(server_name=server_name, url_path=url_path, **kwargs)
        if url_path == "/verify":
            verified.append(kwargs["json"])
            response.body["reward"] = 0.75
        return response

    client.post.side_effect = post
    result = await environment.run_request(_request())
    assert not sentinel.exists()
    if scenario in {"output_then_outage", "model_outage"}:
        assert result.failure.stage == "agent"
        assert result.result is None
        assert len(model_calls) == 2
        assert verified == []
        assert _paths(client).count("/verify") == 0
        receipt = next(iter(instance._closed_session_records.values())).close_response
        assert NOOATerminationReason.INFRASTRUCTURE_ERROR in {gap.code for gap in receipt.agent_observations.gaps}
        return
    assert result.failure is None
    assert result.result.reward == 0.75
    if scenario == "output_then_success":
        assert result.result.response.status == "completed"
        assert result.result.response.error is None
        assert result.result.response.incomplete_details is None
        assert result.result.response.output[-1].content[0].text == "recovered"
        assert len(model_calls) == 2
    else:
        assert_partial_evidence(result.result.response, scenario=scenario)
        assert len(model_calls) == 1
    assert len(verified) == _paths(client).count("/verify") == 1
    assert verified[0]["response"] == result.result.response.model_dump(mode="json")
    record = _episode_record(result.model_dump(mode="json"))
    assert "_ng_failure_class" not in record
    assert record["reward"] == 0.75
    gaps = {gap["code"] for gap in record["ng_agent_observations"]["gaps"]}
    if reason is not None:
        assert reason.value in gaps
    else:
        assert NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED not in gaps
    assert _paths(client)[-3:] == ["/v1/agent_sessions/close", "/verify", "/close_session"]


@pytest.mark.parametrize("scenario", ["retries", "text_only", "model_outage"])
async def test_other_codeact_limits_and_unrecovered_outage_fail_agent_activation(scenario: str) -> None:
    instance, model_calls = iteration_agent(scenario=scenario)
    req = await open_session(instance)
    try:
        with pytest.raises((NOOARunFailure, HTTPException)):
            await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert len(model_calls) == (2 if scenario == "model_outage" else 1)
        assert instance._require_agent_session("session").result.episode.response.output
    finally:
        receipt = await close(instance)
    gaps = {gap.code for gap in receipt.agent_observations.gaps}
    assert NOOATerminationReason.INFRASTRUCTURE_ERROR in gaps
    assert NOOATerminationReason.MAX_ITERATIONS_EXCEEDED not in gaps


async def test_matching_generation_error_message_is_not_a_gradable_execution_limit() -> None:
    instance, model_calls = iteration_agent()

    async def invoke_error(agent: object, request: object) -> object:
        raise GenerationError("Generation failed after 1 iterations (max_iterations=1). Unable to complete `analyze`.")

    req = await open_session(instance)
    instance._require_agent_session("session").runner._invocation_adapter = invoke_error
    try:
        with pytest.raises(NOOARunFailure):
            await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert model_calls == []
    finally:
        receipt = await close(instance)
    assert NOOATerminationReason.INFRASTRUCTURE_ERROR in {gap.code for gap in receipt.agent_observations.gaps}


@pytest.mark.parametrize("scenario", ["iterations", "output_tool"])
async def test_cancellation_after_limit_stays_cancelled_in_agent_session(scenario: str, tmp_path: Path) -> None:
    sentinel = tmp_path / "must-not-execute"
    instance, _ = iteration_agent(scenario=scenario, sentinel=sentinel)

    async def cancel_after_limit(
        agent: IterationLimitedAgent, request: NeMoGymResponseCreateParamsNonStreaming
    ) -> object:
        try:
            await invoke_text(agent, request)
        except GenerationError:
            raise asyncio.CancelledError

    req = await open_session(instance)
    instance._require_agent_session("session").runner._invocation_adapter = cancel_after_limit
    try:
        with pytest.raises(asyncio.CancelledError):
            await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert instance._require_agent_session("session").result.episode.response.output
    finally:
        receipt = await close(instance)
    gaps = {gap.code for gap in receipt.agent_observations.gaps}
    assert NOOATerminationReason.CANCELLED in gaps
    assert NOOATerminationReason.MAX_ITERATIONS_EXCEEDED not in gaps
    assert NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED not in gaps
    assert not sentinel.exists()


async def test_codeact_cleanup_failure_after_iteration_exhaustion_skips_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, model_calls = iteration_agent()
    environment, client = _environment(instance)
    cleanup_failure = ConnectionError("sandbox cleanup connection lost")
    cleanup = AsyncMock(side_effect=cleanup_failure)
    monkeypatch.setattr(CodeActStrategy, "_close_sandbox", cleanup)

    result = await environment.run_request(_request())

    cleanup.assert_awaited_once()
    assert isinstance(cleanup_failure.__context__, GenerationError)
    assert result.failure.stage == "agent"
    assert result.result is None
    assert len(model_calls) == 1
    assert "/verify" not in _paths(client)
    assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]
    receipt = next(iter(instance._closed_session_records.values())).close_response
    gaps = {gap.code for gap in receipt.agent_observations.gaps}
    assert NOOATerminationReason.INFRASTRUCTURE_ERROR in gaps
    assert NOOATerminationReason.MAX_ITERATIONS_EXCEEDED not in gaps
    assert receipt.agent_observations.records
