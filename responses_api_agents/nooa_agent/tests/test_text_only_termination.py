# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify the real pinned CodeAct non-tool limit without hiding infrastructure failures."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CodeActConfig
from nooa.errors import GenerationError

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import _episode_record
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent import runner
from responses_api_agents.nooa_agent.app import NOOAAgent
from responses_api_agents.nooa_agent.runner import NOOARunFailure
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_app import close, open_session
from responses_api_agents.nooa_agent.tests.test_config import agent_config
from responses_api_agents.nooa_agent.tests.test_environment import _environment, _paths, _request
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


class ThreeTextTurns(Agent):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=10, max_retries=3, max_consecutive_text_only=3)))
    async def analyze(self, text: str) -> str:
        """Use tools and return_result to complete the task."""
        ...


async def invoke(agent: ThreeTextTurns, request: NeMoGymResponseCreateParamsNonStreaming) -> str:
    return await agent.analyze(str(request.input))


def make_agent(
    *, text: str = "", status: str = "completed", outage: bool = False
) -> tuple[NOOAAgent, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []
    client = MagicMock(spec=ServerClient)

    async def post(**kwargs: object) -> FakeHTTPResponse:
        calls.append(kwargs)
        if outage and len(calls) >= 3:
            raise ConnectionError("synthetic model outage")
        i = len(calls)
        output = [{"type": "reasoning", "id": f"rs-{i}", "summary": [{"type": "summary_text", "text": "Plan"}]}]
        if text:
            output.append(
                {
                    "type": "message",
                    "id": f"msg-{i}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                    "status": "completed",
                }
            )
        response = model_response(*output, response_id=f"response-{i}")
        response["status"] = status
        return FakeHTTPResponse(response)

    client.post = AsyncMock(side_effect=post)
    instance = NOOAAgent(
        config=agent_config(
            nooa={"agent_class": f"{__name__}:ThreeTextTurns", "invocation_adapter": f"{__name__}:invoke"}
        ),
        server_client=client,
    )
    return instance, calls


@pytest.mark.parametrize("text", ["", "I am done"])
async def test_three_completed_non_tool_turns_reach_verifier_with_authoritative_root(text: str) -> None:
    instance, calls = make_agent(text=text)
    environment, client = _environment(instance)
    original = client.post.side_effect
    verified = []

    async def post(**kwargs: object):
        result = await original(**kwargs)
        if kwargs["url_path"] == "/verify":
            verified.append(kwargs["json"])
            result.body["reward"] = 0.75
        return result

    client.post.side_effect = post
    result = await environment.run_request(_request())
    assert result.failure is None and result.result.reward == 0.75
    assert len(calls) == 3 and len(verified) == 1
    response = result.result.response
    assert response.status == "failed" and response.error is not None
    assert response.metadata["terminal_response_id"] == "response-3"
    assert verified[0]["response"]["metadata"]["terminal_response_id"] == "response-3"
    assert len([item for item in response.output if item.type == "reasoning"]) == 3
    row = _episode_record(result.model_dump(mode="json"))
    assert "_ng_failure_class" not in row
    assert row["reward"] == 0.75
    assert NOOATerminationReason.INVALID_POLICY_OUTPUT in {gap["code"] for gap in row["ng_agent_observations"]["gaps"]}
    roots = [
        r
        for r in row["ng_agent_observations"]["records"]
        if r["kind"] == "agent_invocation" and not r.get("parent_invocation_id")
    ]
    assert len(roots) == 1 and len(roots[0]["model_calls"]) == 3
    assert roots[0]["model_calls"][-1]["response_id"] == "response-3"
    assert _paths(client)[-3:] == ["/v1/agent_sessions/close", "/verify", "/close_session"]


@pytest.mark.parametrize("status", ["incomplete", "failed"])
async def test_uncompleted_non_tool_response_is_not_gradable_policy_output(status: str) -> None:
    instance, calls = make_agent(status=status)
    req = await open_session(instance)
    try:
        with pytest.raises(NOOARunFailure):
            await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert len(calls) == 3
    finally:
        receipt = await close(instance)
    assert NOOATerminationReason.INVALID_POLICY_OUTPUT not in {g.code for g in receipt.agent_observations.gaps}


async def test_unrecovered_outage_after_text_turns_stays_infrastructure_failure() -> None:
    instance, calls = make_agent(outage=True)
    environment, client = _environment(instance)
    result = await environment.run_request(_request())
    assert result.failure.stage == "agent" and result.result is None
    assert _paths(client).count("/verify") == 0
    assert len(calls) == 5


async def test_fatal_dispatcher_overrides_completed_text_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    original = runner.ResourceToolDispatcher

    def dispatcher(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        result.fatal_error = ConnectionError("synthetic unrecovered tool transport")
        return result

    monkeypatch.setattr(runner, "ResourceToolDispatcher", dispatcher)
    instance, calls = make_agent()
    environment, client = _environment(instance)
    result = await environment.run_request(_request())
    assert len(calls) == 3
    assert result.failure.stage == "agent" and result.result is None
    assert _paths(client).count("/verify") == 0


def test_same_error_text_with_wrong_origin_or_absent_state_is_unrecognized(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        raise GenerationError("CodeAct aborted: LLM returned plain text without a tool call 3 times in a row")
    except GenerationError as error:
        assert not runner._is_codeact_text_only_exhaustion(error)
        monkeypatch.setattr(runner, "_generation_error_frame", lambda _: None)
        assert not runner._is_codeact_text_only_exhaustion(error)


@pytest.mark.parametrize("missing", ["session", "self", "response", "max_text_only"])
async def test_exact_codeact_frame_requires_live_limit_state(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    predicate = runner._is_codeact_text_only_exhaustion
    failures: list[BaseException] = []

    def intercept(error: BaseException) -> bool:
        if predicate(error):
            failures.append(error)
        return False

    monkeypatch.setattr(runner, "_is_codeact_text_only_exhaustion", intercept)
    instance, _ = make_agent()
    environment, client = _environment(instance)
    result = await environment.run_request(_request())
    assert result.failure is not None and _paths(client).count("/verify") == 0
    assert len(failures) == 1
    error = failures[0]
    frame = runner._generation_error_frame(error)
    assert frame is not None and predicate(error)
    state = dict(frame.f_locals)
    del state[missing]
    monkeypatch.setattr(
        runner, "_generation_error_frame", lambda _: SimpleNamespace(f_code=frame.f_code, f_locals=state)
    )
    assert not predicate(error)
