# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import orjson
import pytest
from fastapi import Response

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.webarena.schemas import (
    WebArenaAgentRunRequest,
    WebArenaEnvStateEasyInputMessage,
    WebArenaNeMoGymResponse,
)
from responses_api_agents.webarena_agent import app as web_agent_app
from responses_api_agents.webarena_agent.app import (
    TASK_INPUT_IMAGE_MARKER,
    TerminationReason,
    WebAgent,
    WebAgentConfig,
    WebArenaRolloutError,
    _validated_request_body,
    compact_model_input,
)


class FakeResponse:
    def __init__(self, payload, cookies=None):
        self.payload = payload
        self.cookies = cookies or {}
        self.ok = True

    async def read(self):
        return orjson.dumps(self.payload)


def _task():
    return {
        "task_id": "webarena-test",
        "sites": ["shopping"],
        "start_urls": ["http://shopping.test"],
        "intent": "Complete the task",
        "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "done"}},
    }


def _env_obs(step, screenshot, *, task_image=None):
    content = [
        {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot}", "detail": "auto"},
        {"type": "input_text", "text": f"step {step}"},
    ]
    if task_image is not None:
        content.insert(
            1,
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{task_image}",
                "detail": "auto",
                TASK_INPUT_IMAGE_MARKER: True,
            },
        )
    return {
        "role": "user",
        "content": content,
        "is_env_state": True,
        "is_browser_screenshot": True,
        "step": step,
    }


def _model_response(response_id, output, *, input_tokens=1, output_tokens=2):
    return {
        "id": response_id,
        "created_at": 1.0,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _call(call_id, name="computer", arguments=None):
    if arguments is None:
        arguments = {"status": "success"} if name == "terminate" else {"actions": [{"action": "wait", "duration": 1}]}
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


def _message(text):
    return {
        "id": f"msg-{text}",
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


@pytest.fixture
def agent():
    config = WebAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="web-agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="webarena"),
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        max_steps=3,
        max_recent_screenshots=1,
    )
    return WebAgent(config=config, server_client=MagicMock(spec=ServerClient))


def test_config_reserves_time_for_finalization():
    with pytest.raises(ValueError, match="leave time"):
        WebAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="web-agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="webarena"),
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            rollout_timeout_seconds=300,
            finalization_reserve_seconds=30,
        )


@pytest.mark.asyncio
async def test_seed_wait_is_outside_interaction_and_request_timeouts(
    agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_completed = False
    requests: list[tuple[str, float | None]] = []

    async def post_json(_server_name, url_path, _payload, cookies, timeout_seconds):
        nonlocal seed_completed
        requests.append((url_path, timeout_seconds))
        if url_path == "/seed_session":
            seed_completed = True
            return {
                "env_id": "env-seed-wait",
                "obs": [_env_obs(1, "SEED")],
                "tools": [],
            }, cookies
        if url_path == "/v1/responses":
            return _model_response(
                "response-terminate",
                [_call("call-terminate", "terminate")],
            ), cookies
        if url_path == "/step":
            return {
                "obs": [],
                "done": True,
                "termination_status": "success",
            }, cookies
        if url_path == "/close":
            return {"success": True}, cookies
        raise AssertionError(f"Unexpected path: {url_path}")

    class TimeoutAfterSeed:
        async def __aenter__(self):
            assert seed_completed

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

    agent._post_json = AsyncMock(side_effect=post_json)
    monkeypatch.setattr(
        web_agent_app.asyncio,
        "timeout_at",
        lambda _deadline: TimeoutAfterSeed(),
    )

    await agent._rollout(_request(), {})

    assert requests == [
        ("/seed_session", None),
        ("/v1/responses", None),
        ("/step", agent.config.resource_request_timeout_seconds),
        ("/close", web_agent_app._CLOSE_TIMEOUT_SECONDS),
    ]


def test_validated_request_body_revalidates_input_items():
    request = _validated_request_body(
        NeMoGymResponseCreateParamsNonStreaming(input=[], temperature=0.5),
        request_input=[{"role": "system", "content": "System prompt", "type": "message"}],
    )

    assert isinstance(request.input, list)
    assert isinstance(request.input[0], NeMoGymEasyInputMessage)
    assert request.temperature == 0.5
    request.model_dump(mode="json", warnings="error")


def _request():
    return WebArenaAgentRunRequest(
        task_payload=_task(),
        task_idx=7,
        responses_create_params={"input": [], "parallel_tool_calls": False},
    )


def test_compact_model_input_only_prunes_old_browser_screenshots():
    task_image = {
        "type": "input_image",
        "image_url": "data:image/png;base64,TASK-OUTSIDE-ENV",
        "detail": "auto",
        TASK_INPUT_IMAGE_MARKER: True,
    }
    history = [
        {"role": "user", "content": [task_image, {"type": "input_text", "text": "task"}]},
        WebArenaEnvStateEasyInputMessage.model_validate(_env_obs(1, "BROWSER-1", task_image="TASK-1")),
        {"type": "function_call_output", "call_id": "one", "output": "unchanged"},
        WebArenaEnvStateEasyInputMessage.model_validate(_env_obs(2, "BROWSER-2")),
        WebArenaEnvStateEasyInputMessage.model_validate(_env_obs(3, "BROWSER-3")),
    ]

    compacted = compact_model_input(history, max_recent_screenshots=2)

    assert "BROWSER-1" not in json.dumps(compacted)
    assert "BROWSER-2" in json.dumps(compacted)
    assert "BROWSER-3" in json.dumps(compacted)
    assert "TASK-1" in json.dumps(compacted)
    assert "TASK-OUTSIDE-ENV" in json.dumps(compacted)
    assert TASK_INPUT_IMAGE_MARKER not in json.dumps(compacted)
    assert compacted[2]["output"] == "unchanged"


@pytest.mark.asyncio
async def test_rollout_preserves_flat_images_cookies_usage_and_request_ids(agent):
    env_id = "env-1"
    responses = [
        FakeResponse(
            {"env_id": env_id, "obs": [_env_obs(1, "SEED")], "tools": []},
            {"resource": "seed-cookie"},
        ),
        FakeResponse(_model_response("response-1", [_call("call-1")]), {"model": "one"}),
        FakeResponse(
            {
                "obs": [
                    {"type": "function_call_output", "call_id": "call-1", "output": "executed"},
                    _env_obs(2, "LATER"),
                ],
                "done": False,
            },
            {"resource": "step-cookie"},
        ),
        FakeResponse(
            _model_response("response-2", [_call("call-2", "terminate")], input_tokens=3, output_tokens=4),
            {"model": "two"},
        ),
        FakeResponse(
            {
                "obs": [{"type": "function_call_output", "call_id": "call-2", "output": "terminated"}],
                "done": True,
                "termination_status": "success",
            },
            {"resource": "done-cookie"},
        ),
        FakeResponse({"success": True}),
    ]
    agent.server_client.post = AsyncMock(side_effect=responses)

    result, cookies = await agent._rollout(_request(), {"incoming": "cookie"})

    assert result.env_id == env_id
    assert result.group_id == "7"
    assert result.contains_transitions is False
    assert "SEED" in json.dumps(result.seed_obs[0].model_dump())
    assert "LATER" in json.dumps(result.model_dump()["output"])
    assert result.usage.input_tokens == 4
    assert result.usage.output_tokens == 6
    assert cookies == {"incoming": "cookie", "resource": "done-cookie"}

    calls = agent.server_client.post.await_args_list
    seed_json = calls[0].kwargs["json"]
    assert set(seed_json) == {"request_id", "task"}
    assert seed_json["task"]["task_id"] == "webarena-test"
    assert seed_json["request_id"]

    first_model_input = calls[1].kwargs["json"].input
    assert first_model_input[0].role == "system"
    assert "GUI agent controlling a web browser" in first_model_input[0].content
    assert calls[1].kwargs["json"].parallel_tool_calls is False

    first_step = calls[2].kwargs["json"]
    assert first_step["env_id"] == env_id
    assert first_step["action_request_id"]
    assert len(first_step["action"]) == 1

    second_model_json = calls[3].kwargs["json"].model_dump(mode="json")
    assert "SEED" not in json.dumps(second_model_json)
    assert "LATER" in json.dumps(second_model_json)

    close_json = calls[-1].kwargs["json"]
    assert calls[-1].kwargs["url_path"] == "/close"
    assert close_json["env_id"] == env_id
    assert close_json["close_request_id"]


def _single_violation_rollout(model_output, *, tools=None, incomplete=False):
    model_response = _model_response("response-bad", model_output)
    if incomplete:
        model_response["incomplete_details"] = {"reason": "max_output_tokens"}
        model_response["status"] = "incomplete"
    return [
        FakeResponse({"env_id": "env-bad", "obs": [_env_obs(1, "SEED")], "tools": tools or []}),
        FakeResponse(model_response),
        FakeResponse({"success": True}),
    ]


def _assert_stopped_without_step(agent, result, reason):
    paths = [item.kwargs["url_path"] for item in agent.server_client.post.await_args_list]
    assert paths == ["/seed_session", "/v1/responses", "/close"]
    assert not [item for item in result.output if getattr(item, "type", None) == "function_call_output"]
    assert "protocol_error" not in json.dumps(result.model_dump(mode="json")["output"])
    metadata = json.loads(result.metadata["webarena_termination"])
    assert metadata["reason"] == reason
    return metadata


@pytest.mark.asyncio
async def test_no_tool_call_stops_without_feedback(agent):
    agent.server_client.post = AsyncMock(side_effect=_single_violation_rollout([_message("use a tool")]))

    result, _ = await agent._rollout(_request(), {})

    _assert_stopped_without_step(agent, result, TerminationReason.NO_TOOL_CALL)


@pytest.mark.asyncio
async def test_multiple_calls_stop_without_feedback(agent):
    agent.server_client.post = AsyncMock(side_effect=_single_violation_rollout([_call("one"), _call("two")]))

    result, _ = await agent._rollout(_request(), {})

    _assert_stopped_without_step(agent, result, TerminationReason.MULTIPLE_TOOL_CALLS)


@pytest.mark.asyncio
async def test_truncated_response_stops_even_with_a_parsable_call(agent):
    agent.server_client.post = AsyncMock(
        side_effect=_single_violation_rollout([_call("cut", "terminate")], incomplete=True)
    )

    result, _ = await agent._rollout(_request(), {})

    _assert_stopped_without_step(agent, result, TerminationReason.TRUNCATED)


@pytest.mark.asyncio
async def test_tool_that_was_not_offered_stops(agent):
    tools = [{"type": "function", "name": "computer", "parameters": {}, "strict": False}]
    agent.server_client.post = AsyncMock(
        side_effect=_single_violation_rollout([_call("bogus", "unoffered_tool", {})], tools=tools)
    )

    result, _ = await agent._rollout(_request(), {})

    metadata = _assert_stopped_without_step(agent, result, TerminationReason.INVALID_TOOL_CALL)
    assert "unoffered_tool" in metadata["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("computer", {}),
        ("computer", {"actions": [{"action": "press", "keys": ["enter"]}]}),
        ("computer", {"actions": [{"action": "left_click", "coordinate": [1.5, 0.2]}]}),
        ("terminate", {"status": "done"}),
    ],
)
async def test_invalid_arguments_stop_before_the_environment(agent, name, arguments):
    agent.server_client.post = AsyncMock(side_effect=_single_violation_rollout([_call("bad", name, arguments)]))

    result, _ = await agent._rollout(_request(), {})

    _assert_stopped_without_step(agent, result, TerminationReason.INVALID_TOOL_CALL)


@pytest.mark.asyncio
async def test_tool_execution_error_stops_without_feeding_the_error_back(agent):
    responses = [
        FakeResponse({"env_id": "env-tool-error", "obs": [_env_obs(1, "SEED")], "tools": []}),
        FakeResponse(_model_response("response-1", [_call("call-1")])),
        FakeResponse(
            {
                "obs": [
                    {"type": "function_call_output", "call_id": "call-1", "output": "Tool error: TimeoutError: slow"},
                    _env_obs(2, "LATER"),
                ],
                "done": False,
                "tool_error": "TimeoutError: slow",
            }
        ),
        FakeResponse({"success": True}),
    ]
    agent.server_client.post = AsyncMock(side_effect=responses)

    result, _ = await agent._rollout(_request(), {})

    paths = [item.kwargs["url_path"] for item in agent.server_client.post.await_args_list]
    assert paths == ["/seed_session", "/v1/responses", "/step", "/close"]
    assert "Tool error" not in json.dumps(result.model_dump(mode="json")["output"])
    metadata = json.loads(result.metadata["webarena_termination"])
    assert metadata["reason"] == TerminationReason.TOOL_EXECUTION_ERROR
    assert metadata["environment_steps"] == 0


@pytest.mark.asyncio
async def test_model_failure_is_not_retried_and_still_closes(agent):
    responses = [
        FakeResponse({"env_id": "env-model-error", "obs": [_env_obs(1, "SEED")], "tools": []}),
        FakeResponse({"not": "a response"}),
        FakeResponse({"success": True}),
    ]
    agent.server_client.post = AsyncMock(side_effect=responses)

    with pytest.raises(WebArenaRolloutError) as exc_info:
        await agent._rollout(_request(), {})

    assert exc_info.value.reason == TerminationReason.MODEL_ERROR
    paths = [item.kwargs["url_path"] for item in agent.server_client.post.await_args_list]
    assert paths == ["/seed_session", "/v1/responses", "/close"]


@pytest.mark.asyncio
async def test_close_failure_is_recorded_without_masking_partial_response(agent):
    responses = [
        FakeResponse({"env_id": "env-close-error", "obs": [_env_obs(1, "SEED")], "tools": []}),
        FakeResponse(_model_response("response-multiple", [_call("one"), _call("two")])),
    ]
    agent.server_client.post = AsyncMock(side_effect=responses)
    agent._close = AsyncMock(return_value="RuntimeError('close failed')")

    result, _ = await agent._rollout(_request(), {})

    metadata = json.loads(result.metadata["webarena_termination"])
    assert metadata["reason"] == TerminationReason.MULTIPLE_TOOL_CALLS
    assert metadata["close_error"] == "RuntimeError('close failed')"
    agent._close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_uses_compact_finalize_and_does_not_close_after_success(agent):
    rollout_response = WebArenaNeMoGymResponse.model_validate(
        {
            **_model_response("final", [_call("final-call")]),
            "env_id": "env-4",
            "group_id": "7",
            "contains_transitions": False,
            "seed_obs": [_env_obs(1, "SEED")],
        }
    )
    agent._rollout = AsyncMock(return_value=(rollout_response, {"resource": "cookie"}))

    async def post(server_name=None, url_path=None, json=None, cookies=None):
        return FakeResponse(
            {
                **json,
                "reward": 1.0,
                "evaluator_details": {},
                "termination_status": "success",
                "answer": "done",
            }
        )

    agent.server_client.post = AsyncMock(side_effect=post)
    request = SimpleNamespace(cookies={"incoming": "cookie"})

    result = await agent.run(request, _request())

    assert result.reward == 1.0
    assert result.answer == "done"
    calls = agent.server_client.post.await_args_list
    assert len(calls) == 1
    finalize_call = calls[0]
    assert finalize_call.kwargs["url_path"] == "/finalize"
    assert finalize_call.kwargs["cookies"] == {"resource": "cookie"}
    assert set(finalize_call.kwargs["json"]) == {"env_id", "finalize_request_id"}
    assert finalize_call.kwargs["json"]["env_id"] == "env-4"
    assert finalize_call.kwargs["json"]["finalize_request_id"] == result.verify_request_id
    agent._rollout.assert_awaited_once_with(
        _request(),
        {"incoming": "cookie"},
        close_on_success=False,
        logical_rollout_id=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "expected_reward", "masked"),
    [
        (TerminationReason.ENV_DONE, 1.0, False),
        (TerminationReason.NO_TOOL_CALL, 0.0, False),
        (TerminationReason.INVALID_TOOL_CALL, 0.0, False),
        (TerminationReason.TRUNCATED, 0.0, False),
        (TerminationReason.TOOL_EXECUTION_ERROR, 1.0, True),
        (TerminationReason.ENVIRONMENT_ERROR, 1.0, True),
    ],
)
async def test_run_zeroes_policy_mistakes_and_masks_other_failures(agent, reason, expected_reward, masked):
    rollout_response = WebArenaNeMoGymResponse.model_validate(
        {
            **_model_response("final", [_call("final-call")]),
            "env_id": "env-outcome",
            "group_id": "7",
            "contains_transitions": False,
            "seed_obs": [_env_obs(1, "SEED")],
            "metadata": {
                "webarena_termination": web_agent_app.WebArenaTerminationMetadata(
                    reason=reason, detail="why"
                ).model_dump_json()
            },
        }
    )
    agent._rollout = AsyncMock(return_value=(rollout_response, {}))

    async def post(server_name=None, url_path=None, json=None, cookies=None):
        return FakeResponse({**json, "reward": 1.0, "evaluator_details": {}})

    agent.server_client.post = AsyncMock(side_effect=post)

    result = await agent.run(SimpleNamespace(cookies={}), _request())
    dumped = result.model_dump(mode="json")

    assert result.reward == expected_reward
    assert dumped["webarena_termination_reason"] == reason.value
    assert bool((dumped.get("instance_config") or {}).get("mask_sample")) is masked
    if reason == TerminationReason.ENV_DONE:
        assert result.failure_reason is None
    else:
        assert result.failure_reason == f"{reason.value}: why"


@pytest.mark.asyncio
async def test_run_retries_finalize_with_stable_request_id(agent, monkeypatch):
    rollout_response = WebArenaNeMoGymResponse.model_validate(
        {
            **_model_response("final", [_call("final-call")]),
            "env_id": "env-409",
            "group_id": "7",
            "contains_transitions": False,
            "seed_obs": [_env_obs(1, "SEED")],
        }
    )
    agent._rollout = AsyncMock(return_value=(rollout_response, {}))
    agent._close = AsyncMock(return_value=None)
    attempts = 0

    async def post_json(_server_name, _url_path, payload, _cookies, _timeout_seconds):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("transport timeout")
        return {
            **payload,
            "reward": 1.0,
            "evaluator_details": {},
        }, {}

    agent._post_json = AsyncMock(side_effect=post_json)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    result = await agent.run(SimpleNamespace(cookies={}), _request())

    assert result.reward == 1.0
    assert agent._post_json.await_count == 2
    first_payload = agent._post_json.await_args_list[0].args[2]
    second_payload = agent._post_json.await_args_list[1].args[2]
    assert first_payload == second_payload
    assert first_payload["finalize_request_id"] == result.verify_request_id
    assert sleeps == [1.0]
    agent._close.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_does_not_retry_non_transport_finalize_errors_and_aborts(agent):
    rollout_response = WebArenaNeMoGymResponse.model_validate(
        {
            **_model_response("final", [_call("final-call")]),
            "env_id": "env-500",
            "group_id": "7",
            "contains_transitions": False,
            "seed_obs": [_env_obs(1, "SEED")],
        }
    )
    agent._rollout = AsyncMock(return_value=(rollout_response, {}))
    agent._close = AsyncMock(return_value=None)
    error = aiohttp.ClientResponseError(request_info=MagicMock(), history=(), status=500, message="boom")
    agent._post_json = AsyncMock(side_effect=error)

    with pytest.raises(aiohttp.ClientResponseError):
        await agent.run(SimpleNamespace(cookies={}), _request())

    assert agent._post_json.await_count == 1
    agent._close.assert_awaited_once()


@pytest.mark.asyncio
async def test_responses_propagates_resource_cookies(agent):
    result = MagicMock()
    agent._rollout = AsyncMock(return_value=(result, {"session": "abc"}))
    response = Response()

    returned = await agent.responses(SimpleNamespace(cookies={}), response, _request())

    assert returned is result
    assert "session=abc" in response.headers["set-cookie"]
