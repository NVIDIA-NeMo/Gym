# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.web.models import (
    BROWSER_TARGET_CLOSED_STATUS,
    CAPTCHA_BUDGET_EXHAUSTED_STATUS,
    WebActionProfile,
    WebBenchmark,
    WebTask,
)
from responses_api_agents.web_agent.app import (
    WebAgent,
    WebAgentConfig,
    WebAgentRunRequest,
    _incomplete_model_reason,
    _merge_usage,
    _native_parse_retry_messages,
    _parse_response_action,
    _redact_old_images,
)


def _model_response(text: str) -> dict:
    return {
        "id": "response-1",
        "created_at": 0.0,
        "model": "policy",
        "object": "response",
        "output": [
            {
                "id": "message-1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _native_model_response(name: str, arguments: str) -> dict:
    payload = _model_response("")
    payload["output"] = [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": name,
            "arguments": arguments,
            "status": "completed",
        }
    ]
    return payload


def _length_model_response() -> dict:
    payload = _model_response("")
    payload["status"] = "incomplete"
    payload["incomplete_details"] = {"reason": "max_output_tokens"}
    return payload


def test_incomplete_model_reason_reads_openai_response_details() -> None:
    response = NeMoGymResponse.model_validate(_length_model_response())

    assert _incomplete_model_reason(response) == "max_output_tokens"
    assert _incomplete_model_reason(NeMoGymResponse.model_validate(_model_response("ok"))) is None


def test_merge_usage_accumulates_cached_and_reasoning_tokens() -> None:
    first_payload = _model_response("first")
    first_payload["usage"] = {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 2},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 3},
        "total_tokens": 15,
    }
    second_payload = _model_response("second")
    second_payload["usage"] = {
        "input_tokens": 20,
        "input_tokens_details": {"cached_tokens": 7},
        "output_tokens": 8,
        "output_tokens_details": {"reasoning_tokens": 4},
        "total_tokens": 28,
    }

    usage = _merge_usage(None, NeMoGymResponse.model_validate(first_payload))
    usage = _merge_usage(usage, NeMoGymResponse.model_validate(second_payload))

    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (30, 13, 43)
    assert usage.input_tokens_details.cached_tokens == 9
    assert usage.output_tokens_details.reasoning_tokens == 7


def test_native_profile_reads_structured_function_calls_not_message_text():
    payload = _model_response("")
    payload["output"] = [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "tabs_focus",
            "arguments": '{"tab_id": 2}',
            "status": "completed",
        }
    ]

    action = _parse_response_action(
        NeMoGymResponse.model_validate(payload),
        WebActionProfile.NATIVE_TOOLCALL,
    )

    assert action.name == "tabs_focus"
    assert action.arguments["calls"][0]["arguments"] == {"tab_id": 2}


def test_native_retry_feedback_contains_error_and_no_image() -> None:
    response = NeMoGymResponse.model_validate(_native_model_response("click", '{"x":0.2,"y":0.3}'))

    messages = _native_parse_retry_messages(response, ValueError("unsupported native browser tool: 'click'"))

    assert [message.role for message in messages] == ["assistant", "user"]
    assert "unsupported native browser tool" in messages[1].content
    assert "Use `left_click`, never `click`" in messages[1].content
    assert "arguments.actions" in messages[1].content
    assert "input_image" not in json.dumps([message.model_dump() for message in messages])


@pytest.mark.asyncio
async def test_native_parse_retry_injects_feedback_and_retry_temperature() -> None:
    agent = _agent(
        parse_retries=1,
        judge=True,
        native_parse_retry_feedback=True,
        native_parse_retry_temperature=0.2,
        model_retry_delay_secs=0,
    )
    final_observation = _observation("https://example.test/final")
    final_observation["screenshot"]["data_url"] = "data:image/png;base64,final"
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed("Allrecipes--0")],
            "/v1/responses": [
                _native_model_response("click", '{"coordinate":[0.2,0.3]}'),
                _native_model_response("terminate", '{"status":"success","answer":"done"}'),
            ],
            "/step": [
                {
                    "operation_id": "step-0",
                    "observation": final_observation,
                    "execution_ok": True,
                    "terminated": True,
                }
            ],
            "/evaluate": [{"result": {"valid_sample": False, "failure_kind": "external_judge_required"}}],
            "/verify": [
                {
                    "reward": 1.0,
                    "raw_score": 1.0,
                    "task_success": True,
                    "mask_sample": False,
                }
            ],
            "/close": [{"closed": True}],
        },
    )
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": [], "temperature": 0.1},
        web_task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="Allrecipes--0",
            intent="Find a recipe",
            start_urls=["https://example.test"],
            runtime_profile="native_visual",
            observation_profile="screenshot",
            action_profile="native_toolcall",
        ),
    )

    result = await agent.run(request, body)

    model_bodies = [call_body for _server, path, call_body in calls if path == "/v1/responses"]
    assert result.model_turns == 2
    assert len(model_bodies) == 2
    assert model_bodies[0].temperature == 0.1
    assert model_bodies[1].temperature == 0.2
    assert sum("input_image" in json.dumps(item.model_dump()) for item in model_bodies[1].input) == 1
    assert any(
        getattr(item, "role", None) == "user" and "arguments.actions" in str(item.content)
        for item in model_bodies[1].input
    )
    judge_body = next(call_body for _server, path, call_body in calls if path == "/verify")
    assert judge_body["screenshots"][-1] == "data:image/png;base64,final"
    assert judge_body["page_urls"][-1] == "https://example.test/final"


@pytest.mark.asyncio
async def test_native_length_response_ends_as_valid_truncation_without_parse_retry() -> None:
    agent = _agent(parse_retries=2, judge=True)
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed("Huggingface--18")],
            "/v1/responses": [_length_model_response()],
            "/evaluate": [{"result": {"valid_sample": False, "failure_kind": "external_judge_required"}}],
            "/verify": [
                {
                    "reward": 0.0,
                    "raw_score": 0.0,
                    "task_success": False,
                    "mask_sample": False,
                }
            ],
            "/close": [{"closed": True}],
        },
    )
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": [], "temperature": 0.1},
        web_task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="Huggingface--18",
            intent="Find documentation",
            start_urls=["https://huggingface.co"],
            runtime_profile="native_visual",
            observation_profile="screenshot",
            action_profile="native_toolcall",
        ),
    )

    result = await agent.run(request, body)

    model_calls = [path for _server, path, _body in calls if path == "/v1/responses"]
    assert model_calls == ["/v1/responses"]
    assert result.model_turns == 1
    assert result.environment_steps == 0
    assert result.truncated is True
    assert result.mask_sample is False
    assert result.failure_kind is None


def _observation(url="https://example.test") -> dict:
    return {
        "goal": [{"type": "text", "text": "Complete the task"}],
        "axtree_text": "[a1] button 'Done'",
        "screenshot": {"data_url": "data:image/png;base64,abc"},
        "url": url,
    }


class _FakeHttpResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.cookies = {}
        self.status = 200
        self.ok = True

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode()

    @property
    def content(self):
        payload = self._payload

        class _Body:
            async def read(self):
                return json.dumps(payload).encode()

        return _Body()

    def raise_for_status(self):
        return None


def _agent(*, parse_retries=1, judge=False, **config_updates):
    config_values = dict(
        name="web_agent",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(
            type="resources_servers",
            name="judge" if judge else "browser",
        ),
        environment_server=(ResourcesServerRef(type="resources_servers", name="browser") if judge else None),
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        max_parse_retries=parse_retries,
    )
    config_values.update(config_updates)
    config = WebAgentConfig(**config_values)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    return WebAgent(config=config, server_client=client)


def _wire(agent, payloads):
    calls = []

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        calls.append((server_name, url_path, json))
        return _FakeHttpResponse(payloads[url_path].pop(0))

    agent.server_client.post = AsyncMock(side_effect=post)
    return calls


def _seed(task_id="0"):
    return {
        "session_id": "session-a",
        "task_id": task_id,
        "status": "ready",
        "observation": _observation(),
        "info": {},
    }


def _recording():
    return {
        "uri": "file:///artifacts/session-a/video/task.webm",
        "mime_type": "video/webm",
        "size_bytes": 123,
        "sha256": "a" * 64,
    }


def test_redact_old_visual_observation_removes_both_image_and_page_text():
    old = NeMoGymEasyInputMessage(
        role="user",
        content=[
            {"type": "input_text", "text": "large page tree"},
            {"type": "input_image", "image_url": "data:image/png;base64,old", "detail": "high"},
        ],
    )
    current = NeMoGymEasyInputMessage(
        role="user",
        content=[
            {"type": "input_text", "text": "current page tree"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,current",
                "detail": "high",
            },
        ],
    )

    redacted = _redact_old_images(
        [old, current],
        1,
        redact_observation_text=True,
    )

    assert redacted[0].content == [
        {
            "type": "input_text",
            "text": "[Earlier screenshot and page text omitted from context.]",
        }
    ]
    assert len(redacted[1].content) == 2
    assert old.content[0]["text"] == "large page tree"


def test_native_image_history_preserves_task_images_while_compacting_browser_screenshots():
    first = NeMoGymEasyInputMessage(
        role="user",
        content=[
            {"type": "input_image", "image_url": "data:image/png;base64,browser-0", "detail": "high"},
            {"type": "input_text", "text": "Task image 1 of 1:"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,task",
                "detail": "high",
            },
        ],
    )
    later = [
        NeMoGymEasyInputMessage(
            role="user",
            content=[{"type": "input_image", "image_url": f"data:image/png;base64,browser-{index}", "detail": "high"}],
        )
        for index in range(1, 5)
    ]

    redacted = _redact_old_images(
        [first, *later],
        3,
        append_redaction_notice=False,
    )

    assert [block["image_url"] for block in redacted[0].content if block.get("type") == "input_image"] == [
        "data:image/png;base64,task"
    ]
    assert sum(block.get("type") == "input_image" for message in redacted for block in message.content) == 4


@pytest.mark.parametrize("benchmark", [WebBenchmark.WEBARENA, WebBenchmark.VISUALWEBARENA])
@pytest.mark.asyncio
async def test_arena_family_rollout_uses_colocated_evaluator_and_closes_session(benchmark):
    agent = _agent()
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed()],
            "/v1/responses": [_model_response("Thought: done\nAction: send_msg_to_user('answer')")],
            "/step": [
                {
                    "operation_id": "step-0",
                    "observation": _observation(),
                    "execution_ok": True,
                    "benchmark_reward": 1.0,
                    "terminated": True,
                    "truncated": False,
                    "info": {},
                }
            ],
            "/evaluate": [
                {
                    "result": {
                        "reward": 1.0,
                        "raw_score": 1.0,
                        "task_success": True,
                        "valid_sample": True,
                    }
                }
            ],
            "/close": [
                {
                    "closed": True,
                    "session_id": "session-a",
                    "recording_artifacts": [_recording()],
                }
            ],
        },
    )
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve the task"},
        web_task=WebTask(benchmark=benchmark, task_id="0"),
    )

    result = await agent.run(request, body)

    assert result.reward == 1.0
    assert result.task_success is True
    assert result.environment_steps == 1
    assert result.mask_sample is False
    assert result.artifact_session_id == "session-a"
    assert result.recording_artifacts[0].mime_type == "video/webm"
    step_body = next(body for _server, path, body in calls if path == "/step")
    assert step_body["action"]["script"] == "send_msg_to_user('answer')"
    assert [path for _server, path, _body in calls][-2:] == ["/evaluate", "/close"]
    assert calls[-1][1] == "/close"


@pytest.mark.asyncio
async def test_action_parse_failure_is_retried_without_stepping_browser():
    agent = _agent(parse_retries=1)
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed()],
            "/v1/responses": [
                _model_response("I forgot the action"),
                _model_response("Thought: retry\nAction: send_msg_to_user('answer')"),
            ],
            "/step": [
                {
                    "operation_id": "step-0",
                    "observation": _observation(),
                    "execution_ok": True,
                    "benchmark_reward": 0.0,
                    "terminated": True,
                    "truncated": False,
                }
            ],
            "/evaluate": [{"result": {"valid_sample": True}}],
            "/close": [{"closed": True}],
        },
    )
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0"),
    )

    result = await agent.run(request, body)

    assert result.model_turns == 2
    assert result.environment_steps == 1
    assert [path for _server, path, _body in calls].count("/step") == 1


@pytest.mark.asyncio
async def test_webvoyager_routes_final_evidence_to_external_judge(caplog):
    agent = _agent(judge=True)
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed("Allrecipes--0")],
            "/v1/responses": [_model_response("Thought: done\nAction: ANSWER; [42]")],
            "/step": [
                {
                    "operation_id": "step-0",
                    "observation": _observation("https://example.test/result"),
                    "execution_ok": True,
                }
            ],
            "/evaluate": [
                {
                    "result": {
                        "valid_sample": False,
                        "failure_kind": "external_judge_required",
                    }
                }
            ],
            "/verify": [
                {
                    "reward": 1.0,
                    "raw_score": 1.0,
                    "task_success": True,
                    "mask_sample": False,
                }
            ],
            "/close": [{"closed": True}],
        },
    )
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="Allrecipes--0",
            intent="Find the answer",
            start_urls=["https://example.test"],
            action_profile="webvoyager_legacy",
        ),
    )

    with caplog.at_level(logging.INFO, logger="nemo_gym.responses_api_agents.web_agent"):
        result = await agent.run(request, body)

    assert result.reward == 1.0
    judge_call = next(call for call in calls if call[1] == "/verify")
    assert judge_call[0] == "judge"
    assert judge_call[2]["final_answer"] == "42"
    assert len(judge_call[2]["screenshots"]) == 2
    judge_response = judge_call[2]["response"]
    assert judge_response["output"] == []
    assert "webvoyager_judge_evidence" not in judge_response
    persisted_response = result.response.model_dump(mode="json")
    assert persisted_response["output"]
    persisted_evidence = persisted_response["webvoyager_judge_evidence"]
    assert persisted_evidence["final_answer"] == "42"
    assert "screenshots" not in persisted_evidence
    assert len(persisted_evidence["screenshot_sequence"]) == 2
    paths = [path for _server, path, _body in calls]
    assert paths.index("/evaluate") < paths.index("/close") < paths.index("/verify")
    messages = "\n".join(record.getMessage() for record in caplog.records)
    for event in (
        "event=web_rollout_start",
        "event=web_seed_complete",
        "event=web_model_turn_complete",
        "event=web_action_parsed",
        "event=web_environment_step_complete",
        "event=web_session_close_complete",
        "event=web_judge_complete",
        "event=web_rollout_complete",
    ):
        assert event in messages
    assert "data:image/png;base64,abc" not in messages


@pytest.mark.asyncio
async def test_webvoyager_preserves_standard_judge_failure_after_browser_is_closed():
    agent = _agent(judge=True)
    calls = []
    judge_attempts = 0

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        nonlocal judge_attempts
        del cookies, kwargs
        calls.append((server_name, url_path, json))
        if url_path == "/seed_session":
            return _FakeHttpResponse(_seed("ArXiv--0"))
        if url_path == "/v1/responses":
            return _FakeHttpResponse(_model_response("Thought: done\nAction: ANSWER; [42]"))
        if url_path == "/step":
            return _FakeHttpResponse(
                {
                    "operation_id": "step-0",
                    "observation": _observation("https://example.test/result"),
                    "execution_ok": True,
                }
            )
        if url_path == "/evaluate":
            return _FakeHttpResponse({"result": {"valid_sample": True}})
        if url_path == "/close":
            return _FakeHttpResponse({"closed": True, "session_id": "session-a"})
        if url_path == "/verify":
            judge_attempts += 1
            return _FakeHttpResponse(
                {
                    "reward": 0.0,
                    "_ng_failure_class": "judge_failed",
                    "_ng_failure_judge_error": "TimeoutError: judge endpoint is warming",
                }
            )
        raise AssertionError(f"unexpected path: {url_path}")

    agent.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="ArXiv--0",
            intent="Find the answer",
            start_urls=["https://example.test"],
            action_profile="webvoyager_legacy",
        ),
    )

    result = await agent.run(request, body)

    assert result.reward == 0.0
    assert result.mask_sample is True
    assert result.failure_kind == "judge_failed"
    assert result.model_dump()["_ng_failure_class"] == "judge_failed"
    assert judge_attempts == 1
    paths = [path for _server, path, _body in calls]
    assert paths.count("/seed_session") == 1
    assert paths.count("/step") == 1
    assert paths.count("/close") == 1
    assert paths.count("/verify") == 1
    assert paths.index("/close") < paths.index("/verify")


@pytest.mark.asyncio
async def test_browser_request_timeout_is_retryable_and_cleanup_is_bounded():
    agent = _agent(
        resources_request_timeout_secs=0.01,
        close_request_timeout_secs=0.1,
        run_timeout_secs=1.0,
    )
    calls = []

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        del server_name, json, cookies, kwargs
        calls.append(url_path)
        if url_path == "/seed_session":
            return _FakeHttpResponse(_seed())
        if url_path == "/v1/responses":
            return _FakeHttpResponse(_model_response("Thought: click\nAction: click('a1')"))
        if url_path == "/step":
            await asyncio.sleep(1.0)
        if url_path == "/close":
            return _FakeHttpResponse(
                {
                    "closed": True,
                    "session_id": "session-a",
                    "recording_artifacts": [_recording()],
                }
            )
        raise AssertionError(f"unexpected path: {url_path}")

    agent.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0"),
    )

    result = await agent.run(request, body)
    dumped = result.model_dump()

    assert result.reward == 0.0
    assert result.mask_sample is True
    assert result.failure_kind == "infrastructure_error:TimeoutError"
    assert dumped["_ng_failure_class"] == "retryable_infrastructure"
    assert "_ng_no_persist" not in dumped
    assert result.artifact_session_id == "session-a"
    assert result.recording_artifacts[0].uri.endswith("/task.webm")
    assert "/close" in calls


@pytest.mark.asyncio
async def test_model_context_overflow_skips_futile_request_retries_but_remains_rollout_retryable():
    agent = _agent(model_turn_max_retries=20, model_retry_delay_secs=0)
    model_attempts = 0
    model_headers = {}
    overflow = ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=500,
        message="Internal Server Error",
    )
    overflow.response_content = json.dumps(
        {
            "error": {
                "message": ("The decoder prompt (length 128107) is longer than the maximum model length of 128000."),
                "type": "BadRequestError",
                "code": 400,
            }
        }
    ).encode()

    async def post_json(*, url_path, **kwargs):
        nonlocal model_attempts, model_headers
        if url_path == "/seed_session":
            response = _FakeHttpResponse(_seed())
            return response, await response.json()
        if url_path == "/v1/responses":
            model_attempts += 1
            model_headers = kwargs["headers"]
            raise overflow
        if url_path == "/close":
            response = _FakeHttpResponse({"closed": True, "session_id": "session-a"})
            return response, await response.json()
        raise AssertionError(f"unexpected path: {url_path}")

    agent._post_json = AsyncMock(side_effect=post_json)
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="Google Map--14",
        ),
    )

    result = await agent.run(request, body)
    dumped = result.model_dump()

    assert model_attempts == 1
    assert model_headers == {
        "x-nemo-gym-log-adapter": "web_agent",
        "x-nemo-gym-log-task-id": "Google Map--14",
        "x-nemo-gym-log-domain": "webvoyager",
        "x-nemo-gym-log-step": "0",
        "x-nemo-gym-log-parse-attempt": "0",
    }
    assert result.mask_sample is True
    assert result.failure_kind == "model_context_overflow"
    assert dumped["_ng_failure_class"] == "retryable_infrastructure"
    assert "_ng_failure_terminal" not in dumped
    assert result.verifier_result.metadata["error_kind"] == "model_context_overflow"


@pytest.mark.asyncio
async def test_seed_session_uses_independent_long_poll_timeout():
    agent = _agent(
        resources_request_timeout_secs=0.01,
        seed_request_timeout_secs=0.2,
        run_timeout_secs=1.0,
    )

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        del server_name, json, cookies, kwargs
        if url_path == "/seed_session":
            await asyncio.sleep(0.05)
            return _FakeHttpResponse(_seed())
        if url_path == "/v1/responses":
            return _FakeHttpResponse(_model_response("Thought: done\nAction: send_msg_to_user('done')"))
        if url_path == "/step":
            return _FakeHttpResponse(
                {
                    "operation_id": "step-0",
                    "observation": _observation(),
                    "execution_ok": True,
                    "terminated": True,
                }
            )
        if url_path == "/evaluate":
            return _FakeHttpResponse({"result": {"valid_sample": True}})
        if url_path == "/close":
            return _FakeHttpResponse({"closed": True})
        raise AssertionError(f"unexpected path: {url_path}")

    agent.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0"),
    )

    result = await agent.run(request, body)

    assert result.failure_kind is None
    assert result.environment_steps == 1


@pytest.mark.asyncio
async def test_seed_session_retries_transient_server_failure():
    agent = _agent(
        seed_request_timeout_secs=1.0,
        seed_retry_initial_delay_secs=0.0,
        seed_retry_max_delay_secs=0.0,
    )
    attempts = 0

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        nonlocal attempts
        del server_name, json, cookies, kwargs
        if url_path == "/seed_session":
            attempts += 1
            if attempts == 1:
                error = ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=503,
                    message="site temporarily unavailable",
                )
                error.response_content = b"site temporarily unavailable"
                raise error
            return _FakeHttpResponse(_seed())
        if url_path == "/v1/responses":
            return _FakeHttpResponse(_model_response("Thought: done\nAction: send_msg_to_user('done')"))
        if url_path == "/step":
            return _FakeHttpResponse(
                {
                    "operation_id": "step-0",
                    "observation": _observation(),
                    "execution_ok": True,
                    "terminated": True,
                }
            )
        if url_path == "/evaluate":
            return _FakeHttpResponse({"result": {"valid_sample": True}})
        if url_path == "/close":
            return _FakeHttpResponse({"closed": True})
        raise AssertionError(f"unexpected path: {url_path}")

    agent.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0"),
    )

    result = await agent.run(request, body)

    assert attempts == 2
    assert result.failure_kind is None


@pytest.mark.asyncio
async def test_seed_session_does_not_retry_client_failure():
    agent = _agent(
        seed_request_timeout_secs=1.0,
        seed_retry_initial_delay_secs=0.0,
        seed_retry_max_delay_secs=0.0,
    )
    error = ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=400,
        message="bad task",
    )
    agent._post_json = AsyncMock(side_effect=error)

    with pytest.raises(ClientResponseError, match="bad task"):
        await agent._seed_session(
            task=WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0"),
            cookies={},
        )

    agent._post_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_classifies_seed_precondition_as_terminal_masked_failure():
    agent = _agent(seed_request_timeout_secs=1.0)
    error = ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=422,
        message="benchmark setup failed",
    )
    error.response_content = json.dumps(
        {
            "detail": "Could not download image: HTTP 404",
            "error_kind": "benchmark_precondition",
            "retryable": False,
        }
    ).encode()
    agent._post_json = AsyncMock(side_effect=error)
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(benchmark=WebBenchmark.VISUALWEBARENA, task_id="36"),
    )

    result = await agent.run(request, body)
    dumped = result.model_dump()

    assert result.mask_sample is True
    assert result.failure_kind == "benchmark_precondition"
    assert dumped["_ng_failure_class"] == "benchmark_precondition"
    assert dumped["_ng_failure_terminal"] is True
    assert result.verifier_result.metadata["http_status"] == 422
    assert result.verifier_result.metadata["error_kind"] == "benchmark_precondition"
    assert "Could not download image" in dumped["error"]
    assert result.artifact_session_id is None
    assert result.recording_artifacts == []
    agent._post_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_classifies_missing_evaluator_as_terminal_configuration_failure():
    agent = _agent(seed_request_timeout_secs=1.0)
    error = ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=422,
        message="evaluator is not configured",
    )
    error.response_content = json.dumps(
        {
            "detail": "configure webarena_evaluator_model",
            "error_kind": "evaluator_configuration",
            "retryable": False,
        }
    ).encode()
    agent._post_json = AsyncMock(side_effect=error)
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(benchmark=WebBenchmark.WEBARENA, task_id="8"),
    )

    result = await agent.run(request, body)
    dumped = result.model_dump()

    assert result.mask_sample is True
    assert result.failure_kind == "evaluator_configuration"
    assert dumped["_ng_failure_class"] == "configuration_error"
    assert dumped["_ng_failure_terminal"] is True
    assert result.verifier_result.metadata["error_kind"] == "evaluator_configuration"
    agent._post_json.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_status,action_error",
    [
        (
            CAPTCHA_BUDGET_EXHAUSTED_STATUS,
            "RuntimeError: Captcha solver failed more than 3 times",
        ),
    ],
)
async def test_environment_access_failure_is_masked_instead_of_judged(
    caplog,
    native_status,
    action_error,
):
    """A site the browser cannot reach makes the policy's work unmeasurable."""

    agent = _agent(judge=True)
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed("Allrecipes--0")],
            "/v1/responses": [_model_response("Thought: keep going\nAction: Click [7]")],
            "/step": [
                {
                    "operation_id": "step-0",
                    "observation": _observation("https://example.test/challenge"),
                    "execution_ok": False,
                    "terminated": True,
                    "info": {
                        "action_error": action_error,
                        "native_status": native_status,
                    },
                }
            ],
            "/evaluate": [{"result": {"valid_sample": False, "failure_kind": "external_judge_required"}}],
            "/close": [{"closed": True}],
        },
    )
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="Allrecipes--0",
            intent="Find the answer",
            start_urls=["https://example.test"],
            action_profile="webvoyager_legacy",
        ),
    )

    with caplog.at_level(logging.WARNING, logger="nemo_gym.responses_api_agents.web_agent"):
        result = await agent.run(request, body)

    assert result.mask_sample is True
    assert result.failure_kind == native_status
    assert result.reward == 0.0
    assert result.task_success is False
    # Judging a forced stop would score a site-access failure as a policy failure.
    assert "/verify" not in [path for _server, path, _body in calls]
    assert "/close" in [path for _server, path, _body in calls]
    assert "event=web_environment_access_failed" in "\n".join(record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_browser_target_closed_after_action_is_judged_as_policy_failure(caplog):
    agent = _agent(judge=True)
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed("ESPN--10")],
            "/v1/responses": [
                _native_model_response("computer", '{"actions":[{"action":"left_click","coordinate":[0.08,0.015]}]}')
            ],
            "/step": [
                {
                    "operation_id": "step-0",
                    "observation": _observation("https://www.espn.com/"),
                    "execution_ok": False,
                    "terminated": True,
                    "info": {
                        "action_error": "BrowserTargetClosedDuringCaptcha: browser target closed",
                        "native_status": BROWSER_TARGET_CLOSED_STATUS,
                    },
                }
            ],
            "/evaluate": [{"result": {"valid_sample": False, "failure_kind": "external_judge_required"}}],
            "/verify": [
                {
                    "reward": 0.0,
                    "raw_score": 0.0,
                    "task_success": False,
                    "mask_sample": False,
                }
            ],
            "/close": [{"closed": True}],
        },
    )
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": [], "temperature": 0.1},
        web_task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="ESPN--10",
            intent="Find the latest championship recap",
            start_urls=["https://www.espn.com/"],
            runtime_profile="native_visual",
            observation_profile="screenshot",
            action_profile="native_toolcall",
        ),
    )

    with caplog.at_level(logging.WARNING, logger="nemo_gym.responses_api_agents.web_agent"):
        result = await agent.run(request, body)

    paths = [path for _server, path, _body in calls]
    assert result.reward == 0.0
    assert result.task_success is False
    assert result.mask_sample is False
    assert result.failure_kind is None
    assert paths.count("/verify") == 1
    assert paths.index("/close") < paths.index("/verify")
    judge_body = next(call_body for _server, path, call_body in calls if path == "/verify")
    assert len(judge_body["screenshots"]) == 1
    assert judge_body["page_urls"] == ["https://example.test"]
    assert "event=web_environment_target_closed_after_action" in "\n".join(
        record.getMessage() for record in caplog.records
    )
