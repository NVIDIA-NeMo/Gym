# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi import HTTPException, Response
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from pydantic import ValidationError
from pytest import MonkeyPatch
from synclaw.rollout.loop import ControlledTermination, RolloutResult, RolloutState, run_rollout
from synclaw.rollout.policy import DEFAULT_MAX_MODEL_LEN, RolloutPolicyState, prepare_tool_call
from synclaw.rollout.prompts import PROMPT_FINGERPRINT
from synclaw.token_usage import usage_from_payload
from synclaw.tools.contracts import TOOL_NAMES, filter_responses_tool_schemas
from yarl import URL

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from resources_servers.synclaw.app import SynClawResourcesServer
from resources_servers.synclaw.schema import SYNCLAW_SUBMISSION_KEY, SynClawRunRequest, SynClawSeedSessionRequest
from resources_servers.synclaw.task_data import TaskData
from responses_api_agents.synclaw_agent.app import SynClawAgent, SynClawAgentConfig, _GymTransport


def _make_agent(**config_overrides) -> tuple[SynClawAgent, MagicMock]:
    config = SynClawAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="synclaw_agent",
        model_server=ModelServerRef(type="responses_api_models", name="model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        **config_overrides,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"observability_enabled": False}
    return SynClawAgent(config=config, server_client=server_client), server_client


def _mock_response(payload=None, *, status=200) -> MagicMock:
    r = MagicMock(status=status, ok=status < 400, cookies={})
    r.read = AsyncMock(return_value=json.dumps(payload or {}).encode())
    return r


def _mock_error_response(body: str, *, status: int) -> MagicMock:
    """A response that drives nemo_gym.client.http_client.raise_for_status into raising."""
    url = URL("http://test/v1/responses")
    request_info = RequestInfo(url=url, method="POST", headers=CIMultiDict(), real_url=url)
    error = ClientResponseError(request_info, (), status=status, message=body, headers=CIMultiDict())
    r = MagicMock(status=status, ok=False, cookies={}, request_info=request_info)
    r.content.read = AsyncMock(return_value=body.encode())
    r.raise_for_status = MagicMock(side_effect=error)
    return r


# --- Responses fixtures -------------------------------------------------------------------

_USAGE = {
    "input_tokens": 100,
    "output_tokens": 50,
    "total_tokens": 150,
    "input_tokens_details": {"cached_tokens": 20},
    "output_tokens_details": {"reasoning_tokens": 10},
}

_REASONING_ITEM = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "plan it"}]}
_CALL_ITEM = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call-1",
    "name": "submit_pathway",
    "arguments": '{"steps": []}',
}
_MESSAGE_ITEM = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "I found it.", "annotations": []}],
}


def _responses_payload(
    output: list[dict], *, status: str = "completed", incomplete_details=None, usage: dict | None = None
) -> dict:
    """A Responses payload shaped the way a Gym model server really emits one.

    Round-tripping through NeMoGymResponse reproduces the explicit nulls a real server sends
    (``reasoning.content``, ``function_call.namespace``, ``message.phase``, ...), because the
    server serializes with ``model_dump(mode="json")`` and no ``exclude_none``.
    """
    response = NeMoGymResponse.model_validate(
        {
            "object": "response",
            "id": "resp_1",
            "created_at": 0,
            "model": "test-model",
            "tool_choice": "auto",
            "tools": [],
            "parallel_tool_calls": False,
            "status": status,
            "incomplete_details": incomplete_details,
            "usage": usage or _USAGE,
            "output": output,
        }
    )
    return response.model_dump(mode="json")


def _make_transport(post, *, seed: list[dict] | None = None, **kwargs) -> _GymTransport:
    server_client = MagicMock(spec=ServerClient)
    server_client.post = post
    return _GymTransport(
        server_client=server_client,
        model_server_name="model",
        input_history=seed if seed is not None else [{"type": "message", "role": "user", "content": "Make ethanol."}],
        max_tokens=256,
        reasoning_effort="high",
        model_semaphore=asyncio.Semaphore(1),
        **kwargs,
    )


def _capturing_post(payloads: list[dict], recorded: list[dict]):
    """Serve ``payloads`` in order, recording each outbound request body."""
    remaining = list(payloads)

    async def post(**kwargs):
        recorded.append(kwargs)
        return _mock_response(remaining.pop(0))

    return post


def _prepared_call(call_id: str, name: str = "submit_pathway", arguments: dict | None = None):
    return prepare_tool_call(
        raw_name=name,
        raw_arguments=json.dumps(arguments if arguments is not None else {"steps": []}),
        raw_tool_call={"id": call_id},
        event_index=0,
    )


def _make_rollout_result(
    messages: list | None = None,
    submission: dict | None = None,
    **kwargs: Any,
) -> RolloutResult:
    state = RolloutState(
        messages=messages or [],
        events=[],
        policy=RolloutPolicyState(),
    )
    return RolloutResult(
        state=state,
        reason="terminal_submission",
        submission=submission,
        submission_arguments=None,
        **kwargs,
    )


def _base_run_request() -> SynClawRunRequest:
    return SynClawRunRequest(
        task_id="task-1",
        target_smiles="CCO",
        system_prompt="You are a chemist.",
        question="Synthesize ethanol.",
    )


class TestPromptFingerprint:
    @pytest.mark.parametrize("fields", [{}, {"prompt_fingerprint": None}, {"prompt_fingerprint": PROMPT_FINGERPRINT}])
    def test_current_and_custom_tasks_are_supported(self, fields: dict[str, str | None]) -> None:
        payload = _base_run_request().model_dump(exclude_unset=True) | fields
        request = SynClawRunRequest.model_validate(payload)
        assert request.prompt_fingerprint == fields.get("prompt_fingerprint")
        assert request.system_prompt == payload["system_prompt"]

    @pytest.mark.parametrize("path", ["/run", "/v1/responses"])
    @pytest.mark.parametrize("fingerprint", ["e1c6e40d5f82", ""])
    def test_stale_prompts_are_rejected_before_downstream_calls(self, path: str, fingerprint: str) -> None:
        agent, server_client = _make_agent()
        payload = _base_run_request().model_dump() | {"prompt_fingerprint": fingerprint}
        with TestClient(agent.setup_webserver()) as client:
            response = client.post(path, json=payload)
        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["body", "prompt_fingerprint"]
        assert "Regenerate the taskset with `synclaw make-taskset`" in error["msg"]
        server_client.post.assert_not_called()


class TestTaskBudgets:
    def test_agent_tool_cap_is_nonnegative_with_zero_unlimited(self) -> None:
        with pytest.raises(ValidationError):
            _make_agent(max_tool_calls=-1)
        agent, _ = _make_agent(max_tool_calls=0)
        assert agent._task_policy(_base_run_request()).max_tool_calls == 0

    @pytest.mark.parametrize("agent_limit", [None, 262144])
    @pytest.mark.parametrize("path", ["/run", "/v1/responses"])
    def test_endpoint_rejects_oversized_task_before_downstream_calls(self, agent_limit, path) -> None:
        agent, client = _make_agent(max_model_len=agent_limit)
        agent._fetch_served_max_model_len = AsyncMock(return_value=32768)
        with TestClient(agent.setup_webserver()) as web:
            response = web.post(path, json=_base_run_request().model_dump() | {"max_model_len": 65536})
        assert response.status_code == 422
        assert "this task requires 65536" in response.json()["detail"]
        client.post.assert_not_called()

    @pytest.mark.parametrize("agent_limit", [None, 262144])
    async def test_lower_task_context_fits_endpoint_and_metadata_is_cached(self, agent_limit) -> None:
        agent, _ = _make_agent(max_model_len=agent_limit)
        agent._fetch_served_max_model_len = AsyncMock(return_value=32768)
        for limit in (16384, 32768):
            body = SynClawRunRequest.model_validate(_base_run_request().model_dump() | {"max_model_len": limit})
            config = agent._task_policy(body)
            await agent._require_served_max_model_len(config.max_model_len)
        agent._fetch_served_max_model_len.assert_awaited_once()

    @pytest.mark.parametrize("field", ["max_turns", "max_tokens"])
    def test_unlimited_budgets_require_known_context(self, field: str) -> None:
        agent, client = _make_agent(max_model_len=None)
        with TestClient(agent.setup_webserver()) as web:
            response = web.post("/run", json=_base_run_request().model_dump() | {field: "inf"})
        assert response.status_code == 422
        client.post.assert_not_called()

    @pytest.mark.parametrize("value", [True, "40", 40.0, 0, -1])
    def test_task_data_rejects_non_integer_budgets(self, value: object) -> None:
        for field in ("max_turns", "max_tool_calls", "max_model_len", "max_tokens"):
            with pytest.raises(ValidationError):
                TaskData.model_validate(_base_run_request().model_dump() | {field: value})

    def test_task_data_accepts_unlimited_budgets_and_policy_flags(self) -> None:
        fields = {name: "inf" for name in ("max_turns", "max_tool_calls", "max_model_len", "max_tokens")}
        fields.update(budget_feedback=False)
        row = TaskData.model_validate(_base_run_request().model_dump() | fields)
        assert all(row.model_dump()[name] == value for name, value in fields.items())

    @pytest.mark.parametrize("path", ["/run", "/v1/responses"])
    @pytest.mark.parametrize(
        "fields",
        [
            {field: value}
            for field in ("max_turns", "max_tool_calls", "max_model_len", "max_tokens")
            for value in (True, "40", 40.0, 0, -1)
        ]
        + [
            {"max_model_len": DEFAULT_MAX_MODEL_LEN + 1},
            {"max_model_len": 512},
        ]
        + [{field: value} for field in ("budget_feedback",) for value in ("false", 0)],
    )
    def test_invalid_budgets_fail_before_downstream_calls(self, path: str, fields: dict) -> None:
        agent, client = _make_agent()
        with TestClient(agent.setup_webserver()) as web:
            response = web.post(path, json=_base_run_request().model_dump() | fields)
        assert response.status_code == 422
        client.post.assert_not_called()

    def test_null_and_missing_budgets_inherit_defaults(self) -> None:
        agent, _ = _make_agent()
        for fields in (
            {},
            {name: None for name in agent._policy.budget_fields if name != "context_safety_margin_tokens"},
        ):
            body = SynClawRunRequest.model_validate(_base_run_request().model_dump() | fields)
            assert agent._task_policy(body) == agent._policy


@pytest.fixture
def mock_run_rollout(monkeypatch: MonkeyPatch) -> AsyncMock:
    """Stand in for run_rollout, driving the transport the way the real loop does.

    The agent's response output now comes from the transport's history rather than from
    ``state.messages``, so a fixture that only returns a RolloutResult would leave the output
    empty and let these assertions pass vacuously.
    """

    async def fake_run_rollout(*, transport, **kwargs):
        transport.input_history.extend([_REASONING_ITEM, _CALL_ITEM])
        transport.append_tool(_prepared_call("call-1"), '{"similarity_to_target": 0.9}')
        transport.input_history.append(_MESSAGE_ITEM)
        return _make_rollout_result(
            submission={"constitutional_similarity_to_target": 0.9, "similarity_to_target": 0.9},
        )

    mock = AsyncMock(side_effect=fake_run_rollout)
    monkeypatch.setattr("responses_api_agents.synclaw_agent.app.run_rollout", mock)
    return mock


class TestSanity:
    def test_construction(self) -> None:
        agent, _ = _make_agent()
        assert isinstance(agent, SynClawAgent)

    def test_webserver_setup(self) -> None:
        agent, _ = _make_agent()
        app = agent.setup_webserver()
        routes = {r.path for r in app.routes}
        assert "/v1/responses" in routes
        assert "/run" in routes
        assert "/aggregate_metrics" in routes


class TestConfig:
    @pytest.mark.parametrize(
        "config",
        [
            {"context_warning_remaining_fraction": 0.2, "context_critical_remaining_fraction": 0.3},
        ],
    )
    def test_invalid_policy_fails_at_configuration(self, config) -> None:
        with pytest.raises(ValidationError):
            _make_agent(**config)

    @pytest.mark.parametrize(
        "condition,expected", [("exact", "exact"), (" EXACT ", "exact"), ("model", "model"), ("0.8", 0.8), (0, 0.0)]
    )
    def test_stop_condition(self, condition, expected) -> None:
        agent, _ = _make_agent(stop_condition=condition)
        assert agent.config.stop_condition == expected

    @pytest.mark.parametrize("condition", [-0.1, 1.1, "nan", "unknown", True])
    def test_invalid_stop_condition(self, condition) -> None:
        with pytest.raises(ValidationError):
            _make_agent(stop_condition=condition)

    @pytest.mark.parametrize("effort", ["xhigh", "max"])
    def test_reasoning_effort_accepts_each_endpoints_top_effort(self, effort) -> None:
        # OpenAI's top effort is xhigh; the patched vllm-router's is max.
        assert _make_agent(reasoning_effort=effort)[0].config.reasoning_effort == effort

    def test_reasoning_effort_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            _make_agent(reasoning_effort="ultra")


class TestGymTransport:
    @pytest.mark.parametrize("first_outcome", ["success", "error", "cancel"])
    async def test_shared_inference_limit_releases_after_response_error_or_cancellation(
        self, first_outcome, monkeypatch
    ):
        agent, client = _make_agent(max_concurrent_model_requests=1)
        reading = asyncio.Event()
        release = asyncio.Event()
        entered = []

        async def post(**kwargs):
            entered.append(kwargs)
            response = _mock_response(_responses_payload([_MESSAGE_ITEM]))
            if len(entered) == 1:

                async def read():
                    reading.set()
                    await release.wait()
                    if first_outcome == "error":
                        raise RuntimeError("response read failed")
                    return json.dumps(_responses_payload([_MESSAGE_ITEM])).encode()

                response.read = read
            return response

        async def rollout(*, transport, **kwargs):
            await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)
            return _make_rollout_result()

        client.post = post
        monkeypatch.setattr("responses_api_agents.synclaw_agent.app.run_rollout", rollout)
        first_task = asyncio.create_task(agent.responses(MagicMock(cookies={}), Response(), _base_run_request()))
        await asyncio.wait_for(reading.wait(), 2)
        second_task = asyncio.create_task(agent.responses(MagicMock(cookies={}), Response(), _base_run_request()))
        await asyncio.sleep(0)
        assert len(entered) == 1  # The slot covers response-body consumption as well as headers.
        if first_outcome == "cancel":
            first_task.cancel()
        else:
            release.set()
        if first_outcome == "success":
            await first_task
        else:
            with pytest.raises(asyncio.CancelledError if first_outcome == "cancel" else RuntimeError):
                await first_task
        await asyncio.wait_for(second_task, 2)
        assert len(entered) == 2

    async def test_model_calls_are_uncapped_by_default(self, monkeypatch):
        agent, client = _make_agent()
        assert agent.config.max_concurrent_model_requests is None
        release = asyncio.Event()
        entered = []

        async def post(**kwargs):
            entered.append(kwargs)
            await release.wait()
            return _mock_response(_responses_payload([_MESSAGE_ITEM]))

        async def rollout(*, transport, **kwargs):
            await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)
            return _make_rollout_result()

        client.post = post
        monkeypatch.setattr("responses_api_agents.synclaw_agent.app.run_rollout", rollout)
        tasks = [
            asyncio.create_task(agent.responses(MagicMock(cookies={}), Response(), _base_run_request()))
            for _ in range(3)
        ]
        for _ in range(20):
            if len(entered) == 3:
                break
            await asyncio.sleep(0)
        assert len(entered) == 3  # No rollout waits for another's model call.
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)

    async def test_budget_estimate_uses_retained_reasoning_and_usage(self) -> None:
        recorded = []
        transport = _make_transport(
            _capturing_post([_responses_payload([_REASONING_ITEM, _CALL_ITEM])], recorded), max_model_len=16384
        )
        tools = filter_responses_tool_schemas(set())
        before = transport.budget_snapshot((), tools)
        assert before.source == "utf8_estimate"
        assert before.input_tokens > transport.budget_snapshot((), []).input_tokens
        await transport.complete((), tools=tools, tool_choice="auto", max_output_tokens=256)
        transport.append_tool(_prepared_call("call-1"), '{"accepted": false}')
        after = transport.budget_snapshot((), tools)
        assert after.source == "provider_usage_plus_replay_delta"
        assert 150 < after.input_tokens < 200  # input + output usage, plus only the new tool reply.
        assert after.remaining_tokens == 16384 - after.input_tokens
        assert after.max_output_tokens == 256
        assert after.estimated is True

    def test_feedback_rejects_wrong_tool_call_id(self) -> None:
        transport = _make_transport(None)
        transport.append_tool(_prepared_call("call-1"), "{}")
        with pytest.raises(ValueError, match="last unsent input"):
            transport.replace_last_input({"role": "tool", "tool_call_id": "other", "content": "{}"})

    async def test_posts_responses_shaped_payload(self) -> None:
        recorded: list[dict] = []
        transport = _make_transport(_capturing_post([_responses_payload([_MESSAGE_ITEM])], recorded))

        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)

        assert recorded[0]["url_path"] == "/v1/responses"
        payload = recorded[0]["json"]
        assert payload["input"] == [{"type": "message", "role": "user", "content": "Make ethanol."}]
        assert payload["max_output_tokens"] == 256
        assert payload["reasoning"] == {"effort": "high"}
        # Chat-only spellings must not leak onto the Responses request.
        for chat_only in ("messages", "max_tokens", "reasoning_effort"):
            assert chat_only not in payload

    async def test_records_the_upstream_model_name(self) -> None:
        transport = _make_transport(_capturing_post([_responses_payload([_MESSAGE_ITEM])], []))
        assert transport.model is None

        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)

        assert transport.model == "test-model"

    async def test_omits_temperature_when_unset(self) -> None:
        recorded: list[dict] = []
        transport = _make_transport(_capturing_post([_responses_payload([_MESSAGE_ITEM])], recorded))
        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)
        assert "temperature" not in recorded[0]["json"]

    @pytest.mark.parametrize("cap", [1, 16384, 32768, None])
    async def test_forwards_the_output_cap_chosen_by_the_loop(self, cap: int | None) -> None:
        recorded = []
        transport = _make_transport(
            _capturing_post([_responses_payload([_MESSAGE_ITEM])], recorded), max_model_len=65536
        )
        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=cap)
        assert recorded[0]["json"]["max_output_tokens"] == cap
        assert transport.last_request_output_tokens == cap
        assert transport.max_tokens == 256

    async def test_unknown_input_size_keeps_configured_output_limit(self) -> None:
        recorded = []
        transport = _make_transport(
            _capturing_post([_responses_payload([_MESSAGE_ITEM])], recorded),
            seed=[_REASONING_ITEM],  # Opaque reasoning without a provider-usage anchor.
            max_model_len=262144,
        )
        assert transport.budget_snapshot((), []).remaining_tokens is None
        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)
        assert recorded[0]["json"]["max_output_tokens"] == 256

    async def test_sends_temperature_when_set(self) -> None:
        recorded: list[dict] = []
        transport = _make_transport(_capturing_post([_responses_payload([_MESSAGE_ITEM])], recorded), temperature=0.25)
        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)
        assert recorded[0]["json"]["temperature"] == 0.25

    async def test_request_validates_with_unmodified_synclaw_tool_schemas(self) -> None:
        """BioNeMo-SynClaw's Responses schemas omit `strict`, which FunctionToolParam requires."""
        recorded: list[dict] = []
        transport = _make_transport(_capturing_post([_responses_payload([_MESSAGE_ITEM])], recorded))
        schemas = filter_responses_tool_schemas(set())
        assert all("strict" not in schema for schema in schemas)

        await transport.complete((), tools=schemas, tool_choice="auto", max_output_tokens=256)

        payload = recorded[0]["json"]
        assert all(tool["strict"] is None for tool in payload["tools"])
        # The whole request must satisfy the model server's request model.
        assert NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)

    @pytest.mark.parametrize(
        "given,expected",
        [
            (
                {"type": "function", "function": {"name": "submit_pathway"}},
                {"type": "function", "name": "submit_pathway"},
            ),
            ("auto", "auto"),
        ],
    )
    async def test_flattens_chat_tool_choice(self, given, expected) -> None:
        # BioNeMo-SynClaw's loop builds tool_choice for the chat dialect; Responses rejects the nested form.
        recorded: list[dict] = []
        transport = _make_transport(_capturing_post([_responses_payload([_MESSAGE_ITEM])], recorded))
        await transport.complete(
            (), tools=filter_responses_tool_schemas(set()), tool_choice=given, max_output_tokens=256
        )
        assert recorded[0]["json"]["tool_choice"] == expected

    async def test_echoes_output_items_byte_for_byte(self) -> None:
        """Output items are replayed exactly as the model server sent them.

        Gym model servers serialize with `model_dump(mode="json")` and no `exclude_none`, so
        items arrive carrying explicit nulls (`reasoning.content`, `function_call.namespace`,
        `message.phase`). Those are forwarded untouched: verified harmless on both the
        downconverting path (ResponsesConverter drops them via `exclude_none`) and the
        Responses-native passthrough.
        """
        recorded: list[dict] = []
        payload = _responses_payload([_REASONING_ITEM, _CALL_ITEM, _MESSAGE_ITEM])
        assert any(value is None for item in payload["output"] for value in item.values())

        transport = _make_transport(_capturing_post([payload], recorded))
        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)

        assert transport.new_items == payload["output"]

    async def test_echoes_reasoning_verbatim_into_next_request(self) -> None:
        """The point of the Responses port: turn N's reasoning is replayed on turn N+1."""
        recorded: list[dict] = []
        turn_one = _responses_payload([_REASONING_ITEM, _CALL_ITEM])
        transport = _make_transport(_capturing_post([turn_one, _responses_payload([_MESSAGE_ITEM])], recorded))

        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)
        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)

        second_input = recorded[1]["json"]["input"]
        # Compare against the server-emitted form, not the bare constant: the replay is
        # byte-for-byte, so it carries the nulls the model server materialized.
        served_reasoning = next(item for item in turn_one["output"] if item["type"] == "reasoning")
        assert served_reasoning in second_input
        assert served_reasoning["summary"][0]["text"] == "plan it"
        assert [item.get("type") for item in second_input] == ["message", "reasoning", "function_call"]

    async def test_returns_chat_shaped_message_for_the_loop(self) -> None:
        recorded: list[dict] = []
        transport = _make_transport(
            _capturing_post([_responses_payload([_REASONING_ITEM, _CALL_ITEM, _MESSAGE_ITEM])], recorded)
        )

        turn = await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)

        assert turn.message["tool_calls"][0]["id"] == "call-1"
        assert turn.message["tool_calls"][0]["function"]["name"] == "submit_pathway"
        assert turn.message["content"] == "I found it."
        assert turn.message["reasoning_text"] == "plan it"
        # BioNeMo-SynClaw's trace events read usage off the chat message, not the transport.
        usage = usage_from_payload(turn.message)
        assert usage["input_tokens"] == 100
        assert usage["reasoning_tokens"] == 10

    @pytest.mark.parametrize(
        "status,details,expected",
        [
            ("completed", None, None),
            ("incomplete", {"reason": "max_output_tokens"}, "max_output_tokens"),
            ("incomplete", None, "incomplete"),
        ],
    )
    async def test_finish_reason(self, status, details, expected) -> None:
        recorded: list[dict] = []
        transport = _make_transport(
            _capturing_post([_responses_payload([_MESSAGE_ITEM], status=status, incomplete_details=details)], recorded)
        )
        turn = await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)
        assert turn.finish_reason == expected

    async def test_context_length_error_terminates_without_output_limit_recovery(self) -> None:
        async def post(**kwargs):
            return _mock_error_response("This model's maximum context length is 32768 tokens", status=400)

        transport = _make_transport(post)
        with pytest.raises(ControlledTermination) as exc:
            await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)
        assert exc.value.kind == "context_window_exceeded"
        assert transport.new_items == []
        assert transport.usage is None

    async def test_sums_usage_across_turns(self) -> None:
        recorded: list[dict] = []
        transport = _make_transport(
            _capturing_post([_responses_payload([_CALL_ITEM]), _responses_payload([_MESSAGE_ITEM])], recorded)
        )

        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)
        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)

        assert transport.usage.input_tokens == 200
        assert transport.usage.output_tokens == 100
        assert transport.usage.total_tokens == 300
        assert transport.usage.input_tokens_details.cached_tokens == 40
        assert transport.usage.output_tokens_details.reasoning_tokens == 20

    async def test_zero_reasoning_tokens_stay_zero(self) -> None:
        """A measured zero is not an unknown count."""
        recorded: list[dict] = []
        usage = _USAGE | {"output_tokens_details": {"reasoning_tokens": 0}}
        transport = _make_transport(_capturing_post([_responses_payload([_MESSAGE_ITEM], usage=usage)], recorded))

        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)

        assert transport.usage.output_tokens_details.reasoning_tokens == 0

    async def test_unreported_reasoning_tokens_null_the_total(self) -> None:
        """One turn without a count (e.g. vLLM < 0.28) makes the rollout total unknown, not partial."""
        recorded: list[dict] = []
        unreported = _USAGE | {"output_tokens_details": {"reasoning_tokens": None}}
        transport = _make_transport(
            _capturing_post(
                [_responses_payload([_CALL_ITEM]), _responses_payload([_MESSAGE_ITEM], usage=unreported)], recorded
            )
        )

        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)
        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)

        assert transport.usage.output_tokens == 100
        assert transport.usage.output_tokens_details.reasoning_tokens is None

    async def test_unrelated_error_reraises(self) -> None:
        async def post(**kwargs):
            return _mock_error_response("internal boom", status=500)

        transport = _make_transport(post)
        with pytest.raises(ClientResponseError):
            await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)

    def test_new_items_excludes_the_seed(self) -> None:
        seed = [{"type": "message", "role": "system", "content": "sys"}]
        transport = _make_transport(None, seed=seed)
        assert transport.input_history == seed
        assert transport.new_items == []

    def test_append_user_tags_message_type(self) -> None:
        transport = _make_transport(None)
        # The loop hands over an untagged {"role", "content"} dict.
        transport.append_user({"role": "user", "content": "2 turns remaining."})

        assert transport.new_items == [{"role": "user", "content": "2 turns remaining.", "type": "message"}]
        # Without the type discriminator this raises, 500ing the agent.
        assert NeMoGymResponse.model_validate(_responses_payload([]) | {"output": transport.new_items})

    def test_append_tool_pairs_with_an_echoed_call(self) -> None:
        transport = _make_transport(None)
        transport.input_history.append(_CALL_ITEM)

        transport.append_tool(_prepared_call("call-1"), '{"ok": true}')

        assert [item["type"] for item in transport.new_items] == ["function_call", "function_call_output"]
        assert transport.new_items[-1] == {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"ok": true}',
        }

    def test_append_tool_synthesizes_a_call_for_an_orphan_result(self) -> None:
        """Harmony-parsed calls and BioNeMo-SynClaw's tool-call-{n} fallback have no echoed function_call."""
        transport = _make_transport(None)

        transport.append_tool(_prepared_call("harmony-resp_1"), '{"ok": true}')

        assert [item["type"] for item in transport.new_items] == ["function_call", "function_call_output"]
        synthesized, result = transport.new_items
        assert synthesized["call_id"] == result["call_id"] == "harmony-resp_1"
        assert synthesized["name"] == "submit_pathway"
        assert json.loads(synthesized["arguments"]) == {"steps": []}

    def test_append_tool_does_not_resynthesize_for_a_repeated_call_id(self) -> None:
        transport = _make_transport(None)
        transport.append_tool(_prepared_call("harmony-resp_1"), '{"ok": true}')
        transport.append_tool(_prepared_call("harmony-resp_1"), '{"ok": false}')

        assert [item["type"] for item in transport.new_items] == [
            "function_call",
            "function_call_output",
            "function_call_output",
        ]


class TestResponses:
    async def test_tool_call_conversion(self, mock_run_rollout: AsyncMock) -> None:
        agent, _ = _make_agent()
        result = await agent.responses(MagicMock(cookies={}), Response(), _base_run_request())
        fn_calls = [out for out in result.output if isinstance(out, NeMoGymResponseFunctionToolCall)]
        assert len(fn_calls) == 1
        assert fn_calls[0].name == "submit_pathway"
        assert fn_calls[0].call_id == "call-1"

    async def test_tool_output_conversion(self, mock_run_rollout: AsyncMock) -> None:
        agent, _ = _make_agent()
        result = await agent.responses(MagicMock(cookies={}), Response(), _base_run_request())
        outputs = [out for out in result.output if isinstance(out, NeMoGymFunctionCallOutput)]
        assert len(outputs) == 1
        assert outputs[0].call_id == "call-1"

    async def test_assistant_text_conversion(self, mock_run_rollout: AsyncMock) -> None:
        agent, _ = _make_agent()
        result = await agent.responses(MagicMock(cookies={}), Response(), _base_run_request())
        messages = [out for out in result.output if isinstance(out, NeMoGymResponseOutputMessage)]
        assert len(messages) == 1
        assert messages[0].content[0].text == "I found it."

    async def test_reasoning_is_preserved_in_output(self, mock_run_rollout: AsyncMock) -> None:
        """The chat path dropped reasoning entirely; the Responses path must keep it."""
        agent, _ = _make_agent()
        result = await agent.responses(MagicMock(cookies={}), Response(), _base_run_request())
        reasoning = [out for out in result.output if isinstance(out, NeMoGymResponseReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].summary[0].text == "plan it"

    async def test_output_excludes_the_seeded_prompt(self, mock_run_rollout: AsyncMock) -> None:
        agent, _ = _make_agent()
        body = _base_run_request()
        body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"type": "message", "role": "system", "content": "You are a chemist."},
                {"type": "message", "role": "user", "content": "Synthesize ethanol."},
            ]
        )
        result = await agent.responses(MagicMock(cookies={}), Response(), body)
        texts = [getattr(out, "content", None) for out in result.output]
        assert "You are a chemist." not in texts
        assert "Synthesize ethanol." not in texts

    async def test_transport_targets_the_responses_endpoint(self, mock_run_rollout: AsyncMock) -> None:
        agent, _ = _make_agent()
        await agent.responses(
            MagicMock(cookies={}, path_params={}, url=MagicMock(path="")), Response(), _base_run_request()
        )
        transport = mock_run_rollout.call_args.kwargs["transport"]
        assert transport.model_url_path == "/v1/responses"

    async def test_run_rollout_receives_responses_tool_schemas(self, mock_run_rollout: AsyncMock) -> None:
        agent, _ = _make_agent()
        await agent.responses(MagicMock(cookies={}), Response(), _base_run_request())
        schema = mock_run_rollout.call_args.kwargs["tool_schemas"][0]
        assert set(schema) == {"type", "name", "description", "parameters"}
        assert "function" not in schema

    async def test_response_records_the_task_tool_subset(self, mock_run_rollout: AsyncMock) -> None:
        agent, _ = _make_agent()
        excluded = next(name for name in TOOL_NAMES if name != "submit_pathway")
        body = _base_run_request().model_copy(update={"excluded_tools": [excluded]})

        result = await agent.responses(MagicMock(cookies={}), Response(), body)

        sent = mock_run_rollout.call_args.kwargs["tool_schemas"]
        recorded = [tool.model_dump(exclude_none=True) for tool in result.tools]
        assert recorded == [{k: v for k, v in tool.items() if v is not None} for tool in sent]
        assert {tool.name for tool in result.tools} == set(TOOL_NAMES) - {excluded}

    async def test_submission_embedded_in_response(self, mock_run_rollout: AsyncMock) -> None:
        agent, _ = _make_agent()
        result = await agent.responses(MagicMock(cookies={}), Response(), _base_run_request())
        similarity = result.model_extra[SYNCLAW_SUBMISSION_KEY]["similarity_to_target"]
        assert similarity == pytest.approx(0.9)

    async def test_no_submission_embeds_none(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            "responses_api_agents.synclaw_agent.app.run_rollout",
            AsyncMock(return_value=_make_rollout_result(submission=None)),
        )
        agent, _ = _make_agent()
        result = await agent.responses(MagicMock(cookies={}), Response(), _base_run_request())
        assert result.model_extra[SYNCLAW_SUBMISSION_KEY] is None

    async def test_truncated_rollout_reports_max_output_tokens(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            "responses_api_agents.synclaw_agent.app.run_rollout",
            AsyncMock(return_value=_make_rollout_result(partial=True, termination_kind="max_tokens")),
        )
        agent, _ = _make_agent()
        result = await agent.responses(MagicMock(cookies={}), Response(), _base_run_request())
        assert result.status == "incomplete"
        assert result.incomplete_details.reason == "max_output_tokens"

    async def test_run_rollout_called_with_task_fields(self, mock_run_rollout: AsyncMock) -> None:
        agent, _ = _make_agent()
        await agent.responses(MagicMock(cookies={}), Response(), _base_run_request())
        task_arg = mock_run_rollout.call_args.kwargs["task"]
        assert task_arg.task_id == "task-1"
        assert task_arg.target_smiles == "CCO"


class TestRun:
    @pytest.mark.parametrize("endpoint", ["run", "responses"])
    @pytest.mark.parametrize(
        "excluded,detail",
        [
            ("unknown_tool", "Unknown excluded tool"),
            ("submit_pathway", "Cannot exclude required tool"),
        ],
    )
    async def test_invalid_task_exclusions_fail_before_dispatch(
        self, endpoint: str, excluded: str, detail: str
    ) -> None:
        agent, client = _make_agent()
        body = _base_run_request().model_copy(update={"excluded_tools": [excluded]})
        with pytest.raises(HTTPException) as error:
            if endpoint == "run":
                await agent.run(MagicMock(cookies={}), body)
            else:
                await agent.responses(MagicMock(cookies={}), Response(), body)
        assert error.value.status_code == 422
        assert detail in error.value.detail
        client.post.assert_not_called()

    @pytest.mark.parametrize(
        "config_overrides,expected_context",
        [({}, DEFAULT_MAX_MODEL_LEN), ({"max_model_len": 524288}, 524288), ({"max_model_len": None}, None)],
    )
    async def test_lifecycle(
        self,
        mock_run_rollout: AsyncMock,
        config_overrides: dict[str, int | None],
        expected_context: int | None,
    ) -> None:
        agent, server_client = _make_agent(**config_overrides)
        seed_requests = []

        seed_response = {"input": [{"role": "user", "content": "Synthesize ethanol.", "type": "message"}]}

        async def mock_post(*, server_name, url_path, json, cookies=None):
            if url_path == "/seed_session":
                seed_requests.append(SynClawSeedSessionRequest.model_validate_json(orjson.dumps(json)))
                return _mock_response(seed_response)
            if server_name == "synclaw_agent":
                ng_response = await agent.responses(MagicMock(cookies=cookies or {}), Response(), json)
                return _mock_response(ng_response.model_dump(mode="json"))
            if url_path == "/verify":
                return _mock_response(
                    {
                        **json,
                        "reward": 0.9,
                        "evaluation_completed": True,
                        "submitted": True,
                        "accepted": True,
                        "similarity": 0.9,
                        "exact_reconstruction": False,
                    }
                )
            raise AssertionError(f"Unexpected call: {server_name=} {url_path=}")

        server_client.post = mock_post

        result = await agent.run(MagicMock(cookies={}), _base_run_request())

        assert result.reward == pytest.approx(0.9)
        assert result.model_dump(mode="json")["evaluation_completed"] is True
        assert mock_run_rollout.called
        # Guard against the response going vacuously empty: verification must see the trajectory.
        assert result.response.output
        # The seed prompt and subsequent budget feedback must use the same BioNeMo-SynClaw default or explicit override.
        assert len(seed_requests) == 1
        assert seed_requests[0].config.max_model_len == expected_context
        rollout = mock_run_rollout.call_args.kwargs
        assert rollout["transport"].max_model_len == expected_context
        assert rollout["state"].budget_feedback.announced_context_limit == expected_context

    @pytest.mark.parametrize(
        "config_overrides,models,error",
        [
            (
                {},
                [{"id": "other", "max_model_len": 4096}, {"id": "policy", "max_model_len": DEFAULT_MAX_MODEL_LEN}],
                None,
            ),
            ({}, [{"id": "policy", "max_model_len": 131072}], "serves max_model_len=131072"),
            ({"max_model_len": 131072}, [{"id": "policy", "max_model_len": 131072}], None),
            ({"max_model_len": 131072}, [{"id": "policy", "max_model_len": 262144}], None),
            ({}, [{"id": "policy"}], None),
            ({}, _mock_error_response("unauthorized", status=401), "Could not read http://vllm:8000/v1/models"),
        ],
    )
    async def test_served_context_must_fit_resolved_task_limit(
        self, monkeypatch: MonkeyPatch, config_overrides, models, error: str | None
    ) -> None:
        agent, server_client = _make_agent(**config_overrides)
        server_client.global_config_dict = {
            "model": {
                "responses_api_models": {
                    "vllm_model": {"base_url": ["http://vllm:8000/v1"], "api_key": "key", "model": "policy"}
                }
            }
        }
        requested_urls = []

        async def request(**kwargs):
            requested_urls.append(kwargs["url"])
            return models if isinstance(models, MagicMock) else _mock_response({"data": models})

        monkeypatch.setattr("nemo_gym.client.openai_client.request", request)

        for _ in range(2):
            if error is None:
                await agent._require_served_max_model_len(agent.config.max_model_len)
            else:
                with pytest.raises(HTTPException, match=error):
                    await agent._require_served_max_model_len(agent.config.max_model_len)
        # A lookup that failed is not cached, so every call asks again; a successful one is asked once.
        assert requested_urls == ["http://vllm:8000/v1/models"] * (2 if error and "Could not read" in error else 1)

    async def test_unknown_task_context_skips_served_limit_check(self, monkeypatch: MonkeyPatch) -> None:
        agent, _ = _make_agent(max_model_len=None)
        monkeypatch.setattr(
            "nemo_gym.client.openai_client.request", AsyncMock(side_effect=AssertionError("no request"))
        )
        await agent._require_served_max_model_len(agent.config.max_model_len)

    @pytest.mark.parametrize("incoming", [None, False, True])
    async def test_skip_verification(self, mock_run_rollout: AsyncMock, incoming: bool | None) -> None:
        config = SynClawAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="synclaw_agent",
            model_server=ModelServerRef(type="responses_api_models", name="model"),
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            skip_verification=True,
            skip_verification_reward=1.0,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {"observability_enabled": False}
        agent = SynClawAgent(config=config, server_client=server_client)

        seed_response = {"input": [{"role": "user", "content": "Synthesize ethanol.", "type": "message"}]}

        async def mock_post(*, server_name, url_path, json, cookies=None):
            if url_path == "/seed_session":
                return _mock_response(seed_response)
            if server_name == "synclaw_agent":
                ng_response = await agent.responses(MagicMock(cookies=cookies or {}), Response(), json)
                return _mock_response(ng_response.model_dump(mode="json"))
            raise AssertionError(f"Unexpected call: {server_name=} {url_path=}")

        server_client.post = mock_post

        body = _base_run_request()
        if incoming is not None:
            body = SynClawRunRequest.model_validate(body.model_dump() | {"evaluation_completed": incoming})
        result = await agent.run(MagicMock(cookies={}), body)

        assert result.reward == pytest.approx(1.0)
        assert result.verification_skipped is True

        assert result.model_dump(mode="json")["evaluation_completed"] is False


class TestRolloutIntegration:
    """Drives the real BioNeMo-SynClaw rollout loop against a canned model server.

    `run_rollout` is deliberately NOT stubbed here: the transport's history handling, the
    loop's chat/Responses dialect boundary, and the output reconstruction only compose
    correctly when exercised together.
    """

    async def test_concurrent_task_budgets_control_prompt_transport_and_policy(self) -> None:
        agent, client = _make_agent(max_tokens=1024, max_model_len=65536, skip_verification=True)
        shared_policy = agent._policy
        settings = [
            dict(max_turns=1, max_tool_calls=10, max_model_len=32768, max_tokens=512, budget_feedback=True),
            dict(max_turns=3, max_tool_calls=2, max_model_len=65536, max_tokens=2048, budget_feedback=False),
        ]
        seeds, requests = {}, {"task-0": [], "task-1": []}
        both_entered = asyncio.Event()

        async def post(*, server_name, url_path, json, cookies=None):
            if url_path == "/seed_session":
                wire_body = SynClawSeedSessionRequest.model_validate_json(orjson.dumps(json))
                seed = await SynClawResourcesServer.seed_session(MagicMock(), MagicMock(), wire_body)
                seeds[wire_body.question] = seed
                return _mock_response(seed.model_dump(mode="json"))
            if server_name == "synclaw_agent":
                wire_body = SynClawRunRequest.model_validate_json(orjson.dumps(json.model_dump(exclude_unset=True)))
                result = await agent.responses(MagicMock(cookies={}), Response(), wire_body)
                return _mock_response(result.model_dump(mode="json"))
            if server_name == "model":
                task_id = json["input"][1]["content"].splitlines()[0]
                requests[task_id].append(json)
                if all(requests.values()):
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), 2)
                call = {**_CALL_ITEM, "call_id": f"{task_id}-{len(requests[task_id])}"}
                return _mock_response(_responses_payload([call]))
            assert url_path == "/submit_pathway"
            return _mock_response({"accepted": True, "final_smiles": "CC"})

        client.post = post
        bodies = [
            SynClawRunRequest.model_validate(
                _base_run_request().model_dump() | fields | {"task_id": f"task-{i}", "question": f"task-{i}"}
            )
            for i, fields in enumerate(settings)
        ]
        results = await asyncio.gather(*(agent.run(MagicMock(cookies={}), body) for body in bodies))
        assert [len(requests[body.task_id]) for body in bodies] == [1, 2]  # Turn limit, then tool-call limit.
        for body, fields, result in zip(bodies, settings, results, strict=True):
            prompt = seeds[body.task_id].input[0].content
            turn_label = "turn" if fields["max_turns"] == 1 else "turns"
            assert (
                f"{fields['max_tool_calls']} total tool calls or {fields['max_turns']} assistant {turn_label}."
                in prompt
            )
            assert f"context-window budget is {fields['max_model_len']} tokens" in prompt
            assert f"Each response is capped at {fields['max_tokens']} output tokens" in prompt
            assert all(request["max_output_tokens"] == fields["max_tokens"] for request in requests[body.task_id])
            history = str(requests[body.task_id])
            assert ("Remaining:" in history) is fields["budget_feedback"]
            assert result.response.model_extra[SYNCLAW_SUBMISSION_KEY]["accepted"]
        assert agent._policy is shared_policy
        assert shared_policy.max_tokens == 1024 and shared_policy.max_turns == 80

    @pytest.mark.parametrize("identical", [False, True])
    async def test_model_stop_keeps_best_route_and_clips_unstated_budgets(self, identical: bool) -> None:
        agent, client = _make_agent(max_model_len=65536, skip_verification=True, stop_condition="model")
        shared_policy = agent._policy
        fields = {name: "inf" for name in ("max_turns", "max_tool_calls", "max_model_len", "max_tokens")}
        body = SynClawRunRequest.model_validate(
            _base_run_request().model_dump() | fields | {"stop_condition": "exact"}
        )
        requests = []

        async def post(*, server_name, url_path, json, cookies=None):
            if url_path == "/seed_session":
                wire_body = SynClawSeedSessionRequest.model_validate_json(orjson.dumps(json))
                assert wire_body.config.max_turns is None and wire_body.config.max_tool_calls == 0
                assert wire_body.config.max_tokens is None and wire_body.config.stated_max_model_len is None
                assert wire_body.config.max_model_len == 65536
                assert wire_body.config.stop_condition == "model"
                seed = await SynClawResourcesServer.seed_session(MagicMock(), MagicMock(), wire_body)
                assert seed.input[0].content == body.system_prompt
                return _mock_response(seed.model_dump(mode="json"))
            if server_name == "synclaw_agent":
                wire_body = SynClawRunRequest.model_validate_json(orjson.dumps(json.model_dump(exclude_unset=True)))
                result = await agent.responses(MagicMock(cookies={}), Response(), wire_body)
                return _mock_response(result.model_dump(mode="json"))
            if server_name == "model":
                requests.append(json)
                return _mock_response(
                    _responses_payload(
                        [_CALL_ITEM] if len(requests) == 1 else [_MESSAGE_ITEM],
                        usage={**_USAGE, "input_tokens": 50000, "output_tokens": 1000, "total_tokens": 51000},
                    )
                )
            assert url_path == "/submit_pathway"
            return _mock_response({"accepted": True, "final_smiles": "CC", "identical_to_target": identical})

        client.post = post
        result = await agent.run(MagicMock(cookies={}), body)
        assert len(requests) == 2
        assert 32768 < requests[0]["max_output_tokens"] < 65536 - 2048
        assert 0 < requests[1]["max_output_tokens"] < 32768
        history = str(requests[-1]["input"])
        for text in ("Remaining:", "context-window budget", "assistant turns", "guaranteed", "<FINISH>"):
            assert text not in history
        assert result.response.status == "completed"
        assert result.response.model_extra[SYNCLAW_SUBMISSION_KEY]["final_smiles"] == "CC"
        assert agent._policy is shared_policy
        assert shared_policy.stop_condition == "model"

    @pytest.mark.parametrize(
        "config_excluded,row_exclusions",
        [
            ([], [[], sorted(set(TOOL_NAMES) - {"submit_pathway"})]),
            (["mces"], [[], ["similarity"]]),
        ],
    )
    async def test_concurrent_task_exclusions_are_isolated(self, config_excluded, row_exclusions, monkeypatch) -> None:
        agent, client = _make_agent(excluded_tools=config_excluded, skip_verification=True)
        shared_policy = agent._policy
        both_entered = asyncio.Event()
        seed_exclusions, model_tools = {}, {}

        async def rollout(**kwargs):
            task_id = kwargs["task"].task_id
            assert kwargs["config"] is not shared_policy
            assert kwargs["config"].excluded_tools == seed_exclusions[task_id]
            return await run_rollout(**kwargs)

        monkeypatch.setattr("responses_api_agents.synclaw_agent.app.run_rollout", rollout)

        async def post(*, server_name, url_path, json, cookies=None):
            if url_path == "/seed_session":
                seed = SynClawSeedSessionRequest.model_validate_json(orjson.dumps(json))
                seed_exclusions[seed.question] = set(seed.config.excluded_tools)
                return _mock_response({"input": [{"role": "user", "content": seed.question}]})
            if server_name == "synclaw_agent":
                wire_body = SynClawRunRequest.model_validate_json(orjson.dumps(json.model_dump(exclude_unset=True)))
                result = await agent.responses(MagicMock(cookies={}), Response(), wire_body)
                return _mock_response(result.model_dump(mode="json"))
            if server_name == "model":
                task_id = json["input"][0]["content"].splitlines()[0]
                model_tools[task_id] = {tool["name"] for tool in json["tools"]}
                if len(model_tools) == 2:
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), 2)
                return _mock_response(self._turn(self._submit_turn()))
            assert url_path == "/submit_pathway"
            return _mock_response({"accepted": True, "similarity_to_target": 1.0, "final_smiles": "CCO"})

        client.post = post
        bodies = [
            _base_run_request().model_copy(
                update={"task_id": f"task-{i}", "question": f"task-{i}", "excluded_tools": excluded}
            )
            for i, excluded in enumerate(row_exclusions)
        ]
        results = await asyncio.gather(*(agent.run(MagicMock(cookies={}), body) for body in bodies))
        for body, result in zip(bodies, results, strict=True):
            excluded = set(config_excluded) | set(body.model_extra["excluded_tools"])
            assert seed_exclusions[body.task_id] == excluded
            assert model_tools[body.task_id] == set(TOOL_NAMES) - excluded
            assert result.response.model_extra[SYNCLAW_SUBMISSION_KEY]["accepted"]
        assert agent._policy is shared_policy
        assert shared_policy.excluded_tools == frozenset(config_excluded)

    @staticmethod
    def _seeded_request() -> SynClawRunRequest:
        body = _base_run_request()
        body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"type": "message", "role": "user", "content": "Synthesize ethanol."}]
        )
        return body

    @staticmethod
    def _turn(items: list[dict]) -> dict:
        return _responses_payload(items)

    async def _run(self, agent, server_client, turns: list[dict]) -> tuple[Any, list[list[dict]]]:
        payloads = iter(turns)
        model_inputs: list[list[dict]] = []

        async def post(**kwargs):
            if kwargs.get("server_name") == "model":
                model_inputs.append(kwargs["json"]["input"])
                return _mock_response(next(payloads))
            if kwargs.get("url_path") == "/submit_pathway":
                return _mock_response(
                    {
                        "accepted": True,
                        "similarity_to_target": 1.0,
                        "constitutional_similarity_to_target": 1.0,
                        "final_smiles": "CCO",
                    }
                )
            return _mock_response({"results": ["ethene"]})

        server_client.post = post
        result = await agent.responses(MagicMock(cookies={}), Response(), self._seeded_request())
        return result, model_inputs

    @staticmethod
    def _search_turn() -> list[dict]:
        return [
            _REASONING_ITEM,
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call-1",
                "name": "search_building_blocks",
                "arguments": '{"query": "ethanol"}',
            },
        ]

    @staticmethod
    def _submit_turn() -> list[dict]:
        return [
            {"type": "reasoning", "id": "rs_2", "summary": [{"type": "summary_text", "text": "submit now"}]},
            {
                "type": "function_call",
                "id": "fc_2",
                "call_id": "call-2",
                "name": "submit_pathway",
                "arguments": json.dumps(
                    {"steps": [{"product_smiles": "CCO", "reactant_smiles": ["CC=C"], "reaction_name": "hydration"}]}
                ),
            },
        ]

    async def test_reasoning_from_turn_one_is_replayed_on_turn_two(self) -> None:
        agent, server_client = _make_agent()
        _, model_inputs = await self._run(
            agent, server_client, [self._turn(self._search_turn()), self._turn(self._submit_turn())]
        )

        assert len(model_inputs) == 2
        # The seeded prompt is still first, and turn one's reasoning survives verbatim.
        assert model_inputs[1][0]["content"].startswith("Synthesize ethanol.")
        assert "Remaining: 80 turns, 150 tool calls" in model_inputs[1][0]["content"]
        served_reasoning = next(
            item for item in self._turn(self._search_turn())["output"] if item["type"] == "reasoning"
        )
        assert served_reasoning in model_inputs[1]
        assert served_reasoning["summary"][0]["text"] == "plan it"
        assert [item.get("type") for item in model_inputs[1]] == [
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
        ]

    @pytest.mark.parametrize(
        "tool_name",
        [
            "search_building_blocks --help</parameter>",
            "not_a_real_tool",
            "../verify",
            "search_building_blocks?query=ethanol",
            "search_building_blocks#fragment",
        ],
    )
    async def test_unknown_tool_name_returns_feedback_and_rollout_continues(self, tool_name) -> None:
        agent, server_client = _make_agent()
        invalid_turn = self._search_turn()
        invalid_turn[1]["name"] = tool_name
        turns = iter([self._turn(invalid_turn), self._turn(self._submit_turn())])
        model_inputs = []
        resource_paths = []

        async def post(**kwargs):
            if kwargs["server_name"] == "model":
                model_inputs.append(kwargs["json"]["input"])
                return _mock_response(next(turns))
            resource_paths.append(kwargs["url_path"])
            assert kwargs["url_path"] == "/submit_pathway"
            return _mock_response({"accepted": True, "final_smiles": "CCO", "similarity_to_target": 1.0})

        server_client.post = post
        result = await agent.responses(MagicMock(cookies={}), Response(), self._seeded_request())

        assert resource_paths == ["/submit_pathway"]
        assert len(model_inputs) == 2
        reply = json.loads(model_inputs[1][-1]["output"])
        assert reply["error"] == "invalid_tool_call"
        assert reply["tool_name"] == tool_name
        assert reply["received_arguments"] == {"query": "ethanol"}
        assert reply["retryable"] is True
        assert "Remaining: 79 turns, 149 tool calls" in reply["budget"]
        assert result.status == "completed"
        assert result.model_extra[SYNCLAW_SUBMISSION_KEY]["accepted"] is True

    async def test_output_is_a_well_formed_responses_trajectory(self) -> None:
        agent, server_client = _make_agent()
        result, _ = await self._run(
            agent, server_client, [self._turn(self._search_turn()), self._turn(self._submit_turn())]
        )

        assert [item.get("type") for item in result.model_dump(mode="json")["output"]] == [
            "reasoning",
            "function_call",
            "function_call_output",
            "reasoning",
            "function_call",
            "function_call_output",
        ]
        # The seeded prompt must not be duplicated into the output.
        assert all(item.get("role") != "user" for item in result.model_dump(mode="json")["output"])

    async def test_every_tool_result_is_paired_with_a_preceding_call(self) -> None:
        agent, server_client = _make_agent()
        result, _ = await self._run(
            agent, server_client, [self._turn(self._search_turn()), self._turn(self._submit_turn())]
        )

        seen_calls: set[str] = set()
        for item in result.model_dump(mode="json")["output"]:
            if item.get("type") == "function_call":
                seen_calls.add(item["call_id"])
            elif item.get("type") == "function_call_output":
                # An unpaired result downconverts to an orphan chat `tool` message.
                assert item["call_id"] in seen_calls

    async def test_echoed_history_stays_a_valid_model_server_request(self) -> None:
        agent, server_client = _make_agent()
        _, model_inputs = await self._run(
            agent, server_client, [self._turn(self._search_turn()), self._turn(self._submit_turn())]
        )

        for sent_input in model_inputs:
            assert NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": sent_input})

    async def test_submission_survives_the_real_loop(self) -> None:
        agent, server_client = _make_agent()
        result, _ = await self._run(
            agent, server_client, [self._turn(self._search_turn()), self._turn(self._submit_turn())]
        )

        assert result.model_extra[SYNCLAW_SUBMISSION_KEY]["accepted"] is True

    @pytest.mark.parametrize("exhausted", [False, True])
    @pytest.mark.parametrize("context_limit,input_tokens", [(65536, 37938), (262144, 230000)])
    async def test_dynamic_output_limit_in_real_loop(
        self, exhausted: bool, context_limit: int, input_tokens: int
    ) -> None:
        agent, server_client = _make_agent(max_model_len=context_limit)
        search = self._turn(self._submit_turn() if exhausted else self._search_turn())
        margin = agent._policy.context_safety_margin_tokens
        assert margin == context_limit // 32
        if exhausted:
            input_tokens = context_limit - margin // 2
        search["usage"] = {
            **_USAGE,
            "input_tokens": input_tokens,
            "output_tokens": 1000,
            "total_tokens": input_tokens + 1000,
        }
        requests = []
        turns = iter([search, self._turn(self._submit_turn())])

        async def post(**kwargs):
            if kwargs["server_name"] == "model":
                requests.append(kwargs["json"])
                return _mock_response(next(turns))
            if kwargs["url_path"] == "/submit_pathway":
                return _mock_response({"accepted": True, "final_smiles": "CC" if exhausted else "CCO"})
            return _mock_response({"results": ["ethanol"]})

        server_client.post = post
        result = await agent.responses(MagicMock(cookies={}), Response(), self._seeded_request())

        assert result.model_extra["last_request_output_tokens"] == requests[-1]["max_output_tokens"]
        if exhausted:
            assert len(requests) == 1
            assert result.status == "incomplete"
            assert result.model_extra[SYNCLAW_SUBMISSION_KEY]["final_smiles"] == "CC"
        else:
            assert len(requests) == 2
            assert 0 < requests[1]["max_output_tokens"] < context_limit - input_tokens - 1000 - margin
            assert result.status == "completed"
            assert result.model_extra[SYNCLAW_SUBMISSION_KEY]["accepted"] is True

    async def test_usage_is_summed_over_the_real_loop(self) -> None:
        agent, server_client = _make_agent()
        result, _ = await self._run(
            agent, server_client, [self._turn(self._search_turn()), self._turn(self._submit_turn())]
        )

        assert result.usage.input_tokens == 200
        assert result.usage.output_tokens == 100
        assert result.usage.total_tokens == 300
        assert result.usage.input_tokens_details.cached_tokens == 40
        assert result.usage.output_tokens_details.reasoning_tokens == 20

    @pytest.mark.parametrize("enabled", [True, False])
    async def test_budget_feedback_preserves_tool_turns(self, enabled) -> None:
        agent, server_client = _make_agent(budget_feedback=enabled, max_model_len=16384)
        result, model_inputs = await self._run(
            agent, server_client, [self._turn(self._search_turn()), self._turn(self._submit_turn())]
        )
        last = model_inputs[1][-1]
        assert last["type"] == "function_call_output"
        output = json.loads(last["output"])
        assert output["results"] == ["ethene"]
        if enabled:
            assert "Remaining: 79 turns, 149 tool calls" in output["budget"]
            assert " context tokens." in output["budget"]
            assert result.model_extra["context_budget_source"] == "provider_usage_plus_replay_delta"
        else:
            assert "budget" not in output
            assert result.model_extra["context_budget_source"] == "provider_usage_plus_replay_delta"
            assert model_inputs[0][0]["content"] == "Synthesize ethanol."

    async def test_output_limit_recovers_once_without_executing_truncated_calls(self) -> None:
        agent, server_client = _make_agent()
        truncated = _responses_payload(
            self._search_turn(),
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
            usage={**_USAGE, "input_tokens": 200000, "output_tokens": 1000, "total_tokens": 201000},
        )
        result, model_inputs = await self._run(agent, server_client, [truncated, self._turn(self._submit_turn())])
        reply = json.loads(model_inputs[1][-1]["output"])
        assert reply["executed"] is False
        assert reply["error"] == "output_limit_truncation"
        assert "only output-limit recovery attempt" in reply["notice"]
        assert "149 tool calls" not in reply["budget"]  # Truncated calls do not consume tool budget.
        assert result.model_extra[SYNCLAW_SUBMISSION_KEY]["accepted"] is True
        assert 0 < result.model_extra["last_request_output_tokens"] < 32768

        result, model_inputs = await self._run(agent, server_client, [truncated, truncated])
        assert len(model_inputs) == 2
        assert result.status == "incomplete"
        # The second cap was clipped to remaining context: BioNeMo-SynClaw classifies this as context exhaustion.
        assert result.incomplete_details is None

    async def test_output_limit_does_not_recover_when_feedback_is_disabled(self) -> None:
        agent, server_client = _make_agent(budget_feedback=False)
        truncated = _responses_payload(
            self._search_turn(), status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
        result, model_inputs = await self._run(agent, server_client, [truncated])
        assert len(model_inputs) == 1
        assert result.status == "incomplete"
        assert result.incomplete_details.reason == "max_output_tokens"
        assert result.model_extra[SYNCLAW_SUBMISSION_KEY] is None
        assert not any(item.type == "function_call_output" for item in result.output)

    async def test_cookies_from_the_model_server_are_propagated(self) -> None:
        """Stateful resources servers depend on the session cookie surviving the rollout."""
        agent, server_client = _make_agent()
        payloads = iter([self._turn(self._search_turn()), self._turn(self._submit_turn())])

        async def post(**kwargs):
            if kwargs.get("server_name") == "model":
                response = _mock_response(next(payloads))
                response.cookies = {"model_session": "abc"}
                return response
            response = _mock_response(
                {
                    "accepted": True,
                    "similarity_to_target": 1.0,
                    "constitutional_similarity_to_target": 1.0,
                    "final_smiles": "CCO",
                }
            )
            response.cookies = {"tool_session": "xyz"}
            return response

        server_client.post = post
        fastapi_response = Response()
        await agent.responses(MagicMock(cookies={}), fastapi_response, self._seeded_request())

        set_cookie = fastapi_response.raw_headers
        rendered = b" ".join(value for key, value in set_cookie if key == b"set-cookie")
        assert b"model_session=abc" in rendered
        assert b"tool_session=xyz" in rendered


class TestAggregateMetrics:
    async def test_proxies_to_the_resources_server(self) -> None:
        agent, server_client = _make_agent()
        seen: list[str] = []

        async def post(*, server_name, url_path, json, **kwargs):
            seen.append(url_path)
            return _mock_response({"key_metrics": {"reward": 0.5}})

        server_client.post = post
        result = await agent.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))

        assert seen == ["/aggregate_metrics"]
        assert result.key_metrics == {"reward": 0.5}


class TestReasoningEffort:
    async def test_effort_is_sent_as_a_reasoning_object(self) -> None:
        """Responses spells this `reasoning: {effort}`, not the chat `reasoning_effort`.

        No `summary` is requested: vLLM's reasoning parser returns reasoning items without
        one, and a non-null summary has no Chat Completions representation, so model servers
        that downconvert reject the request outright.
        """
        recorded: list[dict] = []
        transport = _make_transport(_capturing_post([_responses_payload([_MESSAGE_ITEM])], recorded))
        await transport.complete((), tools=[], tool_choice="auto", max_output_tokens=256)

        payload = recorded[0]["json"]
        assert payload["reasoning"] == {"effort": "high"}
        assert "reasoning_effort" not in payload
