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
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from responses_api_agents.gymnasium_agent.app import GymnasiumAgent, GymnasiumAgentConfig, GymnasiumAgentRunRequest


def _make_agent(max_steps=10, observability=True, *, text_only_history=False, **config_overrides):
    config = GymnasiumAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="test_gymnasium_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="my_env"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_steps=max_steps,
        text_only_history=text_only_history,
        **config_overrides,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"observability_enabled": observability}
    return GymnasiumAgent(config=config, server_client=server_client)


def _model_response(text: str, input_toks=1, output_toks=1, cached_toks=0, reasoning_toks=0) -> dict:
    return {
        "id": "r",
        "created_at": 0.0,
        "model": "m",
        "object": "response",
        "output": [
            {
                "id": "msg",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": input_toks,
            "input_tokens_details": {"cached_tokens": cached_toks},
            "output_tokens": output_toks,
            "output_tokens_details": {"reasoning_tokens": reasoning_toks},
            "total_tokens": input_toks + output_toks,
        },
    }


def _model_response_with_reasoning(text: str) -> dict:
    response = _model_response(text)
    response["output"].insert(
        0,
        {
            "id": "reasoning",
            "summary": [{"text": "provider-visible reasoning summary", "type": "summary_text"}],
            "type": "reasoning",
        },
    )
    return response


def _model_response_with_reasoning_and_tool_call(text: str) -> dict:
    response = _model_response_with_reasoning(text)
    response["output"].append(
        {
            "arguments": '{"query":"example"}',
            "call_id": "call-1",
            "id": "function-call-1",
            "name": "lookup",
            "status": "completed",
            "type": "function_call",
        }
    )
    return response


class _FakeHttpResp:
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
        class _Body:
            async def read(inner):
                return json.dumps(self._payload).encode()

        return _Body()

    def raise_for_status(self):
        return None


class _FailedHttpResp(_FakeHttpResp):
    def __init__(self, payload: dict, *, message: str):
        super().__init__(payload)
        self.ok = False
        self.status = 500
        self._message = message

    def raise_for_status(self):
        raise RuntimeError(self._message)


def _wire_mock_client(agent, responses_per_url):
    """Wire agent.server_client.post to return payloads keyed by url_path."""
    call_log = []

    async def _post(server_name, url_path, json=None, cookies=None, **kw):
        call_log.append((server_name, url_path, json))
        payload = responses_per_url[url_path].pop(0)
        return _FakeHttpResp(payload)

    agent.server_client.post = AsyncMock(side_effect=_post)
    return call_log


class TestRoutes:
    def test_routes_registered(self):
        app = _make_agent().setup_webserver()
        routes = {r.path for r in app.routes}
        assert {"/run", "/v1/responses", "/aggregate_metrics"}.issubset(routes)


class TestConfig:
    def test_max_steps_validator_rejects_zero(self):
        with pytest.raises(Exception):
            GymnasiumAgentConfig(
                host="",
                port=0,
                entrypoint="",
                name="x",
                resources_server=ResourcesServerRef(type="resources_servers", name="e"),
                model_server=ModelServerRef(type="responses_api_models", name="m"),
                max_steps=0,
            )

    def test_default_max_steps(self):
        assert _make_agent().config.max_steps == 10
        assert _make_agent().config.text_only_history is False
        assert _make_agent().config.model_response_max_concurrency is None

    def test_model_server_transport_override_is_opt_in(self):
        assert _make_agent()._model_server_request_kwargs() == {}
        agent = _make_agent(model_server_transport_max_attempts=1)
        assert agent._model_server_request_kwargs() == {"_max_num_tries": 1}


class TestRun:
    @pytest.mark.asyncio
    async def test_target_model_semaphore_does_not_cover_environment_calls(self):
        agent = _make_agent(model_response_max_concurrency=1, model_server_transport_max_attempts=2)
        active_model_calls = 0
        peak_model_calls = 0
        reset_calls = 0
        first_model_entered = asyncio.Event()
        second_reset_completed = asyncio.Event()
        release_model = asyncio.Event()

        async def post(server_name, url_path, json=None, cookies=None, **kwargs):
            nonlocal active_model_calls, peak_model_calls, reset_calls
            if url_path == "/v1/responses":
                assert kwargs == {"_max_num_tries": 2}
                active_model_calls += 1
                peak_model_calls = max(peak_model_calls, active_model_calls)
                first_model_entered.set()
                await release_model.wait()
                active_model_calls -= 1
                return _FakeHttpResp(_model_response("answer"))
            assert kwargs == {}
            if url_path == "/reset":
                reset_calls += 1
                if reset_calls == 2:
                    second_reset_completed.set()
                return _FakeHttpResp({"observation": "go", "info": {}})
            assert url_path == "/step"
            return _FakeHttpResp({"reward": 1.0, "terminated": True, "truncated": False, "info": {}})

        agent.server_client.post = AsyncMock(side_effect=post)
        request = MagicMock(cookies={})
        body = GymnasiumAgentRunRequest(responses_create_params={"input": "play"})

        with anyio.fail_after(2):
            async with asyncio.TaskGroup() as group:
                first = group.create_task(agent.run(request, body))
                await first_model_entered.wait()
                second = group.create_task(agent.run(request, body))
                # The second rollout can reset while the first holds the model slot.
                await second_reset_completed.wait()
                await asyncio.sleep(0)
                assert peak_model_calls == 1
                release_model.set()

        assert first.result().reward == second.result().reward == 1.0
        assert peak_model_calls == 1

    @pytest.mark.asyncio
    async def test_terminates_on_first_step(self):
        agent = _make_agent()
        model_path = "/ng-rollout/2-0/v1/responses"
        payloads = {
            "/reset": [{"observation": "go", "info": {}}],
            model_path: [_model_response("move A")],
            "/step": [{"observation": None, "reward": 1.0, "terminated": True, "truncated": False, "info": {}}],
        }
        seen = []

        async def _post(server_name, url_path, json=None, cookies=None, headers=None, **kw):
            seen.append((url_path, headers))
            return _FakeHttpResp(payloads[url_path].pop(0))

        agent.server_client.post = AsyncMock(side_effect=_post)
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(
            responses_create_params={"input": [{"role": "user", "content": "play"}]},
            **{TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0},
        )
        result = await agent.run(req, body)
        assert result.terminated is True
        assert result.reward == 1.0

        urls = [url for url, _headers in seen]
        assert urls.count("/reset") == 1
        assert urls.count("/step") == 1
        assert urls.count("/close") == 0
        model_calls = [(u, h) for (u, h) in seen if u == model_path]
        assert model_calls == [(model_path, None)]

    @pytest.mark.asyncio
    async def test_successful_rollout_survives_close_http_failure(self):
        agent = _make_agent()
        payloads = {
            "/reset": [{"observation": "go", "info": {"supports_explicit_close": True}}],
            "/v1/responses": [_model_response("move A")],
            "/step": [
                {
                    "observation": None,
                    "reward": 1.25,
                    "terminated": True,
                    "truncated": False,
                    "info": {"step_idx": 1},
                }
            ],
        }

        async def _post(server_name, url_path, json=None, cookies=None, **kw):
            if url_path == "/close":
                return _FailedHttpResp(
                    {"error": "close unavailable"},
                    message="close failure",
                )
            return _FakeHttpResp(payloads[url_path].pop(0))

        agent.server_client.post = AsyncMock(side_effect=_post)
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "play"}]})

        result = await agent.run(req, body)

        assert result.reward == pytest.approx(1.25)
        assert result.terminated is True
        assert result.info["step_idx"] == 1
        assert result.info["cleanup_warning"]["operation"] == "close"
        assert result.info["cleanup_warning"]["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_no_rollout_prefix_when_observability_disabled(self):
        agent = _make_agent(observability=False)
        call_log = _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": "go", "info": {}}],
                "/v1/responses": [_model_response("move A")],
                "/step": [{"observation": None, "reward": 1.0, "terminated": True, "truncated": False, "info": {}}],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(
            responses_create_params={"input": [{"role": "user", "content": "play"}]},
            **{TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0},
        )
        result = await agent.run(req, body)
        assert result.terminated is True
        # Task indices are present, but capture is off -> the model call stays unprefixed.
        assert [u for _s, u, _j in call_log if u.startswith("/v1/")] == ["/v1/responses"]

    @pytest.mark.asyncio
    async def test_multi_step_preserves_output_items_in_history(self):
        agent = _make_agent(max_steps=3)
        call_log = _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": "start", "info": {"supports_step_idempotency": True}}],
                "/v1/responses": [
                    _model_response_with_reasoning_and_tool_call("turn-1"),
                    _model_response("turn-2", output_toks=20),
                ],
                "/step": [
                    {
                        "observation": "obs-1",
                        "reward": 0.5,
                        "terminated": False,
                        "truncated": False,
                        "info": {"tool_outputs": [{"call_id": "call-1", "output": "lookup result"}]},
                    },
                    {"observation": None, "reward": 0.5, "terminated": True, "truncated": False, "info": {}},
                ],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "play"}]})
        result = await agent.run(req, body)
        assert result.reward == 1.0
        assert result.terminated is True
        step_bodies = [payload for _server, url, payload in call_log if url == "/step"]
        request_ids = [payload["_ng_step_request_id"] for payload in step_bodies]
        assert len(request_ids) == len(set(request_ids)) == 2
        assert all(isinstance(request_id, str) and request_id for request_id in request_ids)
        # Inspect turn-2 model call body: its input must contain the full turn-1 output item,
        # not a flattened string, and the obs-1 appended as user message.
        turn2_body = [body for (s, u, body) in call_log if u == "/v1/responses"][1]
        turn2_input = turn2_body.input
        # turn-1 full output item preserved (with structured content list)
        assistant_items = [m for m in turn2_input if getattr(m, "role", None) == "assistant"]
        assert any(
            isinstance(getattr(m, "content", None), list)
            and any(
                getattr(c, "type", None) == "output_text" and getattr(c, "text", "") == "turn-1" for c in m.content
            )
            for m in assistant_items
        ), f"turn-1 output not preserved in structured form: {assistant_items}"
        function_calls = [item for item in turn2_input if getattr(item, "type", None) == "function_call"]
        function_outputs = [item for item in turn2_input if getattr(item, "type", None) == "function_call_output"]
        assert [item.call_id for item in function_calls] == ["call-1"]
        assert [item.call_id for item in function_outputs] == ["call-1"]
        # obs-1 appended as a user message after turn-1
        assert any(getattr(m, "role", None) == "user" and getattr(m, "content", "") == "obs-1" for m in turn2_input)

    @pytest.mark.asyncio
    async def test_text_only_history_replays_only_visible_assistant_text(self):
        agent = _make_agent(max_steps=3, text_only_history=True)
        call_log = _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": "start", "info": {"supports_step_idempotency": True}}],
                "/v1/responses": [
                    _model_response_with_reasoning_and_tool_call("turn-1"),
                    _model_response("turn-2"),
                ],
                "/step": [
                    {
                        "observation": "obs-1",
                        "reward": 0.5,
                        "terminated": False,
                        "truncated": False,
                        "info": {"tool_outputs": [{"call_id": "call-1", "output": "lookup result"}]},
                    },
                    {"observation": None, "reward": 0.5, "terminated": True, "truncated": False, "info": {}},
                ],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "play"}]})

        result = await agent.run(req, body)

        turn2_body = [request_body for (_server, path, request_body) in call_log if path == "/v1/responses"][1]
        turn2_input = turn2_body.input
        assert not any(getattr(item, "type", None) == "reasoning" for item in turn2_input)
        assert not any(getattr(item, "type", None) == "function_call" for item in turn2_input)
        assert not any(getattr(item, "type", None) == "function_call_output" for item in turn2_input)
        assert any(
            getattr(item, "role", None) == "assistant" and getattr(item, "content", None) == "turn-1"
            for item in turn2_input
        )
        assert any(
            getattr(item, "role", None) == "user" and getattr(item, "content", None) == "obs-1" for item in turn2_input
        )

        step_payloads = [request_body for (_server, path, request_body) in call_log if path == "/step"]
        assert all(payload["agent_history_mode"] == "text_only" for payload in step_payloads)
        request_ids = [payload["_ng_step_request_id"] for payload in step_payloads]
        assert len(request_ids) == len(set(request_ids)) == 2
        assert all(isinstance(request_id, str) and request_id for request_id in request_ids)
        assert any(item.type == "reasoning" for item in result.response.output)
        artifact_function_calls = [item for item in result.response.output if item.type == "function_call"]
        artifact_function_outputs = [item for item in result.response.output if item.type == "function_call_output"]
        assert [item.call_id for item in artifact_function_calls] == ["call-1"]
        assert [item.call_id for item in artifact_function_outputs] == ["call-1"]
        assert len([item for item in result.response.output if item.type == "message"]) == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blank_turn", [0, 1, 2], ids=["first", "middle", "final"])
    @pytest.mark.parametrize("output_kind", ["reasoning_only", "no_items", "empty_message"])
    async def test_text_only_history_preserves_blank_assistant_turn(self, blank_turn, output_kind):
        agent = _make_agent(max_steps=3, text_only_history=True)
        answers = ["answer-1", "answer-2", "answer-3"]
        answers[blank_turn] = ""
        responses = [_model_response(answer) for answer in answers]
        if output_kind == "reasoning_only":
            responses[blank_turn] = _model_response_with_reasoning("")
            responses[blank_turn]["output"].pop()
            responses[blank_turn]["status"] = "incomplete"
            responses[blank_turn]["incomplete_details"] = {"reason": "max_output_tokens"}
        elif output_kind == "no_items":
            responses[blank_turn]["output"] = []
        original_responses = [NeMoGymResponse.model_validate(response).model_dump() for response in responses]
        call_log = _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": None}],
                "/v1/responses": responses,
                "/step": [
                    {"observation": "followup-1", "reward": 0.0, "terminated": False, "truncated": False},
                    {"observation": "followup-2", "reward": 0.0, "terminated": False, "truncated": False},
                    {"observation": None, "reward": 0.0, "terminated": True, "truncated": False},
                ],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "play"}]})

        result = await agent.run(req, body)

        # The serialized trajectory must retain three distinct assistant turns,
        # including a blank final turn without a following user-message boundary.
        response = NeMoGymResponse.model_validate_json(result.response.model_dump_json())
        messages = [item for item in response.output if item.type == "message"]
        assert [item.role for item in messages] == ["assistant", "user", "assistant", "user", "assistant"]
        assert [
            item.content if isinstance(item.content, str) else "".join(part.text for part in item.content)
            for item in messages
        ] == [answers[0], "followup-1", answers[1], "followup-2", answers[2]]
        original_output = [item for payload in original_responses for item in payload["output"]]
        preserved_output = [item.model_dump() for item in response.output if getattr(item, "id", None) is not None]
        assert preserved_output == original_output

        # Resource verification sees exactly the provider response, including
        # its original incompleteness, rather than the added transcript marker.
        step_responses = [payload["response"] for _server, path, payload in call_log if path == "/step"]
        assert step_responses == original_responses
        assert response.status == original_responses[-1]["status"]
        assert (
            response.incomplete_details.model_dump() if response.incomplete_details is not None else None
        ) == original_responses[-1]["incomplete_details"]
        model_requests = [payload for _server, path, payload in call_log if path == "/v1/responses"]
        for index, request_body in enumerate(model_requests):
            assert [item.content for item in request_body.input if item.role == "assistant"] == answers[:index]
        assert result.terminated is True
        assert result.truncated is False

    @pytest.mark.asyncio
    async def test_original_history_preserves_reasoning_only_output_without_added_message(self):
        agent = _make_agent(max_steps=1)
        response = _model_response_with_reasoning("")
        response["output"].pop()
        expected_output = NeMoGymResponse.model_validate(response).output
        _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": None}],
                "/v1/responses": [response],
                "/step": [{"observation": None, "reward": 0.0, "terminated": True, "truncated": False}],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "play"}]})

        result = await agent.run(req, body)

        assert result.response.output == expected_output

    @pytest.mark.asyncio
    async def test_max_steps_sets_truncated(self):
        agent = _make_agent(max_steps=2)
        call_log = _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": None, "info": {"supports_explicit_close": True}}],
                "/v1/responses": [_model_response("a"), _model_response("b")],
                "/step": [
                    {"observation": "obs-1", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
                    {"observation": "obs-2", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
                ],
                "/close": [{"ok": True, "already_closed": False, "summary": {}}],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "x"}]})
        result = await agent.run(req, body)
        assert result.truncated is True
        assert result.terminated is False
        assert [url for _server, url, _json in call_log].count("/close") == 1

    @pytest.mark.asyncio
    async def test_model_failure_after_reset_still_closes_environment(self):
        agent = _make_agent(max_steps=2)
        call_log = []

        async def _post(server_name, url_path, json=None, cookies=None, **kw):
            call_log.append((server_name, url_path, json, cookies))
            if url_path == "/reset":
                response = _FakeHttpResp({"observation": "start", "info": {"supports_explicit_close": True}})
                response.cookies = {"session": "episode-cookie"}
                return response
            if url_path == "/close":
                return _FakeHttpResp({"ok": True, "already_closed": False, "summary": {}})
            raise RuntimeError("model server unavailable")

        agent.server_client.post = AsyncMock(side_effect=_post)
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "x"}]})

        with pytest.raises(RuntimeError, match="model server unavailable"):
            await agent.run(req, body)

        close_calls = [entry for entry in call_log if entry[1] == "/close"]
        assert len(close_calls) == 1
        assert close_calls[0][3] == {"session": "episode-cookie"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stalled_close", [False, True])
    async def test_disconnect_cancellation_shields_and_bounds_environment_cleanup(self, stalled_close, caplog):
        agent = _make_agent(environment_cleanup_timeout_seconds=0.02 if stalled_close else 1.0)
        model_entered = anyio.Event()
        close_entered = anyio.Event()
        close_finished = anyio.Event()
        close_exited = anyio.Event()
        original_errors = []
        propagated_errors = []
        close_cookies = []

        async def post(server_name, url_path, json=None, cookies=None, **kwargs):
            if url_path == "/reset":
                response = _FakeHttpResp({"observation": "go", "info": {"supports_explicit_close": True}})
                response.cookies = {"session": "episode-cookie"}
                return response
            if url_path == "/v1/responses":
                model_entered.set()
                try:
                    await anyio.sleep_forever()
                except anyio.get_cancelled_exc_class() as exc:
                    original_errors.append(exc)
                    raise
            if url_path == "/close":
                close_cookies.append(cookies)
                close_entered.set()
                try:
                    if stalled_close:
                        await anyio.sleep_forever()
                    else:
                        # Real checkpoints expose AnyIO's repeated cancellation;
                        # raising CancelledError from a mock would not do so.
                        await anyio.sleep(0)
                        await anyio.sleep(0)
                        close_finished.set()
                        return _FakeHttpResp({"ok": True})
                finally:
                    close_exited.set()
            raise AssertionError(f"unexpected request: {url_path}")

        agent.server_client.post = AsyncMock(side_effect=post)
        request = MagicMock(cookies={"auth": "incoming"})
        body = GymnasiumAgentRunRequest(responses_create_params={"input": "play"})

        async def run():
            try:
                await agent.run(request, body)
            except anyio.get_cancelled_exc_class() as exc:
                propagated_errors.append(exc)
                raise

        with anyio.fail_after(2):
            async with anyio.create_task_group() as group:
                group.start_soon(run)
                await model_entered.wait()
                group.cancel_scope.cancel()

        assert close_entered.is_set()
        assert close_exited.is_set()
        assert close_finished.is_set() is (not stalled_close)
        assert close_cookies == [{"auth": "incoming", "session": "episode-cookie"}]
        assert len(original_errors) == len(propagated_errors) == 1
        assert propagated_errors[0] is original_errors[0]
        if stalled_close:
            assert "Failed to close Gymnasium environment after rollout error" in caplog.text
            assert "TimeoutError" in caplog.text

    @pytest.mark.asyncio
    async def test_environment_cleanup_timeout_preserves_original_model_failure(self, caplog):
        agent = _make_agent(environment_cleanup_timeout_seconds=0.02)
        model_error = RuntimeError("original model failure")
        close_exited = anyio.Event()

        async def post(server_name, url_path, **kwargs):
            if url_path == "/reset":
                return _FakeHttpResp({"observation": "go", "info": {"supports_explicit_close": True}})
            if url_path == "/close":
                try:
                    await anyio.sleep_forever()
                finally:
                    close_exited.set()
            raise model_error

        agent.server_client.post = AsyncMock(side_effect=post)
        request = MagicMock(cookies={})
        body = GymnasiumAgentRunRequest(responses_create_params={"input": "play"})

        with anyio.fail_after(2), pytest.raises(RuntimeError) as raised:
            await agent.run(request, body)

        assert raised.value is model_error
        assert close_exited.is_set()
        assert "TimeoutError" in caplog.text

    @pytest.mark.asyncio
    async def test_malformed_reset_response_still_closes_environment(self):
        agent = _make_agent(max_steps=2)
        call_log = []

        async def _post(server_name, url_path, json=None, cookies=None, **kw):
            call_log.append((server_name, url_path, json, cookies))
            if url_path == "/reset":
                response = _FakeHttpResp(
                    {
                        "observation": {"not": "text"},
                        "info": {"supports_explicit_close": True},
                    }
                )
                response.cookies = {"session": "episode-cookie"}
                return response
            if url_path == "/close":
                return _FakeHttpResp({"ok": True, "already_closed": False, "summary": {}})
            raise AssertionError(f"unexpected request: {url_path}")

        agent.server_client.post = AsyncMock(side_effect=_post)
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "x"}]})

        with pytest.raises(Exception):
            await agent.run(req, body)

        close_calls = [entry for entry in call_log if entry[1] == "/close"]
        assert len(close_calls) == 1
        assert close_calls[0][3] == {"session": "episode-cookie"}

    @pytest.mark.asyncio
    async def test_inbound_request_cookies_are_preserved_for_environment(self):
        agent = _make_agent(max_steps=2)
        call_log = []

        async def _post(server_name, url_path, json=None, cookies=None, **kw):
            call_log.append((server_name, url_path, json, cookies))
            if url_path == "/reset":
                return _FakeHttpResp({"observation": "start", "info": {"supports_explicit_close": True}})
            if url_path == "/close":
                return _FakeHttpResp({"ok": True, "already_closed": False, "summary": {}})
            raise RuntimeError("model server unavailable")

        agent.server_client.post = AsyncMock(side_effect=_post)
        req = MagicMock()
        req.cookies = {"session": "existing-cookie"}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "x"}]})

        with pytest.raises(RuntimeError, match="model server unavailable"):
            await agent.run(req, body)

        reset_calls = [entry for entry in call_log if entry[1] == "/reset"]
        close_calls = [entry for entry in call_log if entry[1] == "/close"]
        assert len(reset_calls) == 1
        assert reset_calls[0][3] == {"session": "existing-cookie"}
        assert len(close_calls) == 1
        assert close_calls[0][3] == {"session": "existing-cookie"}

    @pytest.mark.asyncio
    async def test_usage_accumulates_across_turns(self):
        agent = _make_agent(max_steps=3)
        _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": None, "info": {}}],
                "/v1/responses": [
                    _model_response("a", input_toks=5, output_toks=7, cached_toks=2, reasoning_toks=3),
                    _model_response("b", input_toks=11, output_toks=13, cached_toks=5, reasoning_toks=7),
                ],
                "/step": [
                    {"observation": "o", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
                    {"observation": None, "reward": 0.0, "terminated": True, "truncated": False, "info": {}},
                ],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "x"}]})
        result = await agent.run(req, body)
        # usage summed across both turns
        assert result.response.usage.input_tokens == 16
        assert result.response.usage.output_tokens == 20
        assert result.response.usage.total_tokens == 36
        assert result.response.usage.input_tokens_details.cached_tokens == 7
        assert result.response.usage.output_tokens_details.reasoning_tokens == 10
