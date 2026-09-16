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
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from starlette.middleware.base import BaseHTTPMiddleware

from nemo_gym.base_responses_api_model import (
    CaptureStore,
    _CaptureMiddleware,
    aggregate_model_call_metrics,
    read_model_call_records,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model import app as openai_model_module
from responses_api_models.openai_model.app import (
    NeMoGymAsyncOpenAI,
    SimpleModelServer,
    SimpleModelServerConfig,
    UpstreamRetryPolicy,
)


PROVIDER_RETRY_POLICY = {
    "max_attempts": 11,
    "pre_request_jitter_seconds": [0.0, 0.2],
    "backoff_initial_seconds": 1.0,
    "backoff_multiplier": 2.0,
    "backoff_jitter_fraction": 0.5,
    "terminal_http_status_codes": [400],
}


def _response_data() -> dict:
    return {
        "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
        "created_at": 1753983920.0,
        "model": "dummy_model",
        "object": "response",
        "output": [
            {
                "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                "content": [
                    {
                        "annotations": [],
                        "text": "Hello! How can I help you today?",
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


class TestApp:
    def _setup_server(self, max_concurrent_requests=None, **config_overrides):
        config = SimpleModelServerConfig(
            host="0.0.0.0",
            port=8081,
            openai_base_url="https://api.openai.com/v1",
            openai_api_key="dummy_key",  # pragma: allowlist secret
            openai_model="dummy_model",
            entrypoint="",
            name="test_model_server",
            max_concurrent_requests=max_concurrent_requests,
            **config_overrides,
        )
        return SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))

    async def test_sanity(self) -> None:
        self._setup_server()

    async def test_chat_completions(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        server = self._setup_server()
        server.server_client.global_config_dict = {
            "observability_enabled": True,
            "model_call_capture_dir": str(tmp_path),
        }
        app = server.setup_webserver()
        client = TestClient(app)

        mock_chat_data = {
            "id": "chatcmpl-BzRdCFjIEIp59xXLBNYjdPPrcpDaa",  # pragma: allowlist secret
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Hello! How can I help you today?",
                        "role": "assistant",
                    },
                }
            ],
            "created": 1753983922,
            "model": "dummy_model",
            "object": "chat.completion",
        }

        called_args_chat = {}

        async def mock_create_chat(**kwargs):
            nonlocal called_args_chat
            called_args_chat = kwargs
            return mock_chat_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        chat_no_model = client.post(
            "/ng-rollout/chat-test/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert chat_no_model.status_code == 200
        assert called_args_chat.get("model") == "dummy_model"

        chat_with_model = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "override_model",
            },
        )
        assert chat_with_model.status_code == 200
        assert called_args_chat.get("model") == "dummy_model"

        server._client.create_chat_completion.assert_any_await(
            messages=[{"role": "user", "content": "hi"}],
            model="dummy_model",
        )

        chat_244_fields = {
            "moderation": {"model": "omni-moderation-latest"},
            "prompt_cache_key": "cache-key",
            "prompt_cache_retention": "24h",
            "safety_identifier": "safe-user",
            "verbosity": "high",
        }
        forwarded = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], **chat_244_fields},
        )
        assert forwarded.status_code == 200
        assert {field: called_args_chat[field] for field in chat_244_fields} == chat_244_fields

        calls = read_model_call_records(CaptureStore(tmp_path), "chat-test")
        assert len(calls) == 1 and calls[0].dialect == "chat"

    async def test_responses(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        server = self._setup_server()
        server.server_client.global_config_dict = {
            "observability_enabled": True,
            "model_call_capture_dir": str(tmp_path),
        }
        app = server.setup_webserver()
        client = TestClient(app)

        called_args_response = {}

        async def mock_create_response(**kwargs):
            nonlocal called_args_response
            called_args_response = kwargs
            return {**_response_data(), "reasoning": {"effort": "none"}}

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(side_effect=mock_create_response)

        # No model provided should use the one from the config
        res_no_model = client.post("/ng-rollout/openai-test/v1/responses", json={"input": "hello"})
        assert res_no_model.status_code == 200
        assert called_args_response.get("model") == "dummy_model"
        assert res_no_model.json()["reasoning"]["effort"] == "minimal"

        # model provided should override config
        res_with_model = client.post("/v1/responses", json={"input": "hello", "model": "override_model"})
        assert res_with_model.status_code == 200
        assert called_args_response.get("model") == "dummy_model"

        server._client.create_response.assert_any_await(input="hello", model="dummy_model")
        calls = read_model_call_records(CaptureStore(tmp_path), "openai-test")
        assert len(calls) == 1
        assert calls[0].dialect == "responses"
        assert calls[0].model_ref is not None
        assert calls[0].model_ref.name == "test_model_server"
        assert calls[0].request == {"input": "hello"}
        assert aggregate_model_call_metrics(CaptureStore(tmp_path), "openai-test")["num_calls"] == 1

    def test_streaming_messages_capture(self, tmp_path) -> None:
        server = self._setup_server()
        server.server_client.global_config_dict = {
            "observability_enabled": True,
            "model_call_capture_dir": str(tmp_path),
        }
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(return_value=_response_data())
        app = server.setup_webserver()
        assert app.user_middleware[0].cls is _CaptureMiddleware
        assert not issubclass(_CaptureMiddleware, BaseHTTPMiddleware)
        client = TestClient(app)

        response = client.post(
            "/ng-rollout/messages-test/v1/messages",
            json={
                "model": "claude-test",
                "max_tokens": 32,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert response.status_code == 200
        assert "event: message_stop" in response.text
        calls = read_model_call_records(CaptureStore(tmp_path), "messages-test")
        assert len(calls) == 1 and calls[0].dialect == "messages"
        assert calls[0].error_category is None

    async def test_responses_parses_hosted_mcp_call(self, monkeypatch: MonkeyPatch) -> None:
        """A server-side ``mcp_call`` output item must validate (200), not 500.

        NVIDIA-hosted gpt-oss surfaces its built-in python tool as an ``mcp_call``;
        before it was in the response schema this returned a 500 that aborted the
        whole rollout collection.
        """
        server = self._setup_server()
        client = TestClient(server.setup_webserver())

        mock_response_data = {
            "id": "resp_mcp",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "type": "mcp_call",
                    "id": "mcp_1",
                    "name": "python",
                    "server_label": "exec",
                    "arguments": '{"code": "print(42)"}',
                    "output": "42\n",
                    "status": "completed",
                },
                {
                    "id": "msg_1",
                    "content": [{"annotations": [], "text": "(Answer: 42)", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(return_value=mock_response_data)

        res = client.post("/v1/responses", json={"input": "compute qed"})
        assert res.status_code == 200
        assert res.json()["output"][0]["type"] == "mcp_call"

    async def test_drop_input_reasoning_items_strips_reasoning(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server(drop_input_reasoning_items=True)
        client = TestClient(server.setup_webserver())

        called_args = {}

        async def mock_create_response(**kwargs):
            nonlocal called_args
            called_args = kwargs
            return {
                "id": "resp_1",
                "created_at": 0.0,
                "model": "dummy_model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(side_effect=mock_create_response)

        res = client.post(
            "/v1/responses",
            json={
                "input": [
                    {"type": "reasoning", "id": "r1", "summary": []},
                    {"role": "user", "content": "hi", "type": "message"},
                ]
            },
        )
        assert res.status_code == 200
        sent_types = [item.get("type") for item in called_args["input"]]
        assert "reasoning" not in sent_types
        assert "message" in sent_types

    @pytest.mark.parametrize("replacement, expected_effort", [("low", "low"), (None, "none")])
    async def test_responses_reasoning_effort_none_replacement_is_configurable(
        self, replacement, expected_effort
    ) -> None:
        server = self._setup_server(reasoning_effort_none_replacement=replacement)
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_data = {
            "id": "resp_1",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "reasoning": {"effort": "none"},
        }
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(return_value=mock_response_data)

        res = client.post("/v1/responses", json={"input": "hello"})
        assert res.status_code == 200
        assert res.json()["reasoning"]["effort"] == expected_effort

    async def test_responses_reasoning_workarounds_apply_when_retrying(self) -> None:
        server = self._setup_server(
            drop_input_reasoning_items=True,
            reasoning_effort_none_replacement="low",
            upstream_max_num_tries=1,
            upstream_retry_policy={"max_attempts": 2, "backoff_initial_seconds": 0},
        )
        response_data = {**_response_data(), "reasoning": {"effort": "none"}}
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(
            side_effect=[TimeoutError("transient provider timeout"), response_data]
        )
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"type": "reasoning", "id": "r1", "summary": []},
                {"role": "user", "content": "hi", "type": "message"},
            ]
        )

        response = await server.responses(body)

        assert response.reasoning.effort == "low"
        assert server._client.create_response.await_count == 2
        for call in server._client.create_response.await_args_list:
            assert call.kwargs == {
                "input": [{"role": "user", "content": "hi", "type": "message"}],
                "model": "dummy_model",
            }
        assert body.input[0].type == "reasoning"
        assert response_data["reasoning"]["effort"] == "none"

    def test_semaphore_disabled_by_default(self) -> None:
        server = self._setup_server()
        assert isinstance(server._semaphore, type(nullcontext()))

    @pytest.mark.asyncio
    async def test_semaphore_caps_concurrency(self) -> None:
        server = self._setup_server(max_concurrent_requests=2)
        assert isinstance(server._semaphore, asyncio.Semaphore)

        in_flight = 0
        peak = 0

        async def worker() -> None:
            nonlocal in_flight, peak
            async with server._semaphore:
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        await asyncio.gather(*(worker() for _ in range(8)))
        assert peak == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("upstream_retry_policy", "upstream_max_num_tries"),
        [
            ({}, None),
            (PROVIDER_RETRY_POLICY, 1),
        ],
    )
    async def test_upstream_calls_respect_max_concurrency(
        self,
        upstream_retry_policy: dict,
        upstream_max_num_tries: int | None,
        monkeypatch: MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(openai_model_module.random, "uniform", lambda _low, _high: 0.0)
        server = self._setup_server(
            max_concurrent_requests=2,
            upstream_retry_policy=upstream_retry_policy,
            upstream_max_num_tries=upstream_max_num_tries,
        )
        release_operations = asyncio.Event()
        two_operations_started = asyncio.Event()
        started = 0
        in_flight = 0
        peak = 0

        async def operation() -> str:
            nonlocal started, in_flight, peak
            started += 1
            in_flight += 1
            peak = max(peak, in_flight)
            if started == 2:
                two_operations_started.set()
            await release_operations.wait()
            in_flight -= 1
            return "completed"

        tasks = [asyncio.create_task(server._call_upstream(operation)) for _ in range(8)]
        await asyncio.wait_for(two_operations_started.wait(), timeout=1.0)

        # Give every queued task an opportunity to run. Only two provider
        # operations can have crossed the semaphore at this point.
        await asyncio.sleep(0)
        assert started == 2
        assert peak == 2

        release_operations.set()
        assert await asyncio.gather(*tasks) == ["completed"] * 8

    @pytest.mark.asyncio
    async def test_upstream_pool_timeout_only_times_slot_acquisition(self) -> None:
        server = self._setup_server(
            max_concurrent_requests=1,
            upstream_pool_timeout_seconds=0.01,
        )
        operation_called = False
        await server._semaphore.acquire()

        async def operation() -> str:
            nonlocal operation_called
            operation_called = True
            return "completed"

        try:
            with pytest.raises(TimeoutError, match="acquiring an upstream provider slot"):
                await server._call_upstream(operation)
        finally:
            server._semaphore.release()

        assert operation_called is False

    @pytest.mark.asyncio
    async def test_retry_backoff_releases_concurrency_slot(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        original_sleep = asyncio.sleep
        first_request_in_backoff = asyncio.Event()
        permit_first_request_retry = asyncio.Event()
        second_operation_started = asyncio.Event()

        async def controlled_sleep(seconds: float) -> None:
            if seconds == 1.0:
                first_request_in_backoff.set()
                await permit_first_request_retry.wait()
            else:
                await original_sleep(0)

        monkeypatch.setattr(openai_model_module.asyncio, "sleep", controlled_sleep)
        monkeypatch.setattr(openai_model_module.random, "uniform", lambda _low, _high: 0.0)
        server = self._setup_server(
            max_concurrent_requests=1,
            upstream_retry_policy=PROVIDER_RETRY_POLICY,
            upstream_max_num_tries=1,
        )
        first_attempts = 0

        async def first_operation() -> str:
            nonlocal first_attempts
            first_attempts += 1
            if first_attempts == 1:
                raise RuntimeError("transient")
            return "first completed"

        async def second_operation() -> str:
            second_operation_started.set()
            return "second completed"

        first_task = asyncio.create_task(server._call_upstream(first_operation))
        await asyncio.wait_for(first_request_in_backoff.wait(), timeout=1.0)

        # With a single slot, this second request can reach the provider only
        # if the failed first attempt released the slot before its backoff.
        second_task = asyncio.create_task(server._call_upstream(second_operation))
        await asyncio.wait_for(second_operation_started.wait(), timeout=1.0)
        assert await second_task == "second completed"

        permit_first_request_retry.set()
        assert await first_task == "first completed"
        assert first_attempts == 2

    def test_multi_attempt_retry_policy_requires_inner_retries_disabled(self) -> None:
        with pytest.raises(ValueError, match="upstream_max_num_tries=1"):
            self._setup_server(
                upstream_retry_policy=PROVIDER_RETRY_POLICY,
            )

    @pytest.mark.asyncio
    async def test_retry_policy_matches_configured_jitter_and_backoff(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        sleeps = []
        jitter_values = iter((0.1, 0.25, 0.1))

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(openai_model_module.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(
            openai_model_module.random,
            "uniform",
            lambda _low, _high: next(jitter_values),
        )
        server = self._setup_server(
            max_concurrent_requests=50,
            upstream_retry_policy=PROVIDER_RETRY_POLICY,
            upstream_max_num_tries=1,
            upstream_request_timeout_seconds=300,
            upstream_connect_timeout_seconds=300,
            upstream_pool_timeout_seconds=300,
        )
        calls = 0

        async def flaky_operation():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient")
            return "completed"

        assert await server._call_upstream(flaky_operation) == "completed"
        assert calls == 2
        assert sleeps == [0.1, 1.25, 0.1]
        assert server._client.max_num_tries == 1
        assert server._client.request_timeout_seconds == 300
        assert server._client.connect_timeout_seconds == 300
        assert server.config.upstream_pool_timeout_seconds == 300

    def test_pool_timeout_requires_a_concurrency_limit(self) -> None:
        with pytest.raises(
            ValueError,
            match="upstream_pool_timeout_seconds requires max_concurrent_requests",
        ):
            self._setup_server(upstream_pool_timeout_seconds=300)

    def test_connect_timeout_requires_a_request_timeout(self) -> None:
        with pytest.raises(
            ValueError,
            match=("upstream_connect_timeout_seconds requires upstream_request_timeout_seconds"),
        ):
            self._setup_server(upstream_connect_timeout_seconds=60)

    def test_propagated_status_codes_must_be_http_errors(self) -> None:
        with pytest.raises(
            ValueError,
            match="must contain HTTP error statuses",
        ):
            self._setup_server(propagate_upstream_http_status_codes=[200, 600])

    @pytest.mark.asyncio
    async def test_retry_policy_does_not_retry_terminal_http_400(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(openai_model_module.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(
            openai_model_module.random,
            "uniform",
            lambda _low, _high: 0.1,
        )
        server = self._setup_server(
            upstream_retry_policy=PROVIDER_RETRY_POLICY,
            upstream_max_num_tries=1,
        )
        calls = 0

        async def bad_request():
            nonlocal calls
            calls += 1
            raise ClientResponseError(
                SimpleNamespace(real_url="https://api.openai.com/v1/responses"),
                (),
                status=400,
                message="bad request",
            )

        with pytest.raises(ClientResponseError):
            await server._call_upstream(bad_request)

        assert calls == 1
        assert sleeps == [0.1]

    @pytest.mark.asyncio
    async def test_responses_preserves_provider_http_400_across_server_hop(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        async def fake_sleep(_seconds):
            return None

        monkeypatch.setattr(openai_model_module.asyncio, "sleep", fake_sleep)
        server = self._setup_server(
            upstream_retry_policy=PROVIDER_RETRY_POLICY,
            upstream_max_num_tries=1,
            propagate_upstream_http_status_codes=[400],
        )
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(
            side_effect=ClientResponseError(
                SimpleNamespace(real_url="https://api.openai.com/v1/responses"),
                (),
                status=400,
                message="bad request",
            )
        )

        with pytest.raises(HTTPException) as exc_info:
            await server.responses(NeMoGymResponseCreateParamsNonStreaming(input="hello"))

        assert exc_info.value.status_code == 400
        assert server._client.create_response.await_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("endpoint", ["responses", "chat_completions"])
    async def test_default_retry_policy_does_not_translate_provider_http_400(
        self,
        endpoint: str,
    ) -> None:
        provider_error = ClientResponseError(
            SimpleNamespace(real_url="https://api.openai.com/v1"),
            (),
            status=400,
            message="bad request",
        )
        server = self._setup_server()
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)

        if endpoint == "responses":
            server._client.create_response = AsyncMock(side_effect=provider_error)
            operation = server.responses(NeMoGymResponseCreateParamsNonStreaming(input="hello"))
        else:
            server._client.create_chat_completion = AsyncMock(side_effect=provider_error)
            operation = server.chat_completions(
                NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "hello"}])
            )

        with pytest.raises(ClientResponseError) as exc_info:
            await operation

        assert exc_info.value is provider_error

    @pytest.mark.parametrize("endpoint", ["responses", "chat_completions"])
    def test_opt_in_preserves_provider_http_400_across_server_hop(self, endpoint: str) -> None:
        provider_error = ClientResponseError(
            SimpleNamespace(real_url="https://api.openai.com/v1"),
            (),
            status=400,
            message="bad request",
        )
        server = self._setup_server(propagate_upstream_http_status_codes=[400])
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        app = server.setup_webserver()
        server.setup_exception_middleware(app)
        client = TestClient(app)

        if endpoint == "responses":
            operation = server._client.create_response = AsyncMock(side_effect=provider_error)
            response = client.post("/v1/responses", json={"input": "hello"})
        else:
            operation = server._client.create_chat_completion = AsyncMock(side_effect=provider_error)
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})

        assert response.status_code == 400
        assert response.json() == {"detail": "Upstream provider request failed with HTTP 400"}
        operation.assert_awaited_once()

    @pytest.mark.parametrize("endpoint", ["responses", "chat_completions"])
    @pytest.mark.parametrize("status_code", [429, 503])
    def test_exhausted_provider_http_retries_return_server_error(self, endpoint: str, status_code: int) -> None:
        provider_error = ClientResponseError(
            SimpleNamespace(real_url="https://api.openai.com/v1"),
            (),
            status=status_code,
            message="private upstream error details",
        )
        server = self._setup_server(
            propagate_upstream_http_status_codes=[status_code],
            upstream_max_num_tries=1,
            upstream_retry_policy={"max_attempts": 2, "backoff_initial_seconds": 0},
        )
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        app = server.setup_webserver()
        server.setup_exception_middleware(app)
        client = TestClient(app)

        if endpoint == "responses":
            operation = server._client.create_response = AsyncMock(side_effect=provider_error)
            response = client.post("/v1/responses", json={"input": "hello"})
        else:
            operation = server._client.create_chat_completion = AsyncMock(side_effect=provider_error)
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})

        assert response.status_code == 500
        assert "after 2 attempts" in response.text
        assert operation.await_count == 2

    @pytest.mark.parametrize("propagate_status_codes", [[], [400], [429]])
    async def test_exhausted_retries_wrap_http_errors_even_with_matching_propagation(
        self, propagate_status_codes
    ) -> None:
        provider_error = ClientResponseError(
            SimpleNamespace(real_url="https://api.openai.com/v1"),
            (),
            status=429,
            message="rate limited",
        )
        server = self._setup_server(
            propagate_upstream_http_status_codes=propagate_status_codes,
            upstream_max_num_tries=1,
            upstream_retry_policy={"max_attempts": 2, "backoff_initial_seconds": 0},
        )
        operation = AsyncMock(side_effect=provider_error)

        with pytest.raises(RuntimeError, match="after 2 attempts") as exc_info:
            await server._call_upstream(operation)

        assert exc_info.value.__cause__ is provider_error
        assert operation.await_count == 2

    @pytest.mark.asyncio
    async def test_retry_policy_stops_after_configured_provider_attempts(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(openai_model_module.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(
            openai_model_module.random,
            "uniform",
            lambda _low, _high: 0.0,
        )
        server = self._setup_server(
            upstream_retry_policy=PROVIDER_RETRY_POLICY,
            upstream_max_num_tries=1,
        )
        calls = 0

        async def always_fails():
            nonlocal calls
            calls += 1
            raise RuntimeError("transient")

        with pytest.raises(RuntimeError, match="after 11 attempts"):
            await server._call_upstream(always_fails)

        assert calls == 11
        assert sleeps == [
            0.0,
            1.0,
            0.0,
            2.0,
            0.0,
            4.0,
            0.0,
            8.0,
            0.0,
            16.0,
            0.0,
            32.0,
            0.0,
            64.0,
            0.0,
            128.0,
            0.0,
            256.0,
            0.0,
            512.0,
            0.0,
        ]

    @pytest.mark.parametrize(
        "pre_request_jitter_seconds",
        [(-0.1, 0.2), (0.2, 0.1)],
    )
    def test_retry_policy_rejects_invalid_pre_request_jitter(
        self,
        pre_request_jitter_seconds: tuple[float, float],
    ) -> None:
        with pytest.raises(ValueError, match="nonnegative, ordered"):
            UpstreamRetryPolicy(
                pre_request_jitter_seconds=pre_request_jitter_seconds,
            )
