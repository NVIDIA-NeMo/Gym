# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import json as json_module
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.perplexity_summarizer.prompts import TOOL_CALL_DISABLE_SUFFIX
from responses_api_agents.perplexity_summarizer_agent.app import (
    ModelServerRef,
    PerplexitySummarizerAgent,
    PerplexitySummarizerAgentConfig,
    ResourcesServerRef,
)


def _make_config(**overrides):
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="test_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="pplx_sqa"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_tool_calls=3,
    )
    defaults.update(overrides)
    return PerplexitySummarizerAgentConfig(**defaults)


def _make_model_response_with_tool_call(call_id="call_1", fn_name="search_web", arguments='{"queries": ["test"]}'):
    return {
        "id": "resp_1",
        "created_at": 0.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": call_id,
                "name": fn_name,
                "arguments": arguments,
                "status": "completed",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _make_model_response_with_message(text="Hello!"):
    return {
        "id": "resp_2",
        "created_at": 0.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class TestApp:
    def test_sanity(self):
        config = _make_config()
        PerplexitySummarizerAgent(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_responses_basic(self):
        """Test basic responses flow: model returns a message directly."""
        config = _make_config()
        server = PerplexitySummarizerAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response = _make_model_response_with_message("The answer is 42.")
        dotjson_mock = AsyncMock()
        dotjson_mock.read.return_value = json_module.dumps(mock_response)
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res.status_code == 200
        assert "The answer is 42." in res.text

    async def test_tool_call_counting(self):
        """Tool calls are correctly counted across multiple steps."""
        config = _make_config(max_tool_calls=2)
        server = PerplexitySummarizerAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        # Step 1: model returns tool call
        tool_call_resp_1 = _make_model_response_with_tool_call(call_id="call_1")
        # Step 2: model returns tool call (reaches limit)
        tool_call_resp_2 = _make_model_response_with_tool_call(call_id="call_2")
        # Step 3: model returns message (forced by tool_choice=none)
        message_resp = _make_model_response_with_message("Final answer")

        tool_output_mock = AsyncMock()
        tool_output_mock.content.read = AsyncMock(return_value=b'{"search_results": "[]"}')
        tool_output_mock.cookies = MagicMock()

        model_mock = AsyncMock()
        model_mock.read = AsyncMock()
        model_mock.cookies = MagicMock()

        call_count = 0

        async def mock_post(server_name, url_path, json=None, cookies=None):
            nonlocal call_count
            if url_path == "/v1/responses":
                call_count += 1
                model_mock.read.return_value = {
                    1: json_module.dumps(tool_call_resp_1),
                    2: json_module.dumps(tool_call_resp_2),
                    3: json_module.dumps(message_resp),
                }.get(call_count, json_module.dumps(message_resp))
                return model_mock
            else:
                return tool_output_mock

        server.server_client.post = mock_post

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "test"}]})
        assert res.status_code == 200
        # 3 model calls: 2 with tool calls + 1 with tool_choice=none
        assert call_count == 3

    async def test_tool_choice_none_after_limit(self):
        """After max_tool_calls, tool_choice is set to 'none'."""
        config = _make_config(max_tool_calls=1)
        server = PerplexitySummarizerAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        tool_call_resp = _make_model_response_with_tool_call()
        message_resp = _make_model_response_with_message("Done")

        tool_output_mock = AsyncMock()
        tool_output_mock.content.read = AsyncMock(return_value=b'{"search_results": "[]"}')
        tool_output_mock.cookies = MagicMock()

        model_mock = AsyncMock()
        model_mock.cookies = MagicMock()

        posted_bodies = []
        call_idx = 0

        async def mock_post(server_name, url_path, json=None, cookies=None):
            nonlocal call_idx
            if url_path == "/v1/responses":
                call_idx += 1
                # Capture the body that was sent
                if json is not None:
                    posted_bodies.append(json)
                model_mock.read = AsyncMock(
                    return_value=json_module.dumps(tool_call_resp if call_idx == 1 else message_resp)
                )
                return model_mock
            else:
                return tool_output_mock

        server.server_client.post = mock_post

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "test"}]})
        assert res.status_code == 200

        # The second model call should have tool_choice="none"
        assert len(posted_bodies) >= 2
        second_body = posted_bodies[1]
        # posted_bodies contains NeMoGymResponseCreateParamsNonStreaming objects
        assert second_body.tool_choice == "none"

    async def test_unlimited_tool_calls(self):
        """max_tool_calls=-1 should never limit tool calls."""
        config = _make_config(max_tool_calls=-1)
        server = PerplexitySummarizerAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        # 5 tool calls then a message
        call_idx = 0

        tool_output_mock = AsyncMock()
        tool_output_mock.content.read = AsyncMock(return_value=b'{"search_results": "[]"}')
        tool_output_mock.cookies = MagicMock()

        model_mock = AsyncMock()
        model_mock.cookies = MagicMock()

        posted_bodies = []

        async def mock_post(server_name, url_path, json=None, cookies=None):
            nonlocal call_idx
            if url_path == "/v1/responses":
                call_idx += 1
                if json is not None:
                    posted_bodies.append(json)
                resp = (
                    _make_model_response_with_tool_call(call_id=f"call_{call_idx}")
                    if call_idx <= 5
                    else _make_model_response_with_message("Final")
                )
                model_mock.read = AsyncMock(return_value=json_module.dumps(resp))
                return model_mock
            else:
                return tool_output_mock

        server.server_client.post = mock_post

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "test"}]})
        assert res.status_code == 200

        # None of the bodies should have tool_choice="none"
        for body in posted_bodies:
            assert body.tool_choice != "none"

    async def test_usage_accumulation(self):
        """Token usage is accumulated across turns."""
        config = _make_config(max_tool_calls=1)
        server = PerplexitySummarizerAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        tool_call_resp = _make_model_response_with_tool_call()
        tool_call_resp["usage"] = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

        message_resp = _make_model_response_with_message("Done")
        message_resp["usage"] = {
            "input_tokens": 20,
            "output_tokens": 10,
            "total_tokens": 30,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

        tool_output_mock = AsyncMock()
        tool_output_mock.content.read = AsyncMock(return_value=b'{"search_results": "[]"}')
        tool_output_mock.cookies = MagicMock()

        model_mock = AsyncMock()
        model_mock.cookies = MagicMock()

        call_idx = 0

        async def mock_post(server_name, url_path, json=None, cookies=None):
            nonlocal call_idx
            if url_path == "/v1/responses":
                call_idx += 1
                model_mock.read = AsyncMock(
                    return_value=json_module.dumps(tool_call_resp if call_idx == 1 else message_resp)
                )
                return model_mock
            else:
                return tool_output_mock

        server.server_client.post = mock_post

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "test"}]})
        assert res.status_code == 200
        data = res.json()
        # Usage accumulation follows simple_agent pattern: first call sets usage then adds to it,
        # so first call tokens are counted twice: (10+10)+20=40 input, (5+5)+10=20 output
        assert data["usage"]["input_tokens"] == 40
        assert data["usage"]["output_tokens"] == 20
        assert data["usage"]["total_tokens"] == 60

    async def test_tool_call_disable_hint_appended(self):
        """When tool limit reached, TOOL_CALL_DISABLE_SUFFIX is appended to last tool result."""
        config = _make_config(max_tool_calls=1)
        server = PerplexitySummarizerAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        tool_call_resp = _make_model_response_with_tool_call()
        message_resp = _make_model_response_with_message("Summary")

        tool_output_mock = AsyncMock()
        tool_output_mock.content.read = AsyncMock(return_value=b'[{"id": "web:1", "content": "result"}]')
        tool_output_mock.cookies = MagicMock()

        model_mock = AsyncMock()
        model_mock.cookies = MagicMock()

        posted_bodies = []
        call_idx = 0

        async def mock_post(server_name, url_path, json=None, cookies=None):
            nonlocal call_idx
            if url_path == "/v1/responses":
                call_idx += 1
                if json is not None:
                    posted_bodies.append(json)
                model_mock.read = AsyncMock(
                    return_value=json_module.dumps(tool_call_resp if call_idx == 1 else message_resp)
                )
                return model_mock
            else:
                return tool_output_mock

        server.server_client.post = mock_post

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "test"}]})
        assert res.status_code == 200

        # Second model call should have the hint appended to the last function_call_output
        assert len(posted_bodies) >= 2
        second_body = posted_bodies[1]
        # Find the last function_call_output in input
        fn_outputs = [
            item for item in second_body.input if hasattr(item, "type") and item.type == "function_call_output"
        ]
        assert len(fn_outputs) >= 1
        last_fn_output = fn_outputs[-1]
        assert last_fn_output.output.endswith(TOOL_CALL_DISABLE_SUFFIX)

    async def test_tool_call_disable_hint_idempotent(self):
        """Hint is not appended twice if already present."""
        config = _make_config(max_tool_calls=0)
        server = PerplexitySummarizerAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        message_resp = _make_model_response_with_message("Summary")

        model_mock = AsyncMock()
        model_mock.cookies = MagicMock()

        posted_bodies = []

        async def mock_post(server_name, url_path, json=None, cookies=None):
            if url_path == "/v1/responses":
                if json is not None:
                    posted_bodies.append(json)
                model_mock.read = AsyncMock(return_value=json_module.dumps(message_resp))
                return model_mock

        server.server_client.post = mock_post

        # Input already has a function_call_output with the suffix
        tool_output_with_hint = '[{"id": "web:1"}]' + TOOL_CALL_DISABLE_SUFFIX
        res = client.post(
            "/v1/responses",
            json={
                "input": [
                    {"role": "user", "content": "test"},
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "c1",
                        "name": "search_web",
                        "arguments": "{}",
                        "status": "completed",
                    },
                    {"type": "function_call_output", "call_id": "c1", "output": tool_output_with_hint},
                ]
            },
        )
        assert res.status_code == 200

        # The suffix should appear exactly once, not doubled
        assert len(posted_bodies) >= 1
        fn_outputs = [
            item for item in posted_bodies[0].input if hasattr(item, "type") and item.type == "function_call_output"
        ]
        assert len(fn_outputs) >= 1
        assert fn_outputs[-1].output.count(TOOL_CALL_DISABLE_SUFFIX) == 1

    async def test_bad_words_injected_when_configured(self):
        """When bad_words is set in config, metadata.extra_body contains bad_words."""
        config = _make_config(max_tool_calls=1, bad_words=["<tool_call>", "</tool_call>"])
        server = PerplexitySummarizerAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        tool_call_resp = _make_model_response_with_tool_call()
        message_resp = _make_model_response_with_message("Done")

        tool_output_mock = AsyncMock()
        tool_output_mock.content.read = AsyncMock(return_value=b'{"search_results": "[]"}')
        tool_output_mock.cookies = MagicMock()

        model_mock = AsyncMock()
        model_mock.cookies = MagicMock()

        posted_bodies = []
        call_idx = 0

        async def mock_post(server_name, url_path, json=None, cookies=None):
            nonlocal call_idx
            if url_path == "/v1/responses":
                call_idx += 1
                if json is not None:
                    posted_bodies.append(json)
                model_mock.read = AsyncMock(
                    return_value=json_module.dumps(tool_call_resp if call_idx == 1 else message_resp)
                )
                return model_mock
            else:
                return tool_output_mock

        server.server_client.post = mock_post

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "test"}]})
        assert res.status_code == 200

        # Second body should have bad_words in metadata
        assert len(posted_bodies) >= 2
        second_body = posted_bodies[1]
        assert second_body.metadata is not None
        extra_body = json_module.loads(second_body.metadata["extra_body"])
        assert extra_body["bad_words"] == ["<tool_call>", "</tool_call>"]

    async def test_bad_words_not_injected_when_none(self):
        """When bad_words is None, no metadata is set."""
        config = _make_config(max_tool_calls=1, bad_words=None)
        server = PerplexitySummarizerAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        tool_call_resp = _make_model_response_with_tool_call()
        message_resp = _make_model_response_with_message("Done")

        tool_output_mock = AsyncMock()
        tool_output_mock.content.read = AsyncMock(return_value=b'{"search_results": "[]"}')
        tool_output_mock.cookies = MagicMock()

        model_mock = AsyncMock()
        model_mock.cookies = MagicMock()

        posted_bodies = []
        call_idx = 0

        async def mock_post(server_name, url_path, json=None, cookies=None):
            nonlocal call_idx
            if url_path == "/v1/responses":
                call_idx += 1
                if json is not None:
                    posted_bodies.append(json)
                model_mock.read = AsyncMock(
                    return_value=json_module.dumps(tool_call_resp if call_idx == 1 else message_resp)
                )
                return model_mock
            else:
                return tool_output_mock

        server.server_client.post = mock_post

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "test"}]})
        assert res.status_code == 200

        # Second body should NOT have metadata set
        assert len(posted_bodies) >= 2
        second_body = posted_bodies[1]
        assert second_body.metadata is None
