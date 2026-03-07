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
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.gdpval_agent.app import Cookies, GDPValAgent, GDPValAgentConfig


def _make_config(**kwargs) -> GDPValAgentConfig:
    defaults = dict(
        name="gdpval_agent",
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="my_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="my_model"),
    )
    defaults.update(kwargs)
    return GDPValAgentConfig(**defaults)


def _make_agent(config: GDPValAgentConfig) -> GDPValAgent:
    return GDPValAgent(config=config, server_client=MagicMock(spec=ServerClient))


def _make_http_mock(read_data: dict, content_read_data: bytes | None = None) -> AsyncMock:
    """Build a mock aiohttp ClientResponse-like object."""
    mock = AsyncMock()
    mock.ok = True
    mock.cookies = {}
    mock.read.return_value = json.dumps(read_data).encode()
    if content_read_data is not None:
        mock.content.read.return_value = content_read_data
    return mock


# Minimal valid Response JSON template (openai API shape)
def _model_response_json(output: list, usage: dict | None = None) -> dict:
    data = {
        "id": "resp_test",
        "created_at": 1234567890.0,
        "model": "test_model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if usage is not None:
        data["usage"] = usage
    return data


def _assistant_message_output(text: str, msg_id: str = "msg_test") -> dict:
    return {
        "id": msg_id,
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def _finish_call_output(call_id: str = "call_finish") -> dict:
    return {
        "id": "fc_finish",
        "call_id": call_id,
        "name": "finish",
        "arguments": "{}",
        "type": "function_call",
    }


def _usage_dict(total_tokens: int) -> dict:
    return {
        "input_tokens": total_tokens - 100,
        "output_tokens": 100,
        "total_tokens": total_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


class TestGDPValAgent:
    def test_sanity(self) -> None:
        config = _make_config()
        GDPValAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_config_defaults(self) -> None:
        config = _make_config()

        assert config.max_steps == 100
        assert config.context_window_tokens == 128000
        assert config.context_summarization_cutoff == 0.7
        assert config.remaining_step_warning_threshold == 20

    async def test_summarize_messages(self) -> None:
        from unittest.mock import patch

        config = _make_config()
        agent = _make_agent(config)

        system_msg = NeMoGymEasyInputMessage(role="system", content="You are an agent.")
        user_msg = NeMoGymEasyInputMessage(role="user", content="Do the task.")
        # A non-initial message (function call output) — breaks task_context extraction
        history_msg = MagicMock()

        messages = [system_msg, user_msg, history_msg]
        model_params = NeMoGymResponseCreateParamsNonStreaming(input=[system_msg, user_msg])

        # Mock single_response to return a response with summary text
        summary_text = "Summary: made progress on the task."
        mock_content_block = MagicMock()
        mock_content_block.text = summary_text
        mock_output_item = MagicMock()
        mock_output_item.content = [mock_content_block]
        mock_response = MagicMock()
        mock_response.output = [mock_output_item]

        # Patch at class level (pydantic models disallow instance-level method patching)
        async def fake_single_response(_self, _params, _cookies):
            return mock_response, Cookies()

        with patch("responses_api_agents.gdpval_agent.app.GDPValAgent.single_response", fake_single_response):
            result, _ = await agent.summarize_messages(messages, model_params, Cookies())

        # task_context (system + user) + bridge message + acknowledgement
        assert len(result) == 4
        assert result[0] is system_msg
        assert result[1] is user_msg
        assert isinstance(result[2], NeMoGymEasyInputMessage)
        assert result[2].role == "user"
        assert summary_text in result[2].content
        assert result[3].content == "Got it, thanks!"

    def test_summarization_triggered_when_tokens_exceed_cutoff(self, tmp_path: Path) -> None:
        config = _make_config(
            context_window_tokens=1000,
            context_summarization_cutoff=0.7,
            max_steps=5,
            remaining_step_warning_threshold=None,
        )
        agent = _make_agent(config)
        app = agent.setup_webserver()
        client = TestClient(app)

        output_dir = str(tmp_path / "output")

        # Build mock side_effect sequence for server_client.post:
        # 1. seed_session → resources_server
        seed_mock = _make_http_mock({"session_id": "test_session_001"})

        # 2. step 1 model call → assistant message + 800 tokens (80% > 70% cutoff)
        step1_mock = _make_http_mock(
            _model_response_json(
                output=[_assistant_message_output("Working on it...", "msg_step1")],
                usage=_usage_dict(800),
            )
        )

        # 3. summarization model call → returns summary text
        summary_mock = _make_http_mock(
            _model_response_json(
                output=[_assistant_message_output("Summary of work done so far.", "msg_summary")],
            )
        )

        # 4. step 2 model call → finish tool call + 400 tokens (40% < 70%)
        step2_mock = _make_http_mock(
            _model_response_json(
                output=[_finish_call_output("call_finish_001")],
                usage=_usage_dict(400),
            )
        )

        # 5. finish tool response → resources_server
        finish_tool_mock = _make_http_mock({})
        finish_tool_mock.content.read.return_value = b'{"saved": []}'

        agent.server_client.post.side_effect = [
            seed_mock,
            step1_mock,
            summary_mock,
            step2_mock,
            finish_tool_mock,
        ]

        request_body = {
            "task_prompt": "Test task",
            "system_prompt": "You are an agent.",
            "output_dir": output_dir,
            "task_id": "test_task_001",
            "responses_create_params": {"input": [{"role": "user", "content": "placeholder", "type": "message"}]},
        }

        res = client.post("/v1/responses", json=request_body)

        assert res.status_code == 200
        # Confirm all 5 server_client.post calls were made (summarization happened)
        assert agent.server_client.post.call_count == 5
