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
"""Unit tests for CubeAgent.

Integration tests (requiring a live CubeResourcesServer) are marked with
``@pytest.mark.integration`` and are skipped by default.  Run them with:

    pytest -m integration responses_api_agents/cube_agent/tests/test_cube_agent.py
"""
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, call, patch

import orjson
import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.cube_agent.app import (
    CubeAgent,
    CubeAgentConfig,
    CubeAgentRunRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESOURCES_SERVER_NAME = "cube_resources_server"
_MODEL_SERVER_NAME = "policy_model"


def _make_config(**kwargs) -> CubeAgentConfig:
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="cube_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name=_RESOURCES_SERVER_NAME),
        model_server=ModelServerRef(type="responses_api_models", name=_MODEL_SERVER_NAME),
    )
    defaults.update(kwargs)
    return CubeAgentConfig(**defaults)


def _make_agent(**config_kwargs) -> CubeAgent:
    config = _make_config(**config_kwargs)
    return CubeAgent(config=config, server_client=MagicMock(spec=ServerClient))


def _base_model_response(call_id: str = "call_1", tool_name: str = "click", arguments: dict = None) -> dict:
    """Build a minimal NeMoGymResponse dict with one function_call output."""
    if arguments is None:
        arguments = {"x": 10, "y": 20}
    return {
        "id": "resp_abc123",
        "created_at": 1_700_000_000.0,
        "model": "dummy_model",
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": tool_name,
                "arguments": json.dumps(arguments),
                "id": None,
                "status": None,
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _assistant_message_response(text: str = "Done, no more actions needed.") -> dict:
    """Build a model response that contains only an assistant message (no tool calls)."""
    return {
        "id": "resp_msg",
        "created_at": 1_700_000_001.0,
        "model": "dummy_model",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _seed_session_response(task_id: str = "task_001") -> dict:
    return {
        "obs": [{"role": "user", "content": "You are in a terminal.", "type": "message"}],
        "tools": [
            {
                "type": "function",
                "name": "click",
                "description": "Click at coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x", "y"],
                },
            }
        ],
        "task_id": task_id,
    }


def _step_response_text(output: str = "Clicked.", done: bool = False) -> dict:
    return {"output": output, "content_type": "text/plain", "done": done, "reward": 1.0 if done else 0.0}


def _step_response_image(url: str = "http://localhost:8000/screenshots/sess/step_0001.png", done: bool = False) -> dict:
    return {"output": url, "content_type": "image/png", "done": done, "reward": 0.0}


def _verify_response(reward: float = 1.0) -> dict:
    return {
        "reward": reward,
        "reward_info": {"solved": reward == 1.0},
        "responses_create_params": {"input": []},
        "response": {
            "id": "resp_abc123",
            "created_at": 1_700_000_000.0,
            "model": "dummy_model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
        "task_id": None,
        "seed": None,
    }


def _close_response() -> dict:
    return {"message": "Closed", "success": True}


def _make_mock_response(*json_payloads) -> AsyncMock:
    """Return a shared AsyncMock that cycles through json_payloads on successive .read() calls.

    get_response_json() in nemo_gym.server_utils calls ``await response.read()`` and parses
    the result with orjson — so we need to return bytes, not dicts.
    """
    mock = AsyncMock()
    mock.ok = True
    mock.cookies = None
    mock.read = AsyncMock(side_effect=[orjson.dumps(p) for p in json_payloads])
    return mock


def _inject_fake_cube_schemas():
    """Inject stub cube_standard.schemas into sys.modules so the lazy import in run() works."""
    from pydantic import BaseModel

    class CubeSeedSessionRequest(BaseModel):
        task_id: object = None
        seed: object = None

    class CubeSeedSessionResponse(BaseModel):
        obs: list = []
        tools: list = []
        task_id: str = "task_001"

    class CubeStepRequest(BaseModel):
        call_id: str
        name: str
        arguments: dict

    class CubeStepResponse(BaseModel):
        output: str
        content_type: str = "text/plain"
        done: bool
        reward: float = 0.0
        error: object = None

    pkg = types.ModuleType("resources_servers")
    sub = types.ModuleType("resources_servers.cube_standard")
    schemas = types.ModuleType("resources_servers.cube_standard.schemas")
    schemas.CubeSeedSessionRequest = CubeSeedSessionRequest
    schemas.CubeSeedSessionResponse = CubeSeedSessionResponse
    schemas.CubeStepRequest = CubeStepRequest
    schemas.CubeStepResponse = CubeStepResponse

    sys.modules.setdefault("resources_servers", pkg)
    sys.modules["resources_servers.cube_standard"] = sub
    sys.modules["resources_servers.cube_standard.schemas"] = schemas


# Inject stubs at module load time so all tests benefit.
_inject_fake_cube_schemas()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCubeAgentRun:
    """Tests for the run() episode loop."""

    @pytest.mark.asyncio
    async def test_agent_run_text_episode(self):
        """Happy path: seed → model (fn_call) → step (text, done=True) → verify → close."""
        agent = _make_agent()

        mock_resp = _make_mock_response(
            _seed_session_response(),
            _base_model_response(call_id="call_1"),
            _step_response_text(output="Terminal output.", done=True),
            _verify_response(reward=1.0),
            _close_response(),
        )
        agent.server_client.post = AsyncMock(return_value=mock_resp)

        run_request = CubeAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            task_id="task_001",
        )
        result = await agent.run(run_request)

        assert isinstance(result.reward, float)
        assert result.reward == 1.0

        calls = agent.server_client.post.await_args_list
        url_paths = [c[1]["url_path"] for c in calls]
        assert url_paths[0] == "/seed_session"
        assert url_paths[1] == "/v1/responses"
        assert url_paths[2] == "/step"
        assert url_paths[3] == "/verify"
        assert url_paths[4] == "/close"

        # call_id must be forwarded in the /step body
        step_body = calls[2][1]["json"]
        assert step_body["call_id"] == "call_1"

        # The FunctionCallOutput injected into the agent state should use the text output
        # (not "Screenshot captured.") because content_type is text/plain.
        # We verify this indirectly via the model call input on step 2: the agent would
        # have broken out of the loop on done=True before a second model call, so we
        # just verify there were no extra model calls.
        model_calls = [c for c in calls if c[1]["url_path"] == "/v1/responses"]
        assert len(model_calls) == 1

    @pytest.mark.asyncio
    async def test_agent_image_obs_injects_image_url(self):
        """Image observation must inject two items: FunctionCallOutput + EasyInputMessage(image_url)."""
        agent = _make_agent(max_steps=2)

        # First step: image, not done → loop continues
        # Second model call: no tool calls → break
        mock_resp = _make_mock_response(
            _seed_session_response(),
            _base_model_response(call_id="img_call"),
            _step_response_image(url="http://cube:8000/screenshots/s1/step_0001.png", done=False),
            _assistant_message_response(),
            _verify_response(reward=0.0),
            _close_response(),
        )
        agent.server_client.post = AsyncMock(return_value=mock_resp)

        run_request = CubeAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        await agent.run(run_request)

        calls = agent.server_client.post.await_args_list
        # Find the second model call (after the image step)
        model_calls = [c for c in calls if c[1]["url_path"] == "/v1/responses"]
        assert len(model_calls) == 2, "Expected a second model call after the image step"

        second_model_input = model_calls[1][1]["json"]["input"]

        # Find the FunctionCallOutput with "Screenshot captured." and the image_url message
        fn_call_outputs = [m for m in second_model_input if isinstance(m, dict) and m.get("type") == "function_call_output"]
        image_messages = [
            m for m in second_model_input
            if isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "input_image" for b in m["content"])
        ]

        assert any(o.get("output") == "Screenshot captured." for o in fn_call_outputs), (
            "Expected FunctionCallOutput('Screenshot captured.') for image obs"
        )
        assert len(image_messages) >= 1, "Expected an EasyInputMessage with input_image content block"
        image_url_in_msg = image_messages[0]["content"][0]["image_url"]
        assert image_url_in_msg == "http://cube:8000/screenshots/s1/step_0001.png"

    @pytest.mark.asyncio
    async def test_agent_close_called_on_exception(self):
        """The /close endpoint must be called even when the model server raises an exception."""
        agent = _make_agent()

        seed_mock = AsyncMock()
        seed_mock.ok = True
        seed_mock.cookies = None
        seed_mock.read = AsyncMock(return_value=orjson.dumps(_seed_session_response()))

        # Simulate a model server HTTP error: ok=False causes raise_for_status to fire.
        error_mock = AsyncMock()
        error_mock.ok = False
        error_mock.cookies = None
        error_mock.content = AsyncMock()
        error_mock.content.read = AsyncMock(return_value=b"Model server exploded")
        error_mock.raise_for_status = MagicMock(side_effect=RuntimeError("Model server exploded"))

        close_mock = AsyncMock()
        close_mock.ok = True
        close_mock.cookies = None
        close_mock.read = AsyncMock(return_value=orjson.dumps(_close_response()))

        # seed → model (raises) → close
        agent.server_client.post = AsyncMock(side_effect=[seed_mock, error_mock, close_mock])

        run_request = CubeAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        with pytest.raises(Exception):
            await agent.run(run_request)

        calls = agent.server_client.post.await_args_list
        close_calls = [c for c in calls if c[1]["url_path"] == "/close"]
        assert len(close_calls) == 1, "/close must be called exactly once even after an exception"

    @pytest.mark.asyncio
    async def test_agent_respects_max_steps(self):
        """With max_steps=2 and model always returning tool calls, loop exits after 2 steps."""
        agent = _make_agent(max_steps=2)

        step_resp = _step_response_text(output="ok.", done=False)
        mock_resp = _make_mock_response(
            _seed_session_response(),
            _base_model_response(call_id="c1"),
            step_resp,
            _base_model_response(call_id="c2"),
            step_resp,
            _verify_response(reward=0.0),
            _close_response(),
        )
        agent.server_client.post = AsyncMock(return_value=mock_resp)

        run_request = CubeAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        result = await agent.run(run_request)

        calls = agent.server_client.post.await_args_list
        model_calls = [c for c in calls if c[1]["url_path"] == "/v1/responses"]
        step_calls = [c for c in calls if c[1]["url_path"] == "/step"]

        assert len(model_calls) == 2, "Agent should have called the model exactly max_steps=2 times"
        assert len(step_calls) == 2
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_done_if_no_tool_calls_breaks_loop(self):
        """When model returns only an assistant message, loop breaks with done_if_no_tool_calls=True."""
        agent = _make_agent(done_if_no_tool_calls=True)

        mock_resp = _make_mock_response(
            _seed_session_response(),
            _assistant_message_response("Task complete."),
            _verify_response(reward=0.5),
            _close_response(),
        )
        agent.server_client.post = AsyncMock(return_value=mock_resp)

        run_request = CubeAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        result = await agent.run(run_request)

        calls = agent.server_client.post.await_args_list
        model_calls = [c for c in calls if c[1]["url_path"] == "/v1/responses"]
        step_calls = [c for c in calls if c[1]["url_path"] == "/step"]

        assert len(model_calls) == 1
        assert len(step_calls) == 0
        assert result.reward == 0.5

    @pytest.mark.asyncio
    async def test_cookies_propagated_to_step(self):
        """resources_server_cookies from /seed_session must be forwarded to /step."""
        agent = _make_agent()

        seed_mock = AsyncMock()
        seed_mock.ok = True
        seed_mock.cookies = {"session_id": "abc-session-123"}
        seed_mock.read = AsyncMock(return_value=orjson.dumps(_seed_session_response()))

        other_mock = AsyncMock()
        other_mock.ok = True
        other_mock.cookies = {"session_id": "abc-session-123"}
        other_mock.read = AsyncMock(
            side_effect=[
                orjson.dumps(_base_model_response(call_id="c1")),  # model
                orjson.dumps(_step_response_text(done=True)),  # step
                orjson.dumps(_verify_response()),  # verify
                orjson.dumps(_close_response()),  # close
            ]
        )

        agent.server_client.post = AsyncMock(side_effect=[seed_mock, other_mock, other_mock, other_mock, other_mock])

        run_request = CubeAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        await agent.run(run_request)

        calls = agent.server_client.post.await_args_list
        step_call = next(c for c in calls if c[1]["url_path"] == "/step")
        assert step_call[1].get("cookies") == {"session_id": "abc-session-123"}


# ---------------------------------------------------------------------------
# Integration tests (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCubeAgentIntegration:
    """These tests require a live CubeResourcesServer.  Run with: pytest -m integration"""

    async def test_full_episode_against_live_server(self):
        pytest.skip("Requires running CubeResourcesServer — not available in CI")
