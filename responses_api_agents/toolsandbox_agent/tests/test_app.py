# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the ToolSandbox agent harness.

The harness drives a stateful, multi-turn conversation against the resources
server (``seed_session`` -> ``step`` -> ``close``) while sampling turns from the
policy model server. These tests mock ``ServerClient.post`` with a small router
that returns canned payloads per ``url_path`` so the full episode loop can be
exercised without any live server.
"""

from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

from pytest import fixture, mark, raises
from tenacity import RetryError, wait_none

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.toolsandbox_agent.app import (
    ToolSandboxAgent,
    ToolSandboxAgentConfig,
    ToolSandboxAgentRunRequest,
    _assistant_text,
)


class _FakeResp:
    """Minimal stand-in for an aiohttp response the harness consumes."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.cookies: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return self._payload


def _post_router(routes: dict[str, list[dict[str, Any]]]) -> Callable[..., Any]:
    """Build an async ``post`` side effect that dispatches on ``url_path``.

    Each path maps to a FIFO queue of payloads; once a queue is down to its last
    entry that entry is returned for every remaining call (so a steady-state
    response can be repeated across an unbounded episode loop).
    """
    queues = {path: list(payloads) for path, payloads in routes.items()}

    async def _post(*, server_name: str, url_path: str, json: Any = None, cookies: Any = None) -> _FakeResp:
        queue = queues[url_path]
        payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return _FakeResp(payload)

    return _post


def _seed_payload(obs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "env_id": "env-1",
        "scenario": "scenario-a",
        "obs": [{"role": "user", "content": "You are in a sandbox."}] if obs is None else obs,
        "tools": [{"name": "get_time", "parameters": None, "strict": None, "type": "function", "description": None}],
    }


def _text_response(text: str, response_id: str = "resp-text") -> dict[str, Any]:
    return {
        "id": response_id,
        "created_at": 1,
        "model": "policy-model",
        "object": "response",
        "output": [
            {
                "id": "msg-1",
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


def _function_call_response(name: str = "get_time", arguments: str = "{}", call_id: str = "call-1") -> dict[str, Any]:
    return {
        "id": "resp-fn",
        "created_at": 1,
        "model": "policy-model",
        "object": "response",
        "output": [{"arguments": arguments, "call_id": call_id, "name": name, "type": "function_call"}],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _empty_response() -> dict[str, Any]:
    return {
        "id": "resp-empty",
        "created_at": 1,
        "model": "policy-model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _step_payload(done: bool, obs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "obs": [{"role": "user", "content": "Thanks, carry on."}] if obs is None else obs,
        "reward": 0.0,
        "done": done,
    }


class TestApp:
    @fixture(autouse=True)
    def _no_retry_backoff(self) -> Any:
        # ``_seed_session`` retries with an exponential backoff; null the wait so
        # the deliberately-failing seed tests resolve without real sleeping.
        original = ToolSandboxAgent._seed_session.retry.wait
        ToolSandboxAgent._seed_session.retry.wait = wait_none()
        yield
        ToolSandboxAgent._seed_session.retry.wait = original

    @fixture
    def agent_config(self) -> ToolSandboxAgentConfig:
        return ToolSandboxAgentConfig(
            host="localhost",
            port=10002,
            entrypoint="",
            name="toolsandbox_agent",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="toolsandbox_resources_server",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="model_server",
            ),
        )

    def _make_agent(self, config: ToolSandboxAgentConfig, post: Callable[..., Any]) -> ToolSandboxAgent:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=post)
        return ToolSandboxAgent(config=config, server_client=server_client)

    def _run_request(self, task_idx: int = 0, content: str = "Hi there.") -> ToolSandboxAgentRunRequest:
        return ToolSandboxAgentRunRequest(
            task_idx=task_idx,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content=content)]
            ),
        )

    def _calls_to(self, agent: ToolSandboxAgent, url_path: str) -> list[Any]:
        return [c for c in agent.server_client.post.call_args_list if c.kwargs["url_path"] == url_path]

    def test_assistant_text_flattens_and_ignores_non_text(self) -> None:
        message = NeMoGymResponseOutputMessage.model_validate(
            {
                "id": "m",
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "content": [
                    {"annotations": [], "text": "Hello ", "type": "output_text"},
                    {"annotations": [], "text": "world", "type": "output_text"},
                ],
            }
        )
        assert _assistant_text([message]) == "Hello world"
        assert _assistant_text([]) == ""

    async def test_responses_text_turn_ends_episode(self, agent_config: ToolSandboxAgentConfig) -> None:
        post = _post_router(
            {
                "/seed_session": [_seed_payload()],
                "/v1/responses": [_text_response("Here is my answer.")],
                "/step": [_step_payload(done=True)],
                "/close": [{"message": "closed", "success": True}],
            }
        )
        agent = self._make_agent(agent_config, post)

        response = await agent.responses(self._run_request())

        assert response.env_id == "env-1"
        assert response.group_id == "0"
        assert response.contains_transitions is False
        # Flat trajectory: the assistant turn plus the observation from /step.
        assert len(response.output) == 2

        # A natural-language turn is forwarded to the user simulator as text with
        # no function calls.
        step_json = self._calls_to(agent, "/step")[0].kwargs["json"]
        assert step_json["text"] == "Here is my answer."
        assert step_json["function_calls"] == []
        assert step_json["env_id"] == "env-1"

        # The session is always closed exactly once.
        assert len(self._calls_to(agent, "/close")) == 1

    async def test_responses_tool_call_turn(self, agent_config: ToolSandboxAgentConfig) -> None:
        post = _post_router(
            {
                "/seed_session": [_seed_payload()],
                "/v1/responses": [_function_call_response(name="get_time", call_id="c-9")],
                "/step": [_step_payload(done=True, obs=[{"role": "user", "content": "It is noon."}])],
                "/close": [{"message": "closed", "success": True}],
            }
        )
        agent = self._make_agent(agent_config, post)

        response = await agent.responses(self._run_request())

        # Tool calls take precedence over text: the function call is forwarded and
        # ``text`` is suppressed.
        step_json = self._calls_to(agent, "/step")[0].kwargs["json"]
        assert step_json["text"] is None
        assert len(step_json["function_calls"]) == 1
        assert step_json["function_calls"][0]["name"] == "get_time"
        assert step_json["function_calls"][0]["call_id"] == "c-9"
        assert response.env_id == "env-1"

    async def test_responses_no_actionable_output_breaks(self, agent_config: ToolSandboxAgentConfig) -> None:
        post = _post_router(
            {
                "/seed_session": [_seed_payload()],
                "/v1/responses": [_empty_response()],
                "/close": [{"message": "closed", "success": True}],
            }
        )
        agent = self._make_agent(agent_config, post)

        response = await agent.responses(self._run_request())

        # An empty model turn ends the episode before any /step call.
        assert self._calls_to(agent, "/step") == []
        assert response.output == []
        assert len(self._calls_to(agent, "/close")) == 1

    async def test_responses_respects_max_steps(self, agent_config: ToolSandboxAgentConfig) -> None:
        config = agent_config.model_copy(update={"max_steps": 3})
        post = _post_router(
            {
                "/seed_session": [_seed_payload()],
                "/v1/responses": [_text_response("keep going")],
                "/step": [_step_payload(done=False)],
                "/close": [{"message": "closed", "success": True}],
            }
        )
        agent = self._make_agent(config, post)

        response = await agent.responses(self._run_request())

        # The loop is force-ended after max_steps turns even without ``done``.
        assert len(self._calls_to(agent, "/v1/responses")) == 3
        assert len(self._calls_to(agent, "/step")) == 3
        # Each turn contributes the assistant message plus one observation.
        assert len(response.output) == 6
        assert len(self._calls_to(agent, "/close")) == 1

    async def test_responses_return_transitions(self, agent_config: ToolSandboxAgentConfig) -> None:
        config = agent_config.model_copy(update={"return_transitions": True})
        post = _post_router(
            {
                "/seed_session": [_seed_payload()],
                "/v1/responses": [_text_response("done")],
                "/step": [_step_payload(done=True)],
                "/close": [{"message": "closed", "success": True}],
            }
        )
        agent = self._make_agent(config, post)

        response = await agent.responses(self._run_request())

        assert response.contains_transitions is True
        # One transition snapshot, each snapshot itself a list of messages.
        assert len(response.output) == 1
        assert isinstance(response.output[0], list)

    async def test_responses_string_input_is_wrapped(self, agent_config: ToolSandboxAgentConfig) -> None:
        post = _post_router(
            {
                "/seed_session": [_seed_payload()],
                "/v1/responses": [_text_response("ok")],
                "/step": [_step_payload(done=True)],
                "/close": [{"message": "closed", "success": True}],
            }
        )
        agent = self._make_agent(agent_config, post)

        request = ToolSandboxAgentRunRequest(
            task_idx=2,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="just a string"),
        )
        response = await agent.responses(request)

        assert response.group_id == "2"
        # The bare-string input is normalised into a user message ahead of the
        # seeded observations, so the first model call sees both.
        first_model_input = self._calls_to(agent, "/v1/responses")[0].kwargs["json"].input
        assert first_model_input[0].content == "just a string"

    async def test_seed_session_without_obs_raises(self, agent_config: ToolSandboxAgentConfig) -> None:
        post = _post_router({"/seed_session": [_seed_payload(obs=[])]})
        agent = self._make_agent(agent_config, post)

        with raises(RetryError):
            await agent.responses(self._run_request())
        # Seeding is attempted three times before giving up.
        assert len(self._calls_to(agent, "/seed_session")) == 3

    async def test_run_invokes_verify(self, agent_config: ToolSandboxAgentConfig) -> None:
        verify_payload = {
            "responses_create_params": {"input": []},
            "response": {
                **_text_response("final"),
                "env_id": "env-1",
                "group_id": "0",
                "contains_transitions": False,
                "output": [],
            },
            "reward": 1.0,
        }
        post = _post_router(
            {
                "/seed_session": [_seed_payload()],
                "/v1/responses": [_text_response("final")],
                "/step": [_step_payload(done=True)],
                "/close": [{"message": "closed", "success": True}],
                "/verify": [verify_payload],
            }
        )
        agent = self._make_agent(agent_config, post)

        verify_response = await agent.run(self._run_request())

        assert verify_response.reward == 1.0
        assert verify_response.response.env_id == "env-1"
        # /verify is called against the resources server with the built response.
        verify_calls = self._calls_to(agent, "/verify")
        assert len(verify_calls) == 1
        assert verify_calls[0].kwargs["server_name"] == "toolsandbox_resources_server"
        assert verify_calls[0].kwargs["json"]["response"]["env_id"] == "env-1"

    async def test_run_propagates_errors(self, agent_config: ToolSandboxAgentConfig) -> None:
        async def _boom(**_kwargs: Any) -> _FakeResp:
            raise RuntimeError("seed exploded")

        agent = self._make_agent(agent_config, _boom)

        with raises(RetryError):
            await agent.run(self._run_request())


@mark.parametrize(
    "payload",
    [_text_response("hi"), _function_call_response(), _empty_response()],
)
def test_response_builders_are_valid(payload: dict[str, Any]) -> None:
    # Guard the canned fixtures themselves: each must parse as a model response.
    from nemo_gym.openai_utils import NeMoGymResponse

    NeMoGymResponse.model_validate(payload)
