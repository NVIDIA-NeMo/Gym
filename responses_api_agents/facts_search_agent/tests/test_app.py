# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.facts_search_agent.app import FINAL_REQUEST, FACTSSearchAgent, FACTSSearchAgentConfig


class _Response:
    status = 200
    ok = True
    cookies = {}

    def __init__(self, payload=None, content=""):
        self.payload = payload or {}
        self.content = MagicMock()
        self.content.read = AsyncMock(return_value=content.encode())

    async def read(self):
        return json.dumps(self.payload).encode()


def _base(response_id, output):
    return {
        "id": response_id,
        "created_at": 0,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _tool(call_id):
    return {
        "id": f"fc-{call_id}",
        "call_id": call_id,
        "name": "brave_search",
        "arguments": json.dumps({"query": call_id}),
        "type": "function_call",
        "status": "completed",
    }


def _message(text):
    return {
        "id": "message",
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def _agent(max_hops=7):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    config = FACTSSearchAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="facts_search_agent",
        resources_server={"type": "resources_servers", "name": "facts_search"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        max_hops=max_hops,
    )
    return FACTSSearchAgent(config=config, server_client=client), client


def _body():
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "question"}],
        tools=[
            {
                "type": "function",
                "name": "brave_search",
                "description": "Search for information on the web",
                "parameters": {"type": "object", "properties": {}, "required": []},
                "strict": False,
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
    )


async def test_executes_every_parallel_call_for_seven_hops_then_forces_tool_free_final():
    agent, client = _agent()
    model_payloads = []
    for hop in range(7):
        calls = [_tool(f"{hop}-a"), _tool(f"{hop}-b")] if hop == 0 else [_tool(f"{hop}-a")]
        model_payloads.append(_base(f"hop-{hop}", calls))
    model_payloads.append(_base("final", [_message("answer")]))
    payloads = iter(model_payloads)
    model_bodies = []

    async def post(*, server_name, url_path, json=None, **kwargs):
        if server_name == "policy_model":
            model_bodies.append(json)
            return _Response(next(payloads))
        assert server_name == "facts_search" and url_path == "/brave_search"
        return _Response(content='{"result":"search result"}')

    client.post = AsyncMock(side_effect=post)
    response, trajectory, _, _ = await agent._create_episode(
        _body(), model_url_path="/v1/responses", collect_trajectory=True, task_id="task", rollout_id="0-0"
    )
    assert response.metadata == {
        "facts_search_hops": "7",
        "facts_search_queries": "8",
        "facts_search_forced_final": "true",
    }
    assert len(model_bodies) == 8
    assert model_bodies[-1].tools == [] and model_bodies[-1].tool_choice == "none"
    assert model_bodies[-1].input[-1].content == FINAL_REQUEST
    assert trajectory is not None and len(trajectory.turns) == 8 and len(trajectory.tool_calls) == 8
    assert response.output[-1].type == "message"


async def test_stops_on_ordinary_answer_without_forced_final():
    agent, client = _agent()
    client.post = AsyncMock(return_value=_Response(_base("answer", [_message("Paris")])))
    response, trajectory, _, _ = await agent._create_episode(
        _body(), model_url_path="/v1/responses", collect_trajectory=False
    )
    assert trajectory is None
    assert response.metadata["facts_search_hops"] == "0"
    assert response.metadata["facts_search_forced_final"] == "false"
    assert client.post.await_count == 1


async def test_string_input_malformed_tool_args_and_failed_tool_are_preserved():
    agent, client = _agent(max_hops=1)
    bad = _tool("bad")
    bad["arguments"] = "{"  # invalid JSON
    payloads = iter([_base("hop", [bad, _tool("failed")]), _base("final", [_message("answer")])])

    async def post(*, server_name, url_path, json=None, **kwargs):
        if server_name == "policy_model":
            return _Response(next(payloads))
        return_value = _Response(content="service unavailable")
        return_value.status = 503
        return_value.ok = False
        return return_value

    client.post = AsyncMock(side_effect=post)
    body = _body().model_copy(update={"input": "question"})
    response, trajectory, _, _ = await agent._create_episode(
        body, model_url_path="/v1/responses", collect_trajectory=True, task_id="task", rollout_id="run"
    )
    assert response.metadata["facts_search_queries"] == "2"
    assert trajectory is not None
    assert [call.status for call in trajectory.tool_calls] == ["failed", "failed"]
    assert trajectory.tool_calls[0].error_type == "JSONDecodeError"
    assert trajectory.tool_calls[1].error_type == "http_503"


async def test_incomplete_and_empty_model_outputs_stop_without_fabricating_an_answer():
    for payload in (
        _base("incomplete", [_message("partial")]) | {"incomplete_details": {"reason": "max_output_tokens"}},
        _base("empty", []),
    ):
        agent, client = _agent()
        client.post = AsyncMock(return_value=_Response(payload))
        response, trajectory, _, _ = await agent._create_episode(
            _body(), model_url_path="/v1/responses", collect_trajectory=True
        )
        assert response.metadata["facts_search_forced_final"] == "false"
        assert trajectory is not None and trajectory.invocations[0].status == "incomplete"
