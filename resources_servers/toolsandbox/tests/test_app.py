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
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.toolsandbox.app import ToolSandboxResourcesServer, _ServerClientUser
from resources_servers.toolsandbox.schemas import (
    ToolSandboxCloseRequest,
    ToolSandboxNeMoGymResponse,
    ToolSandboxResourcesServerConfig,
    ToolSandboxSeedSessionRequest,
    ToolSandboxStepRequest,
    ToolSandboxVerifyRequest,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _chat_completion(content=None, tool_calls=None) -> dict:
    """Minimal OpenAI ChatCompletion dict the user-sim model returns."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "cc-1",
        "object": "chat.completion",
        "created": 0,
        "model": "user-sim",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }


def _mock_server_client(cc_dict: dict) -> MagicMock:
    """A ServerClient whose /v1/chat/completions returns ``cc_dict``."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = AsyncMock(return_value=cc_dict)
    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(return_value=resp)
    return client


def _make_server(cc_dict=None, **config_overrides) -> ToolSandboxResourcesServer:
    if cc_dict is None:
        cc_dict = _chat_completion(content="ok")
    config = ToolSandboxResourcesServerConfig(
        name="toolsandbox",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        user_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        **config_overrides,
    )
    return ToolSandboxResourcesServer(config=config, server_client=_mock_server_client(cc_dict))


def _end_conversation_tool_call() -> list:
    return [
        {
            "id": "call_end",
            "type": "function",
            "function": {"name": "end_conversation", "arguments": "{}"},
        }
    ]


def _verify_request(env_id: str) -> ToolSandboxVerifyRequest:
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    response = ToolSandboxNeMoGymResponse(
        id="r",
        created_at=0.0,
        model="policy_model",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        env_id=env_id,
        group_id="0",
        contains_transitions=False,
    )
    return ToolSandboxVerifyRequest(responses_create_params=params, response=response)


REQ = MagicMock(spec=Request)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario_index_is_sorted_and_nonempty():
    server = _make_server()
    index = server.scenario_index
    assert len(index) > 100
    names = [n for n, _ in index]
    assert names == sorted(names)


@pytest.mark.asyncio
async def test_seed_session_shape():
    server = _make_server()
    resp = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=0))
    assert resp.env_id in server.envs
    assert resp.scenario
    # System prompt + at least one initial user turn addressed to the agent.
    assert resp.obs, "expected initial observations"
    assert any(m.role == "user" for m in resp.obs)
    assert resp.tools, "expected non-empty tools"
    tool = resp.tools[0]
    assert tool["type"] == "function" and tool["name"]


@pytest.mark.asyncio
async def test_seed_session_out_of_range():
    server = _make_server()
    with pytest.raises(ValueError):
        await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=10**9))


@pytest.mark.asyncio
async def test_step_text_routes_to_user_simulator():
    # User simulator replies with content -> becomes a user-role observation.
    server = _make_server(_chat_completion(content="Please send it to Alice."))
    seed = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=0))
    step = await server.step(REQ, ToolSandboxStepRequest(env_id=seed.env_id, text="Sure, who should I contact?"))
    assert step.done is False
    assert len(step.obs) == 1
    assert isinstance(step.obs[0], NeMoGymEasyInputMessage)
    assert step.obs[0].role == "user"
    assert "Alice" in step.obs[0].content


@pytest.mark.asyncio
async def test_step_user_ends_conversation():
    # User simulator calls end_conversation -> episode is done, no agent obs.
    server = _make_server(_chat_completion(tool_calls=_end_conversation_tool_call()))
    seed = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=0))
    step = await server.step(REQ, ToolSandboxStepRequest(env_id=seed.env_id, text="All done, bye!"))
    assert step.done is True
    assert step.obs == []


@pytest.mark.asyncio
async def test_step_tool_call_produces_function_output():
    server = _make_server()
    seed = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=0))
    tool_name = seed.tools[0]["name"]
    step = await server.step(
        REQ,
        ToolSandboxStepRequest(
            env_id=seed.env_id,
            function_calls=[NeMoGymResponseFunctionToolCall(call_id="call_1", name=tool_name, arguments="{}")],
        ),
    )
    # Executing a tool returns control to the agent with a tool output obs.
    assert step.done is False
    assert len(step.obs) == 1
    assert isinstance(step.obs[0], NeMoGymFunctionCallOutput)
    assert step.obs[0].call_id == "call_1"


@pytest.mark.asyncio
async def test_step_unknown_tool_returns_error_obs():
    server = _make_server()
    seed = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=0))
    step = await server.step(
        REQ,
        ToolSandboxStepRequest(
            env_id=seed.env_id,
            function_calls=[
                NeMoGymResponseFunctionToolCall(call_id="call_x", name="this_tool_does_not_exist", arguments="{}")
            ],
        ),
    )
    assert step.done is False
    assert len(step.obs) == 1
    assert isinstance(step.obs[0], NeMoGymFunctionCallOutput)
    assert "Error" in step.obs[0].output


@pytest.mark.asyncio
async def test_step_malformed_json_args_returns_error_obs():
    server = _make_server()
    seed = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=0))
    tool_name = seed.tools[0]["name"]
    step = await server.step(
        REQ,
        ToolSandboxStepRequest(
            env_id=seed.env_id,
            function_calls=[
                NeMoGymResponseFunctionToolCall(call_id="call_bad", name=tool_name, arguments="{not json")
            ],
        ),
    )
    assert step.done is False
    assert isinstance(step.obs[0], NeMoGymFunctionCallOutput)
    assert "Error" in step.obs[0].output


@pytest.mark.asyncio
async def test_step_unknown_env_id_raises():
    server = _make_server()
    with pytest.raises(KeyError):
        await server.step(REQ, ToolSandboxStepRequest(env_id="nope", text="hi"))


@pytest.mark.asyncio
async def test_context_isolation_across_envs():
    # Two independent episodes; a step in one must not affect the other's state.
    server = _make_server(_chat_completion(content="reply"))
    a = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=0))
    b = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=1))
    idx_a_before = server.envs[a.env_id]["ctx"].max_sandbox_message_index
    idx_b_before = server.envs[b.env_id]["ctx"].max_sandbox_message_index

    await server.step(REQ, ToolSandboxStepRequest(env_id=a.env_id, text="hello from A"))

    # env a advanced; env b untouched.
    assert server.envs[a.env_id]["ctx"].max_sandbox_message_index > idx_a_before
    assert server.envs[b.env_id]["ctx"].max_sandbox_message_index == idx_b_before


@pytest.mark.asyncio
async def test_close_caches_reward_and_verify_returns_it():
    server = _make_server(_chat_completion(content="reply"))
    seed = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=0))
    env_id = seed.env_id
    close_resp = await server.close(REQ, ToolSandboxCloseRequest(env_id=env_id))
    assert close_resp.success is True
    assert env_id not in server.envs  # dropped
    assert env_id in server.scoring
    details = server.scoring[env_id]
    reward = details["reward"]
    assert 0.0 <= reward <= 1.0
    # The full milestone/minefield breakdown is cached for scoring_details.
    assert details["scenario"] == seed.scenario
    assert details["similarity"] == reward
    assert 0.0 <= details["milestone_similarity"] <= 1.0
    assert 0.0 <= details["minefield_similarity"] <= 1.0
    assert isinstance(details["turn_count"], int)
    assert isinstance(details["categories"], list)
    assert details["exception_type"] is None

    verify_resp = await server.verify(_verify_request(env_id))
    assert verify_resp.reward == reward
    # The breakdown is surfaced as top-level response fields (=> scoring_details).
    dumped = verify_resp.model_dump()
    assert dumped["scenario"] == seed.scenario
    assert dumped["similarity"] == reward
    assert dumped["milestone_similarity"] == details["milestone_similarity"]
    assert dumped["minefield_similarity"] == details["minefield_similarity"]
    assert dumped["turn_count"] == details["turn_count"]
    assert dumped["categories"] == details["categories"]
    assert dumped["exception_type"] is None
    # reward consumed on verify.
    assert env_id not in server.scoring


@pytest.mark.asyncio
async def test_verify_unknown_env_defaults_to_zero():
    server = _make_server()
    verify_resp = await server.verify(_verify_request("never-seeded"))
    assert verify_resp.reward == 0.0


@pytest.mark.asyncio
async def test_close_unknown_env_is_noop():
    server = _make_server()
    resp = await server.close(REQ, ToolSandboxCloseRequest(env_id="ghost"))
    assert resp.success is True


@pytest.mark.asyncio
async def test_message_cap_forces_done():
    # A tiny message cap means the episode ends (done) almost immediately.
    server = _make_server(_chat_completion(content="reply"), max_messages=1)
    seed = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=0))
    step = await server.step(REQ, ToolSandboxStepRequest(env_id=seed.env_id, text="hi"))
    assert step.done is True


@pytest.mark.asyncio
async def test_test_mode_scenarios_subset():
    server = _make_server(test_mode=True)
    # test_mode restricts to the smoke subset (a handful of scenarios).
    assert 0 < len(server.scenario_index) <= 3


@pytest.mark.asyncio
async def test_setup_webserver_registers_step_and_close():
    server = _make_server()
    app = server.setup_webserver()
    paths = {route.path for route in app.routes}
    assert {"/seed_session", "/verify", "/step", "/close"} <= paths


@pytest.mark.asyncio
async def test_scenarios_subset_config():
    name = "add_contact_with_name_and_phone_number"
    server = _make_server(scenarios=[name, "not_a_real_scenario"])
    index = server.scenario_index
    assert [n for n, _ in index] == [name]


@pytest.mark.asyncio
async def test_close_evaluate_exception_yields_zero_reward():
    server = _make_server()
    seed = await server.seed_session(REQ, ToolSandboxSeedSessionRequest(task_idx=0))
    boom = MagicMock()
    boom.evaluate = MagicMock(side_effect=RuntimeError("kaboom"))
    server.envs[seed.env_id]["evaluation"] = boom
    resp = await server.close(REQ, ToolSandboxCloseRequest(env_id=seed.env_id))
    assert resp.success is True
    details = server.scoring[seed.env_id]
    assert details["reward"] == 0.0
    assert details["exception_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_user_inference_flattens_extra_body(monkeypatch):
    # enable_thinking on a vLLM-style model id => extra_body flattened into body.
    captured = {}

    async def fake_post(server_name, url_path, json):
        captured["json"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = AsyncMock(return_value=_chat_completion(content="hi"))
        return resp

    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(side_effect=fake_post)
    user = _ServerClientUser(
        server_client=client,
        server_name="policy_model",
        model_name="qwen3.5",
        sampling={"temperature": 0.0, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    from openai import NOT_GIVEN

    await user._model_inference([{"role": "user", "content": "hi"}], NOT_GIVEN)
    body = captured["json"]
    assert body["model"] == "qwen3.5"
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "extra_body" not in body
