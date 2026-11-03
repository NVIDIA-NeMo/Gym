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
from copy import deepcopy
from http.cookies import SimpleCookie
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request, Response
from omegaconf import OmegaConf
from pydantic import ValidationError

from benchmarks.critpt.prepare import prepare_row
from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedRejectedError, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.trajectory import NG_AGENT_OBSERVATIONS_KEY
from resources_servers.critpt.app import CritPtResourcesServer, CritPtResourcesServerConfig, CritPtVerifyRequest
from responses_api_agents.critpt_agent.app import (
    CritPtAgent,
    CritPtAgentConfig,
    CritPtAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
    _extract_output_text,
    _strip_thinking_blocks,
)


TURN2_PROMPT_FPATH = "benchmarks/critpt/prompts/turn2.yaml"


_MODEL_RESPONSE = {
    "id": "resp-id",
    "created_at": 1234.5,
    "model": "test-model",
    "object": "response",
    "output": [
        {
            "id": "msg-id",
            "content": [{"annotations": [], "text": "The answer is 1.23", "type": "output_text"}],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ],
    "parallel_tool_calls": False,
    "tool_choice": "none",
    "tools": [],
}

_VERIFY_RESPONSE = {
    "responses_create_params": {"input": []},
    "response": _MODEL_RESPONSE,
    "reward": 1.0,
    "problem_id": "5",
    "accuracy": 1.0,
    "timeout_rate": 0.0,
}

# Row fields the request does not declare; a resources server that grades from the row needs them.
_EXTRA_TASK_FIELDS = {
    "reference_code": "def solve():\n    return 1.23",
    "entry": "solve",
    "testcases": [{"args": [], "expected": 1.23}],
}


def _make_config() -> CritPtAgentConfig:
    return CritPtAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="critpt_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="critpt_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        turn2_prompt_fpath=TURN2_PROMPT_FPATH,
    )


def _make_server() -> CritPtAgent:
    return CritPtAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))


def _mock_post_response(data: dict) -> AsyncMock:
    mock = AsyncMock()
    mock.read.return_value = json.dumps(data).encode()
    mock.cookies = {}
    return mock


def _make_run_body(
    problem_id: str = "5", code_template: str = "def solve():\n    return ???", **extra_fields
) -> CritPtAgentRunRequest:
    return CritPtAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": "You are a physics assistant."},
                {"role": "user", "content": "Solve this."},
            ]
        ),
        problem_id=problem_id,
        code_template=code_template,
        **extra_fields,
    )


class TestHelpers:
    def test_extract_output_text_single_message(self):
        response_json = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello world"}],
                }
            ]
        }
        assert _extract_output_text(response_json) == "Hello world"

    def test_extract_output_text_skips_non_message(self):
        response_json = {
            "output": [
                {"type": "function_call", "content": []},
                {"type": "message", "content": [{"type": "output_text", "text": "Answer"}]},
            ]
        }
        assert _extract_output_text(response_json) == "Answer"

    def test_extract_output_text_empty(self):
        assert _extract_output_text({"output": []}) == ""

    def test_strip_think_block(self):
        assert _strip_thinking_blocks("<think>reasoning here</think>Final Answer: 42") == "Final Answer: 42"

    def test_strip_thinking_variant(self):
        assert _strip_thinking_blocks("<thinking>blah</thinking>answer") == "answer"

    def test_strip_multiline_block(self):
        text = "<think>line 1\nline 2\nline 3</think>Final Answer: 42"
        assert _strip_thinking_blocks(text) == "Final Answer: 42"

    def test_strip_multiple_blocks(self):
        text = "<think>step 1</think>middle\n<think>step 2</think>conclusion"
        assert _strip_thinking_blocks(text) == "middle\nconclusion"

    def test_strip_no_tags_unchanged(self):
        assert _strip_thinking_blocks("Final Answer: 42") == "Final Answer: 42"

    def test_strip_unclosed_left_intact(self):
        text = "<think>partial reasoning"
        assert _strip_thinking_blocks(text) == text

    def test_turn2_template_loaded_from_yaml(self):
        server = _make_server()
        template = "def solve():\n    return ???"
        rendered = server._turn2_user_template.format(code_template=template)
        assert "```python" in rendered
        assert template in rendered
        assert "Populate your final answer" in rendered
        assert "No additional reasoning" in rendered


class TestRunRequest:
    @pytest.mark.parametrize("missing_field", ["problem_id", "code_template"])
    def test_declared_fields_still_required(self, missing_field):
        row = {
            "responses_create_params": {"input": [{"role": "user", "content": "Solve this."}]},
            "problem_id": "5",
            "code_template": "def solve():\n    return ???",
            **_EXTRA_TASK_FIELDS,
        }
        del row[missing_field]

        with pytest.raises(ValidationError) as exc_info:
            CritPtAgentRunRequest.model_validate(row)

        assert [(error["type"], error["loc"]) for error in exc_info.value.errors()] == [("missing", (missing_field,))]


def _request(session: dict | None = None) -> Request:
    return Request(
        {"type": "http", "path": "/v1/responses", "headers": [], "session": session if session is not None else {}}
    )


def _prepared_params() -> NeMoGymResponseCreateParamsNonStreaming:
    row = prepare_row(_make_run_body().model_dump(exclude_unset=True))
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])


def _seed_body(**kwargs) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="critpt-session",
        episode_id={"rollout_id": "critpt-rollout"},
        task_id={"taskset": "critpt", "task_id": "5"},
        **kwargs,
    )


def _wire_legacy_run(server: CritPtAgent, model_responses: list[dict]) -> None:
    responses = iter(model_responses)

    async def post(*, server_name, url_path, json, cookies):
        if url_path == "/seed_session":
            return _mock_post_response({})
        if url_path == "/verify":
            return _mock_post_response(json | {"reward": 1.0})
        if server_name == server.config.name:
            result = await server.responses(_request(), Response(), json)
            return _mock_post_response(result.model_dump())
        return _mock_post_response(next(responses))

    server.server_client.post.side_effect = post


class TestApp:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_turns", [(), (1,), (2,), (1, 2)])
    @pytest.mark.parametrize("detail_tokens", [None, 0, 5])
    async def test_run_reports_both_turns_usage(self, missing_turns, detail_tokens):
        server = _make_server()
        turn1, turn2 = deepcopy(_MODEL_RESPONSE), deepcopy(_MODEL_RESPONSE)
        turn2["output"][0]["content"][0]["text"] = "def solve():\n    return 1.23"
        for turn, response, input_tokens, output_tokens in [(1, turn1, 100, 200), (2, turn2, 350, 20)]:
            response["usage"] = (
                None
                if turn in missing_turns
                else {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "input_tokens_details": {"cached_tokens": detail_tokens},
                    "output_tokens_details": {"reasoning_tokens": detail_tokens},
                }
            )
        _wire_legacy_run(server, [turn1, turn2])
        result = await server.run(_request(), _make_run_body())
        verify_data = server.server_client.post.call_args_list[-1].kwargs["json"]
        assert (
            NeMoGymResponse.model_validate(verify_data["response"]).output
            == NeMoGymResponse.model_validate(turn2).output
        )
        assert _extract_output_text(result.response.model_dump()) == "def solve():\n    return 1.23"
        assert result.reward == 1.0
        if missing_turns:
            assert verify_data["response"]["usage"] is None
            assert result.response.usage is None
        else:
            expected_usage = {
                "input_tokens": 450,
                "output_tokens": 220,
                "total_tokens": 670,
                "input_tokens_details": {"cached_tokens": None if detail_tokens is None else 2 * detail_tokens},
                "output_tokens_details": {"reasoning_tokens": None if detail_tokens is None else 2 * detail_tokens},
            }
            assert verify_data["response"]["usage"] == expected_usage
            assert result.response.usage.model_dump() == expected_usage

    @pytest.mark.asyncio
    @pytest.mark.parametrize("prepared", [False, True])
    async def test_legacy_run_keeps_row_fields_and_calls_two_models(self, prepared):
        server = _make_server()
        row = _make_run_body(problem_id="42", **_EXTRA_TASK_FIELDS).model_dump(exclude_unset=True)
        if prepared:
            row = prepare_row(row)
        body = CritPtAgentRunRequest.model_validate(row)
        _wire_legacy_run(server, [_MODEL_RESPONSE, _MODEL_RESPONSE])
        result = await server.run(_request(), body)
        calls = server.server_client.post.call_args_list
        assert [call.kwargs["server_name"] for call in calls] == [
            "critpt_resources_server",
            "critpt_agent",
            "policy_model",
            "policy_model",
            "critpt_resources_server",
        ]
        for call in [calls[0], calls[-1]]:
            payload = call.kwargs["json"]
            assert {key: payload[key] for key in _EXTRA_TASK_FIELDS} == _EXTRA_TASK_FIELDS
            assert payload["problem_id"] == "42"
        assert len(calls[2].kwargs["json"].input) == 2
        assert len(calls[3].kwargs["json"].input) == 4
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_session_defers_template_and_retains_both_model_calls(self):
        server = _make_server()
        request = _request()
        seed = _seed_body()
        await server.seed_agent_session(request, seed)
        assert await server.seed_agent_session(request, seed) is not None
        turn1, turn2 = deepcopy(_MODEL_RESPONSE), deepcopy(_MODEL_RESPONSE)
        turn1["id"], turn2["id"] = "solve", "format"
        turn1["output"][0]["content"][0]["text"] = "<think>Private reasoning</think>Final Answer: 1.23"
        first_reply = _mock_post_response(turn1)
        first_reply.cookies = SimpleCookie("model_session=kept")
        server.server_client.post.side_effect = [first_reply, _mock_post_response(turn2)]
        params = _prepared_params().model_copy(update={"temperature": 0.25})
        original = params.model_dump()
        response = Response()
        result = await server.responses(request, response, params)
        calls = server.server_client.post.call_args_list
        assert len(calls) == 2
        assert "Populate" not in str(calls[0].kwargs["json"].input)
        assert [m.role for m in calls[1].kwargs["json"].input] == ["system", "user", "assistant", "user"]
        assert calls[1].kwargs["json"].input[-2].content == "Final Answer: 1.23"
        assert "return ???" in calls[1].kwargs["json"].input[-1].content
        assert all(call.kwargs["json"].temperature == 0.25 for call in calls)
        assert calls[1].kwargs["cookies"]["model_session"] == "kept"
        assert "model_session=kept" in response.headers["set-cookie"]
        assert params.model_dump() == original
        assert result.id == "format"
        close_body = AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
        closed = await server.close_agent_session(request, close_body)
        invocation = closed.agent_observations.records[0]
        assert [ref.response_id for ref in invocation.model_calls] == ["solve", "format"]
        assert invocation.status == "completed"
        assert invocation.duration_ms >= 0
        assert invocation.conversation[-1].id == "msg-id"
        assert await server.close_agent_session(request, close_body) == closed
        with pytest.raises(HTTPException, match="Unknown or closing"):
            await server.responses(request, Response(), params)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failed_turn", [1, 2])
    async def test_model_failure_preserves_partial_evidence(self, failed_turn):
        server = _make_server()
        request = _request()
        seed = _seed_body()
        await server.seed_agent_session(request, seed)
        failed = deepcopy(_MODEL_RESPONSE)
        failed.update(status="failed", error={"code": "server_error", "message": "generation failed"})
        replies = [_MODEL_RESPONSE, failed] if failed_turn == 2 else [failed]
        server.server_client.post.side_effect = [_mock_post_response(reply) for reply in replies]
        result = await server.responses(request, Response(), _prepared_params())
        assert result.status == "failed"
        assert server.server_client.post.call_count == failed_turn
        closed = await server.close_agent_session(
            request, AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
        )
        invocation = closed.agent_observations.records[0]
        assert invocation.status == "failed"
        assert len(invocation.model_calls) == failed_turn

    @pytest.mark.asyncio
    async def test_second_call_exception_retains_first_call(self):
        server = _make_server()
        request = _request()
        seed = _seed_body()
        await server.seed_agent_session(request, seed)
        server.server_client.post.side_effect = [
            _mock_post_response(_MODEL_RESPONSE),
            TimeoutError("format timed out"),
        ]
        with pytest.raises(TimeoutError, match="format timed out"):
            await server.responses(request, Response(), _prepared_params())
        closed = await server.close_agent_session(
            request, AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
        )
        invocation = closed.agent_observations.records[0]
        assert invocation.status == "failed"
        assert invocation.error_type == "TimeoutError"
        assert len(invocation.model_calls) == 1
        assert invocation.conversation[-1].id == "msg-id"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_value", ["Solve this", [], [{"role": "user", "content": "Solve this"}]])
    async def test_unprepared_input_fails_before_inference(self, input_value):
        server = _make_server()
        with pytest.raises(HTTPException, match="gym eval prepare") as error:
            await server.responses(_request(), Response(), NeMoGymResponseCreateParamsNonStreaming(input=input_value))
        assert error.value.status_code == 422
        server.server_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_sandbox_access(self):
        server = _make_server()
        seed = _seed_body(
            sandbox_access={
                "connection": {"provider_config_ref": "sandbox", "descriptor": {}},
                "workdir": "/workspace",
            }
        )
        with pytest.raises(AgentSeedRejectedError, match="sandbox"):
            await server.seed_agent_session(_request(), seed)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured", [False, True])
    async def test_rejects_required_tool_access(self, configured):
        server = _make_server()
        grant = {"kind": "direct_http", "name": "tool", "required": True, "base_url": "http://resources:8000"}
        if configured:
            server.config = CritPtAgentConfig.model_validate(server.config.model_dump() | {"tool_accesses": [grant]})
        seed = _seed_body(tool_accesses=[] if configured else [grant])
        with pytest.raises(AgentSeedRejectedError, match="required tool"):
            await server.seed_agent_session(_request(), seed)


class TestPreparation:
    def test_raw_row_prepares_both_turns_without_verifier_data(self):
        row = {"problem_id": "5", "problem": "Solve this", "code_template": "return ???", **_EXTRA_TASK_FIELDS}
        prepared = prepare_row(row)
        assert "responses_create_params" not in row
        messages = prepared["responses_create_params"]["input"]
        assert [message["role"] for message in messages] == ["system", "user", "user"]
        assert "Solve this" in str(messages[-2])
        assert "return ???" in str(messages[-1])
        assert "1.23" not in str(messages)
        assert all(prepared[key] == value for key, value in _EXTRA_TASK_FIELDS.items())
        assert prepare_row(prepared) == prepared

    def test_existing_messages_and_parameters_are_preserved(self):
        row = _make_run_body().model_dump(exclude_unset=True)
        row["responses_create_params"]["temperature"] = 0.5
        before = deepcopy(row)
        prepared = prepare_row(row)
        assert row == before
        assert prepared["responses_create_params"]["input"][:-1] == row["responses_create_params"]["input"]
        assert prepared["responses_create_params"]["temperature"] == 0.5

    def test_committed_examples_are_prepared(self):
        path = Path(__file__).resolve().parents[3] / "resources_servers/critpt/data/example.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 5
        for row in rows:
            assert prepare_row(row) == row
            assert row["code_template"] in row["responses_create_params"]["input"][-1]["content"]


@pytest.mark.asyncio
async def test_prepared_row_runs_through_environment_sessions(monkeypatch):
    """Exercise the shipped flat-row episode path with actual agent and verifier methods."""
    agent = _make_server()
    client = agent.server_client
    client.global_config_dict = OmegaConf.create(
        {
            "critpt_agent": {"responses_api_agents": {"critpt_agent": agent.config.model_dump()}},
            "token_id_capture": {"enabled": False},
        }
    )
    client._resolve_base_url.return_value = "http://resources:8000"
    resources = CritPtResourcesServer(
        config=CritPtResourcesServerConfig(
            name="critpt_resources_server",
            host="localhost",
            port=8000,
            entrypoint="app.py",
            api_key="test-key",
            fire_after=1,  # pragma: allowlist secret
        ),
        server_client=client,
    )
    score = AsyncMock(return_value={"accuracy": 0.75, "timeout_rate": 0.0})
    monkeypatch.setattr(CritPtResourcesServer, "_call_aa_with_rotation", score)
    environment = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnLegacyEnvironmentServerConfig(
            name="environment",
            host="localhost",
            port=8002,
            entrypoint="app.py",
            agent_server={"type": "responses_api_agents", "name": "critpt_agent"},
            resources_server={"type": "resources_servers", "name": "critpt_resources_server"},
        ),
        server_client=client,
    )
    agent_request = _request()
    model_responses = []
    for response_id, text in [("solve", "<think>reasoning</think>1.23"), ("format", "```python\nreturn 1.23\n```")]:
        result = deepcopy(_MODEL_RESPONSE)
        result["id"] = response_id
        result["output"][0]["content"][0]["text"] = text
        model_responses.append(result)

    async def post(*, server_name, url_path, json, **kwargs):
        if server_name == "policy_model":
            result = model_responses.pop(0)
        elif url_path == "/seed_session":
            result = await resources.seed_session(ResourcesSeedSessionRequest.model_validate(json))
        elif url_path == "/v1/agent_sessions":
            result = await agent.seed_agent_session(agent_request, AgentSeedSessionRequest.model_validate(json))
        elif url_path.endswith("/v1/responses"):
            agent_request.scope["path"] = url_path
            agent_request.scope["path_params"] = {"rollout_id": "critpt-smoke"}
            result = await agent.responses(agent_request, Response(), json)
        elif url_path == "/v1/agent_sessions/close":
            result = await agent.close_agent_session(agent_request, AgentCloseSessionRequest.model_validate(json))
        elif url_path == "/verify":
            result = await resources.verify(CritPtVerifyRequest.model_validate(json))
        elif url_path == "/close_session":
            result = await resources.close_resources_session(ResourcesCloseSessionRequest.model_validate(json))
        else:
            raise AssertionError(f"Unexpected route: {server_name}{url_path}")
        reply = _mock_post_response(result if isinstance(result, dict) else result.model_dump(mode="json"))
        reply.cookies = SimpleCookie("session=test-cookie")
        reply.headers = {}
        return reply

    client.post.side_effect = post
    row = prepare_row(_make_run_body(problem_id="Challenge_1_main").model_dump(exclude_unset=True))
    result = await environment.run_legacy(row | {"_ng_rollout_id": "critpt-smoke"})
    assert "_ng_failure_class" not in result, result
    assert result["reward"] == 0.75
    assert result["response"]["id"] == "format"
    invocation = result[NG_AGENT_OBSERVATIONS_KEY]["records"][0]
    assert [ref["response_id"] for ref in invocation["model_calls"]] == ["solve", "format"]
    paths = [call.kwargs["url_path"] for call in client.post.call_args_list]
    assert paths.count("/ng-rollout/critpt-smoke/v1/responses") == 3
    assert paths.index("/v1/agent_sessions/close") < paths.index("/verify") < paths.index("/close_session")
    submissions = score.call_args.args[0]
    assert submissions[0]["problem_id"] == "Challenge_1_main"
    assert submissions[0]["generated_code"] == "```python\nreturn 1.23\n```"
    assert len(submissions) == 70
