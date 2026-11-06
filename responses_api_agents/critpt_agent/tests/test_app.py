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
from copy import deepcopy
from http.cookies import SimpleCookie
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from omegaconf import OmegaConf
from pydantic import ValidationError

from benchmarks.critpt.prepare import prepare_row
from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedRejectedError, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import OBSERVABILITY_ENABLED_KEY_NAME
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest, seed_restarts
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


def _request(session: dict | None = None, rollout_id: str = "critpt-rollout") -> Request:
    return Request(
        {
            "type": "http",
            "path": "/v1/responses",
            "headers": [],
            "session": session if session is not None else {},
            "path_params": {"rollout_id": rollout_id},
        }
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
            # A batch of one is graded; fire_after's smoke mode would mask the row without calling the grader.
            batch_size=1,  # pragma: allowlist secret
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
    assert len(submissions) == 1


def _close_body(seed: AgentSeedSessionRequest) -> AgentCloseSessionRequest:
    return AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)


class TestSuffixValidationAndSeeds:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "messages",
        [
            # The raw solve shape every pre-migration row has: [system, user] must be refused, not run as one turn.
            [{"role": "system", "content": "s"}, {"role": "user", "content": "Solve this"}],
            [{"role": "user", "content": "Solve this"}, {"role": "assistant", "content": "answer"}],
            [
                {"role": "user", "content": "Solve this"},
                {"role": "user", "content": "Format this"},
                {"role": "assistant", "content": "answer"},
            ],
        ],
        ids=["system-then-user", "ends-with-assistant", "user-pair-not-last"],
    )
    async def test_input_without_two_trailing_user_messages_is_rejected(self, messages):
        server = _make_server()
        with pytest.raises(HTTPException) as error:
            await server.responses(_request(), Response(), NeMoGymResponseCreateParamsNonStreaming(input=messages))
        assert error.value.status_code == 422
        assert "gym eval prepare" in error.value.detail
        server.server_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_seed_accepts_optional_tool_access_and_rejects_continuation(self):
        server = _make_server()
        optional = {"kind": "direct_http", "name": "tool", "required": False, "base_url": "http://resources:8000"}
        accepted = await server.seed_agent_session(_request(), _seed_body(tool_accesses=[optional]))
        assert accepted.agent_session_id == "critpt-session"
        continuation = _seed_body(continuation={"mode": "resume_harness_session", "observations": ["trajectory"]})
        with pytest.raises(HTTPException) as error:
            await _make_server().seed_agent_session(_request(), continuation)
        assert error.value.status_code == 422
        assert "continuation" in error.value.detail

    @pytest.mark.asyncio
    async def test_close_without_activation_runs_no_inference_and_refuses_a_later_activation(self):
        server = _make_server()
        request = _request()
        seed = _seed_body()
        await server.seed_agent_session(request, seed)
        closed = await server.close_agent_session(request, _close_body(seed))
        assert closed.agent_session_id == seed.agent_session_id
        server.server_client.post.assert_not_called()
        with pytest.raises(HTTPException, match="Unknown or closing"):
            await server.responses(request, Response(), _prepared_params())

    @pytest.mark.asyncio
    async def test_restart_only_agent_tells_the_environment_its_episode_restarts(self):
        assert CritPtAgent.checkpoint_mode == "restart_only"
        agent = _make_server()
        agent.server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
        app: FastAPI = agent.setup_webserver()
        seed = _seed_body().model_dump(mode="json")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://agent", headers={"Authorization": "Bearer t"}
        ) as http:
            seeded = await http.post("/v1/agent_sessions", json=seed)
            status = await http.get("/ng-control/v1/checkpoint/status")
        assert seeded.status_code == 200
        assert seed_restarts(seeded.headers)
        assert status.json()["mode"] == "restart_only"


class TestOutcomeAccounting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "overrides, expected",
        [
            ({"status": "incomplete"}, "incomplete"),
            ({"status": "failed"}, "failed"),
            ({"error": {"code": "server_error", "message": "boom"}}, "failed"),
            ({}, "completed"),
        ],
        ids=["incomplete", "failed-status", "error-field", "completed"],
    )
    async def test_format_outcome_is_recorded(self, overrides, expected):
        server = _make_server()
        request = _request()
        seed = _seed_body()
        await server.seed_agent_session(request, seed)
        turn2 = deepcopy(_MODEL_RESPONSE) | overrides
        server.server_client.post.side_effect = [_mock_post_response(_MODEL_RESPONSE), _mock_post_response(turn2)]
        result = await server.responses(request, Response(), _prepared_params())
        assert result.status == overrides.get("status")
        closed = await server.close_agent_session(request, _close_body(seed))
        assert closed.agent_observations.records[0].status == expected
        assert len(closed.agent_observations.records[0].model_calls) == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "overrides",
        [{"status": "failed"}, {"error": {"code": "server_error", "message": "boom"}}],
        ids=["failed-status-only", "error-field-only"],
    )
    async def test_either_failure_signal_on_solve_stops_before_format(self, overrides):
        server = _make_server()
        server.server_client.post.side_effect = [_mock_post_response(deepcopy(_MODEL_RESPONSE) | overrides)]
        result = await server.responses(_request(), Response(), _prepared_params())
        assert server.server_client.post.call_count == 1
        assert (result.status, result.error is not None) == (overrides.get("status"), "error" in overrides)

    @pytest.mark.asyncio
    async def test_incomplete_solve_still_runs_the_format_call(self):
        server = _make_server()
        truncated = deepcopy(_MODEL_RESPONSE) | {"status": "incomplete"}
        server.server_client.post.side_effect = [_mock_post_response(truncated), _mock_post_response(_MODEL_RESPONSE)]
        result = await server.responses(_request(), Response(), _prepared_params())
        assert server.server_client.post.call_count == 2
        assert result.status is None

    @pytest.mark.asyncio
    async def test_first_call_exception_keeps_the_solve_input_as_evidence(self):
        server = _make_server()
        request = _request()
        seed = _seed_body()
        await server.seed_agent_session(request, seed)
        server.server_client.post.side_effect = ConnectionError("model down")
        params = _prepared_params()
        with pytest.raises(ConnectionError):
            await server.responses(request, Response(), params)
        invocation = (await server.close_agent_session(request, _close_body(seed))).agent_observations.records[0]
        assert (invocation.status, invocation.error_type) == ("failed", "ConnectionError")
        assert invocation.model_calls == []
        assert [message.role for message in invocation.conversation] == ["system", "user"]

    @pytest.mark.asyncio
    async def test_observations_name_the_agent_and_the_model_server(self):
        server = _make_server()
        request = _request()
        seed = _seed_body()
        await server.seed_agent_session(request, seed)
        server.server_client.post.side_effect = [_mock_post_response(_MODEL_RESPONSE)] * 2
        await server.responses(request, Response(), _prepared_params())
        bundle = (await server.close_agent_session(request, _close_body(seed))).agent_observations
        assert bundle.source == "critpt_agent"
        assert {ref.model_ref.name for ref in bundle.records[0].model_calls} == {"policy_model"}


class TestLegacyRunAdaptation:
    @pytest.mark.asyncio
    async def test_adapted_row_sends_the_same_model_requests_as_a_prepared_row(self):
        raw = _make_run_body(problem_id="42", code_template="def solve():\n    return ???").model_dump(
            exclude_unset=True
        )
        sent = {}
        for name, row in {"raw": raw, "prepared": prepare_row(raw)}.items():
            server = _make_server()
            _wire_legacy_run(server, [_MODEL_RESPONSE, _MODEL_RESPONSE])
            await server.run(_request(), CritPtAgentRunRequest.model_validate(row))
            model_calls = [
                c for c in server.server_client.post.call_args_list if c.kwargs["server_name"] == "policy_model"
            ]
            sent[name] = [call.kwargs["json"].model_dump() for call in model_calls]
        assert sent["raw"] == sent["prepared"]
        assert "return ???" in sent["raw"][1]["input"][-1]["content"]
        assert "return ???" not in str(sent["raw"][0]["input"])

    @pytest.mark.asyncio
    async def test_adapter_wraps_a_string_input(self):
        server = _make_server()
        body = CritPtAgentRunRequest.model_validate(
            {"responses_create_params": {"input": "Solve this"}, "problem_id": "5", "code_template": "return ???"}
        )
        _wire_legacy_run(server, [_MODEL_RESPONSE, _MODEL_RESPONSE])
        await server.run(_request(), body)
        model_calls = [
            c for c in server.server_client.post.call_args_list if c.kwargs["server_name"] == "policy_model"
        ]
        assert [(m.role, m.content) for m in model_calls[0].kwargs["json"].input] == [("user", "Solve this")]
        assert [m.role for m in model_calls[1].kwargs["json"].input] == ["user", "assistant", "user"]

    @pytest.mark.asyncio
    async def test_session_cookies_flow_from_seed_to_agent_call_to_verify(self):
        server = _make_server()
        seen = {}

        async def post(*, server_name, url_path, json, cookies):
            seen[server_name, url_path] = dict(cookies)
            if url_path == "/seed_session":
                reply = _mock_post_response({})
                reply.cookies = {"resources_session": "seeded"}
            elif url_path == "/verify":
                reply = _mock_post_response(json | {"reward": 1.0})
            elif server_name == server.config.name:
                reply = _mock_post_response((await server.responses(_request(), Response(), json)).model_dump())
                reply.cookies = {"agent_session": "after-agent"}
            else:
                reply = _mock_post_response(_MODEL_RESPONSE)
            return reply

        server.server_client.post.side_effect = post
        await server.run(_request(), _make_run_body())
        assert seen["critpt_resources_server", "/seed_session"] == {}
        assert seen["critpt_agent", "/v1/responses"] == {"resources_session": "seeded"}
        assert seen["critpt_resources_server", "/verify"] == {"agent_session": "after-agent"}


class TestPreparationContract:
    @pytest.mark.parametrize("code_template", [None, 5])
    def test_missing_or_non_string_code_template_is_rejected(self, code_template):
        row = {"problem_id": "5", "problem": "Solve this", "code_template": code_template}
        with pytest.raises(ValueError, match="string code_template"):
            prepare_row(row)

    def test_solve_input_must_end_with_a_user_message(self):
        row = {
            "problem_id": "5",
            "code_template": "return ???",
            "responses_create_params": {
                "input": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
            },
        }
        with pytest.raises(ValueError, match="must end with a user message"):
            prepare_row(row)

    def test_string_input_becomes_the_solve_message_and_nothing_leaks_into_model_params(self):
        row = {
            "problem_id": "5",
            "code_template": "return ???",
            "reference_code": "SECRET_REFERENCE",
            "responses_create_params": {"input": "Solve this"},
        }
        prepared = prepare_row(row)
        messages = prepared["responses_create_params"]["input"]
        assert [(m["role"], m["content"] == "Solve this") for m in messages[:1]] == [("user", True)]
        assert len(messages) == 2 and "return ???" in messages[1]["content"]
        assert "SECRET_REFERENCE" not in json.dumps(prepared["responses_create_params"])
        assert prepared["reference_code"] == "SECRET_REFERENCE"

    def test_committed_example_formatting_prompts_match_the_turn2_template(self):
        root = Path(__file__).resolve().parents[3]
        template = yaml.safe_load((root / "benchmarks/critpt/prompts/turn2.yaml").read_text())["user"]
        assert str(root / "benchmarks/critpt/prompts/turn2.yaml").endswith(TURN2_PROMPT_FPATH)
        for line in (root / "resources_servers/critpt/data/example.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["responses_create_params"]["input"][-1]["content"] == template.format(
                code_template=row["code_template"]
            )


class TestPrepareScript:
    def test_prepare_writes_both_turns_for_every_downloaded_problem(self, tmp_path, monkeypatch):
        import sys
        import types

        from benchmarks.critpt import prepare as prepare_module

        problems = [
            {
                "problem_id": f"Challenge_{i}_main",
                "problem_description": f"Problem {i}",
                "code_template": f"return {i}",
            }
            for i in (1, 2)
        ]
        fake_datasets = types.SimpleNamespace(load_dataset=lambda name, split: problems)
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
        monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
        monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", tmp_path / "critpt_benchmark.jsonl")

        written = prepare_module.prepare()

        rows = [json.loads(line) for line in written.read_text().splitlines()]
        assert [row["problem_id"] for row in rows] == ["Challenge_1_main", "Challenge_2_main"]
        for index, row in enumerate(rows, start=1):
            messages = row["responses_create_params"]["input"]
            assert [message["role"] for message in messages] == ["system", "user", "user"]
            assert f"Problem {index}" in messages[1]["content"]
            assert f"return {index}" in messages[2]["content"]
            assert row["code_template"] == f"return {index}" and row["uuid"] == row["problem_id"]
            assert prepare_row(row) == row

    def test_agent_config_and_preparation_use_the_same_formatting_template(self):
        root = Path(__file__).resolve().parents[3]
        config = yaml.safe_load((root / "responses_api_agents/critpt_agent/configs/critpt_agent.yaml").read_text())
        configured = config["critpt_agent"]["responses_api_agents"]["critpt_agent"]["turn2_prompt_fpath"]
        assert (root / configured).resolve() == (root / "benchmarks/critpt/prompts/turn2.yaml").resolve()

    def test_cached_prepared_file_is_current_only_when_every_row_has_both_user_messages(self, tmp_path):
        from benchmarks.critpt.prepare import is_prepared_data_current

        root = Path(__file__).resolve().parents[3]
        prepared = (root / "resources_servers/critpt/data/example.jsonl").read_text().splitlines()
        old_row = json.loads(prepared[0])
        old_row["responses_create_params"]["input"] = old_row["responses_create_params"]["input"][:-1]
        flat_row = {"problem_id": "1", "problem": "Solve this", "code_template": "return ???"}

        def cached(*rows: str) -> Path:
            path = tmp_path / f"cached{len(list(tmp_path.iterdir()))}.jsonl"
            path.write_text("\n".join(rows) + "\n")
            return path

        assert is_prepared_data_current(cached(*prepared))
        assert not is_prepared_data_current(cached(*prepared, json.dumps(old_row)))
        assert not is_prepared_data_current(cached(json.dumps(flat_row)))
        assert not is_prepared_data_current(cached(""))
        assert not is_prepared_data_current(cached("not json"))


class TestShippedConfig:
    def test_environment_server_applies_no_episode_bound_beyond_the_verify_timeout(self):
        root = Path(__file__).resolve().parents[3]
        config = yaml.safe_load((root / "resources_servers/critpt/configs/critpt.yaml").read_text())
        server_config = config["critpt_agent_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
        assert "default_episode_timeout_seconds" in server_config
        assert server_config["default_episode_timeout_seconds"] is None
        parsed = SingleAgentTurnLegacyEnvironmentServerConfig.model_validate(
            {
                **server_config,
                "name": "critpt_agent_environment_server",
                "host": "127.0.0.1",
                "port": 1,
                "agent_server": {"type": "responses_api_agents", "name": "critpt_agent"},
            }
        )
        assert parsed.default_episode_timeout_seconds is None


class TestSingleActivation:
    @pytest.mark.asyncio
    async def test_identical_retry_joins_the_activation_and_replays_its_cookies(self):
        server = _make_server()
        request = _request()
        seed = _seed_body()
        await server.seed_agent_session(request, seed)
        first_reply = _mock_post_response(_MODEL_RESPONSE)
        first_reply.cookies = SimpleCookie("model_session=kept")
        server.server_client.post.side_effect = [first_reply, _mock_post_response(_MODEL_RESPONSE)]
        params = _prepared_params()
        responses = [Response(), Response()]
        first, second = await asyncio.gather(
            server.responses(request, responses[0], params),
            server.responses(request, responses[1], params.model_copy()),
        )
        assert first == second and server.server_client.post.call_count == 2
        assert all("model_session=kept" in response.headers["set-cookie"] for response in responses)
        later = await server.responses(request, Response(), params)
        assert later == first and server.server_client.post.call_count == 2
        invocation = (await server.close_agent_session(request, _close_body(seed))).agent_observations.records[0]
        assert len(invocation.model_calls) == 2 and invocation.status == "completed"

    @pytest.mark.asyncio
    async def test_a_different_request_is_refused_without_inference(self):
        server = _make_server()
        request = _request()
        await server.seed_agent_session(request, _seed_body())
        server.server_client.post.side_effect = [_mock_post_response(_MODEL_RESPONSE)] * 2
        params = _prepared_params()
        await server.responses(request, Response(), params)
        with pytest.raises(HTTPException) as error:
            await server.responses(request, Response(), params.model_copy(update={"temperature": 0.5}))
        assert error.value.status_code == 409 and "another activation" in error.value.detail
        assert server.server_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_a_request_routed_under_another_rollout_is_refused_without_inference(self):
        server = _make_server()
        request = _request()
        await server.seed_agent_session(request, _seed_body())
        with pytest.raises(HTTPException) as error:
            await server.responses(
                _request(request.session, rollout_id="another-rollout"), Response(), _prepared_params()
            )
        assert error.value.status_code == 409 and "rollout route" in error.value.detail
        server.server_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_cancels_the_running_activation_and_keeps_its_evidence(self):
        server = _make_server()
        request = _request()
        seed = _seed_body()
        await server.seed_agent_session(request, seed)
        second_call_started = asyncio.Event()
        replies = iter([_mock_post_response(_MODEL_RESPONSE)])

        async def post(**kwargs):
            try:
                return next(replies)
            except StopIteration:
                second_call_started.set()
                await asyncio.Event().wait()

        server.server_client.post.side_effect = post
        activation = asyncio.create_task(server.responses(request, Response(), _prepared_params()))
        await second_call_started.wait()
        closed = await server.close_agent_session(request, _close_body(seed))
        with pytest.raises(HTTPException) as error:
            await activation
        assert error.value.status_code == 409 and "closing" in error.value.detail
        invocation = closed.agent_observations.records[0]
        assert invocation.status == "incomplete" and len(invocation.model_calls) == 1
        assert invocation.duration_ms >= 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("capture_enabled", [False, True])
    async def test_model_calls_carry_the_invocation_id_only_when_capture_is_enabled(self, capture_enabled):
        server = _make_server()
        server.server_client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: capture_enabled}
        server.server_client.post.side_effect = [_mock_post_response(_MODEL_RESPONSE)] * 2
        await server.responses(_request(), Response(), _prepared_params())
        for call in server.server_client.post.call_args_list:
            assert call.kwargs.get("headers") == ({"x-session-id": "root"} if capture_enabled else None)
