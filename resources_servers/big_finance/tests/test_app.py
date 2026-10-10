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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from aiohttp import ClientResponseError, ClientTimeout
from big_finance_harness.prompts import SYSTEM_PROMPT
from big_finance_harness.tools import (
    EdgarSearchTool,
    FetchUrlTool,
    FinalAnswerTool,
    PythonExecTool,
    ToolError,
    WebSearchTool,
)
from fastapi.testclient import TestClient
from omegaconf import DictConfig, OmegaConf

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, PermanentEndpointError
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient
from resources_servers.big_finance.app import (
    BigFinanceResourcesServer,
    BigFinanceResourcesServerConfig,
    BigFinanceVerifyRequest,
    extract_final_answer,
    format_trace,
)
from responses_api_agents.finance_agent.app import FinanceAgentVerifyResponse
from responses_api_models.openai_model.app import (
    SimpleModelServer,
    SimpleModelServerConfig,
    UpstreamRetriesExhaustedError,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_FPATH = _REPO_ROOT / "resources_servers/big_finance/configs/big_finance.yaml"
_OPENAI_CONFIG_PATHS = [
    "responses_api_models/openai_model/configs/openai_model.yaml",
    "resources_servers/big_finance/configs/big_finance.yaml",
    "resources_servers/big_finance/configs/openai_model.yaml",
]


def _resolved_config(config_paths: list[str], overrides: dict | None = None) -> DictConfig:
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "config_paths": [str(_REPO_ROOT / path) for path in config_paths],
            "policy_base_url": "https://provider.example/v1",
            "policy_api_key": "test-key",
            "policy_model_name": "policy",
        },
        overrides or {},
    )
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


def _configured_model(config: DictConfig, instance: str) -> SimpleModelServer:
    model_config = OmegaConf.to_container(config[instance].responses_api_models.openai_model, resolve=True)
    return SimpleModelServer(
        config=SimpleModelServerConfig(name=instance, **model_config),
        server_client=MagicMock(spec=ServerClient, global_config_dict=config),
    )


def _provider_error_response(status: int, content: bytes) -> SimpleNamespace:
    request_info = SimpleNamespace(
        url="https://provider.example/v1/responses",
        real_url="https://provider.example/v1/responses",
        method="POST",
        headers={},
    )
    return SimpleNamespace(
        status=status,
        ok=False,
        request_info=request_info,
        content=SimpleNamespace(read=AsyncMock(return_value=content)),
        raise_for_status=MagicMock(side_effect=ClientResponseError(request_info, (), status=status)),
    )


@pytest.mark.parametrize("model_type", ["dummy_model", "vllm_model"])
def test_big_finance_model_copy_without_openai_overlay(model_type: str) -> None:
    paths = [_OPENAI_CONFIG_PATHS[1]]
    if model_type != "dummy_model":
        paths.insert(0, f"responses_api_models/{model_type}/configs/{model_type}.yaml")
    config = _resolved_config(paths)

    assert set(config.big_finance_policy_model.responses_api_models) == {model_type}
    assert config.big_finance_policy_model == config.policy_model
    agent_name = config.get("agent_map", {}).get("big_finance", "big_finance")
    assert config[agent_name].responses_api_agents.finance_agent.model_server.name == "big_finance_policy_model"


@pytest.mark.parametrize("extra_body", [{}, {"reasoning": {"effort": "high"}, "max_output_tokens": 1000}])
def test_big_finance_model_overlay_preserves_mixed_vals_policy(extra_body: dict) -> None:
    vals_paths = [
        _OPENAI_CONFIG_PATHS[0],
        "resources_servers/finance_sec_search/configs/finance_sec_search.yaml",
        "resources_servers/finance_agent_v2/configs/finance_agent_v2.yaml",
    ]
    policy_body = {"service_tier": "auto", "reasoning": {"effort": "low", "summary": "auto"}}
    overrides = {
        "policy_model": {"responses_api_models": {"openai_model": {"extra_body": policy_body}}},
    }
    vals_only = _resolved_config(vals_paths, overrides)
    if extra_body:
        overrides["big_finance_policy_model"] = {"responses_api_models": {"openai_model": {"extra_body": extra_body}}}
    mixed = _resolved_config(vals_paths + _OPENAI_CONFIG_PATHS[1:], overrides)

    assert mixed.policy_model == vals_only.policy_model
    assert mixed.search_judge_model == vals_only.search_judge_model
    for instance, resources_type in (
        ("finance_agent", "finance_sec_search"),
        ("finance_agent_v2", "finance_agent_v2"),
    ):
        agent_name = mixed.get("agent_map", {}).get(instance, instance)
        assert mixed[agent_name].responses_api_agents.finance_agent.model_server.name == "policy_model"
        resources = mixed[f"{resources_type}_resources_server"].resources_servers[resources_type]
        assert resources.retrieval_model_server.name == "policy_model"
        assert resources.judge_model_server.name == "search_judge_model"

    policy = _configured_model(mixed, "policy_model")
    big_finance_policy = _configured_model(mixed, "big_finance_policy_model")
    judge = _configured_model(mixed, "big_finance_judge_model")
    assert policy.config.upstream_retry_policy.max_attempts == 1
    assert policy.config.upstream_max_num_tries is None
    assert policy.config.upstream_request_timeout_seconds is None
    assert big_finance_policy.config.extra_body == OmegaConf.to_container(OmegaConf.merge(policy_body, extra_body))
    assert big_finance_policy.config.openai_base_url == policy.config.openai_base_url
    assert big_finance_policy.config.openai_api_key == policy.config.openai_api_key
    assert big_finance_policy.config.openai_model == policy.config.openai_model
    assert big_finance_policy.config.upstream_retry_policy.max_attempts == 13
    assert judge.config.upstream_retry_policy.max_attempts == 21


@pytest.mark.asyncio
@pytest.mark.parametrize("instance,attempts", [("big_finance_policy_model", 13), ("big_finance_judge_model", 21)])
@pytest.mark.parametrize("failure", ["http_503", "timeout"])
async def test_big_finance_model_retry_budget(
    monkeypatch: pytest.MonkeyPatch, instance: str, attempts: int, failure: str
) -> None:
    server = _configured_model(_resolved_config(_OPENAI_CONFIG_PATHS), instance)
    transport = AsyncMock(return_value=_provider_error_response(503, b"unavailable"))
    if failure == "timeout":
        transport.side_effect = TimeoutError("provider timed out")
    backoff_sleep = AsyncMock()
    inner_sleep = AsyncMock()
    monkeypatch.setattr("nemo_gym.openai_utils.request", transport)
    monkeypatch.setattr("nemo_gym.openai_utils.sleep", inner_sleep)
    monkeypatch.setattr("responses_api_models.openai_model.app.asyncio.sleep", backoff_sleep)

    with pytest.raises(UpstreamRetriesExhaustedError, match=f"after {attempts} attempts") as error:
        await server.responses(NeMoGymResponseCreateParamsNonStreaming(input="hello"))

    assert transport.await_count == attempts
    assert backoff_sleep.await_args_list == [call(0.5)] * (attempts - 1)
    inner_sleep.assert_not_awaited()
    for request_call in transport.await_args_list:
        assert request_call.kwargs["_max_num_tries"] == 1
        assert request_call.kwargs["timeout"] == ClientTimeout(total=1800)
    if failure == "http_503":
        assert error.value.status == 503
        assert error.value.response_content == b"unavailable"
    else:
        assert isinstance(error.value.__cause__, TimeoutError)


@pytest.mark.asyncio
@pytest.mark.parametrize("instance", ["big_finance_policy_model", "big_finance_judge_model"])
@pytest.mark.parametrize("status", [400, 401])
async def test_big_finance_model_terminal_errors(monkeypatch: pytest.MonkeyPatch, instance: str, status: int) -> None:
    server = _configured_model(_resolved_config(_OPENAI_CONFIG_PATHS), instance)
    content = b'{"error":{"code":"invalid_api_key"}}' if status == 401 else b"bad request"
    transport = AsyncMock(return_value=_provider_error_response(status, content))
    backoff_sleep = AsyncMock()
    monkeypatch.setattr("nemo_gym.openai_utils.request", transport)
    monkeypatch.setattr("responses_api_models.openai_model.app.asyncio.sleep", backoff_sleep)

    # A permanent authentication failure also stops later calls to the same endpoint.
    for _ in range(2 if status == 401 else 1):
        with pytest.raises(PermanentEndpointError if status == 401 else ClientResponseError) as error:
            await server.responses(NeMoGymResponseCreateParamsNonStreaming(input="hello"))
        assert error.value.status == status

    transport.assert_awaited_once()
    backoff_sleep.assert_not_awaited()


def _response(output: list[dict]) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="policy",
        object="response",
        output=output,
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=True,
    )


def _assistant_message(text: str) -> dict:
    return {
        "id": "message",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _server(**overrides) -> BigFinanceResourcesServer:
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "big_finance_test",
        "sec_edgar_user_agent": "Test test@example.com",
        "judge_model_server": ModelServerRef(type="responses_api_models", name="judge"),
        "judge_responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=[]),
    }
    values.update(overrides)
    return BigFinanceResourcesServer(
        config=BigFinanceResourcesServerConfig(**values),
        server_client=MagicMock(spec=ServerClient),
    )


def _request(response: NeMoGymResponse) -> BigFinanceVerifyRequest:
    return BigFinanceVerifyRequest(
        id="bf-1",
        query="What was revenue?",
        reference_answer="$10 million",
        rubric=[
            {"text": "Finds revenue.", "points": 2},
            {"text": "Uses the right units.", "points": 1},
        ],
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "What was revenue?"}]
        ),
        response=response,
    )


def _judge_response(text: str) -> bytes:
    payload = json.loads(
        _response(
            [
                {
                    "id": "judge-message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ]
        ).model_dump_json()
    )
    # Reproduce Gym's current internal model-server serialization: OpenAI's
    # ``schema`` alias is emitted as its Python field name ``schema_``.
    payload["text"] = {
        "format": {
            "type": "json_schema",
            "name": "rubric_grading",
            "strict": True,
            "schema_": {"type": "object"},
        }
    }
    return json.dumps(payload).encode()


def test_packaged_tool_surface_matches_pinned_snapshot() -> None:
    spec = json.loads((_REPO_ROOT / "benchmarks/big_finance/upstream_spec.json").read_text(encoding="utf-8"))
    tools = [
        WebSearchTool(),
        EdgarSearchTool(user_agent="Test test@example.com"),
        FetchUrlTool(),
        PythonExecTool(),
        FinalAnswerTool(),
    ]
    actual = [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": False,
        }
        for tool in tools
    ]
    assert [tool.name for tool in tools] == spec["tool_order"]
    assert actual == spec["tools"]
    assert spec["system_prompt"] == SYSTEM_PROMPT


def test_tools_package_dependency_is_commit_pinned() -> None:
    spec = json.loads((_REPO_ROOT / "benchmarks/big_finance/upstream_spec.json").read_text(encoding="utf-8"))
    requirement = next(
        line.strip()
        for line in (_REPO_ROOT / "resources_servers/big_finance/requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("big-finance-harness ")
    )
    url, sha = requirement.rsplit("@", 1)
    assert url.endswith(f"git+{spec['tools_package_repository']}.git")
    assert len(sha) == 40
    assert all(char in "0123456789abcdef" for char in sha)
    assert sha == spec["tools_package_commit_id"]


@pytest.mark.asyncio
async def test_routes_use_upstream_tool_output_and_stateless_reverify() -> None:
    server = _server()
    paths = {route.path for route in server.setup_webserver().routes}
    assert {
        "/web_search",
        "/edgar_search",
        "/fetch_url",
        "/python_exec",
        "/final_answer",
    } <= paths
    handler = server._handler("final_answer")
    response = await handler({"answer": "$10 million"})
    assert response.body.decode() == "$10 million"
    assert await server.get_reverify_mode() == ReverifyMode.STATELESS


@pytest.mark.asyncio
async def test_tool_errors_preserve_upstream_message() -> None:
    server = _server()
    server._tools["web_search"].run = AsyncMock(side_effect=ToolError("query is required"))

    response = await server._handler("web_search")({})

    assert json.loads(response.body) == {"error": "query is required"}


def test_final_answer_and_trace_support_tool_and_prose() -> None:
    response = _response(
        [
            {
                "id": "message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Researching.", "annotations": []}],
            },
            {
                "id": "call",
                "call_id": "call-1",
                "type": "function_call",
                "name": "final_answer",
                "arguments": '{"answer":"$10 million"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "$10 million",
            },
        ]
    )
    assert extract_final_answer(response) == "$10 million"
    trace = format_trace(response)
    assert "assistant: Researching." in trace
    assert "tool_call final_answer" in trace
    assert "tool_result:" in trace

    prose = _response(
        [
            {
                "id": "message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "$11 million", "annotations": []}],
            }
        ]
    )
    assert extract_final_answer(prose) == "$11 million"


def test_truncated_or_failed_terminal_call_does_not_become_final_answer() -> None:
    truncated = _response(
        [
            {
                "id": "call",
                "call_id": "call-1",
                "type": "function_call",
                "name": "final_answer",
                "arguments": '{"answer":"unaccepted"}',
                "status": "completed",
            }
        ]
    )
    truncated.metadata = {"stop_reason": "max_turns"}
    assert extract_final_answer(truncated) is None

    failed = _response(
        [
            {
                "id": "call",
                "call_id": "call-1",
                "type": "function_call",
                "name": "final_answer",
                "arguments": '{"answer":"invalid"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "[ERROR] answer is required",
            },
        ]
    )
    failed.metadata = {"stop_reason": "done_tool"}
    assert extract_final_answer(failed) is None
    assert "tool_result [ERROR]: answer is required" in format_trace(failed)


@pytest.mark.parametrize(
    "boundary",
    [
        {"type": "function_call_output", "call_id": "research", "output": "Revenue was $10 million."},
        {"type": "message", "role": "user", "content": "Continue."},
    ],
    ids=["tool_result", "user_nudge"],
)
@pytest.mark.parametrize(
    "final_texts,expected",
    [([], None), ([""], None), (["Revenue ", "was $10 million."], "Revenue was $10 million."), ([" "], " ")],
    ids=["no_message", "empty_message", "multiple_messages", "whitespace"],
)
def test_assistant_final_answer_stays_in_last_model_turn(
    boundary: dict, final_texts: list[str], expected: str | None
) -> None:
    response = _response(
        [_assistant_message("I will research the company."), boundary]
        + [_assistant_message(text) for text in final_texts]
    )
    response.metadata = {"stop_reason": "assistant_message"}
    original = response.model_dump()

    assert extract_final_answer(response) == expected
    assert response.model_dump() == original


def test_done_tool_answer_keeps_precedence_over_assistant_text() -> None:
    response = _response(
        [
            _assistant_message("Planning, not the answer."),
            {
                "id": "call",
                "call_id": "final",
                "type": "function_call",
                "name": "final_answer",
                "arguments": '{"answer":"$10 million"}',
                "status": "completed",
            },
            {"type": "function_call_output", "call_id": "final", "output": "$10 million"},
        ]
    )
    response.metadata = {"stop_reason": "done_tool"}

    assert extract_final_answer(response) == "$10 million"


@pytest.mark.parametrize(
    "arguments,display",
    [
        (json.dumps({"query": "财" * 600}), '{"query": "' + "财" * 600 + '"}'),
        ('{  \n "query" : "revenue", "year" : 2025 }', '{"query": "revenue", "year": 2025}'),
        ("not-json", '{"_unparsed_arguments": "not-json"}'),
        ("", "{}"),
    ],
    ids=["unicode_before_cap", "json_whitespace", "invalid_json", "empty_arguments"],
)
def test_trace_formats_arguments_without_changing_rollout(arguments: str, display: str) -> None:
    tool_output = ' { "observation" : "原始结果" }\n'
    response = _response(
        [
            {
                "id": "call",
                "call_id": "research",
                "type": "function_call",
                "name": "web_search",
                "arguments": arguments,
                "status": "completed",
            },
            {"type": "function_call_output", "call_id": "research", "output": tool_output},
        ]
    )
    original = response.model_dump()

    assert format_trace(response) == f"=== step 0 ===\ntool_call web_search({display})\ntool_result: {tool_output}"
    assert response.model_dump() == original


@pytest.mark.parametrize(
    "arguments",
    ['{ "value" : ' + "9" * 5000 + " }", '{ "value" : ' + "[" * 10_000 + "0" + "]" * 10_000 + " }"],
    ids=["oversized_integer", "deeply_nested_json"],
)
def test_trace_caps_raw_arguments_when_json_cannot_be_rendered(arguments: str) -> None:
    response = _response(
        [
            {
                "id": "call",
                "call_id": "research",
                "type": "function_call",
                "name": "web_search",
                "arguments": arguments,
                "status": "completed",
            },
            {"type": "function_call_output", "call_id": "research", "output": "unchanged tool result"},
        ]
    )
    original = response.model_dump()

    assert format_trace(response) == (
        f"=== step 0 ===\ntool_call web_search({arguments[:1500]}...)\ntool_result: unchanged tool result"
    )
    assert response.model_dump() == original


@pytest.mark.asyncio
async def test_verify_aggregates_points_and_defaults_reward_to_final_answer() -> None:
    server = _server()
    assert server.config.reward_mode == "final_answer"
    judge = {
        "final_answer_correct": True,
        "rubric": [
            {"index": 1, "satisfied": True, "explanation": "present"},
            {"index": 2, "satisfied": False, "explanation": "missing"},
        ],
    }
    raw = MagicMock(ok=True)
    raw.read = AsyncMock(return_value=_judge_response(json.dumps(judge)))
    server.server_client.post = AsyncMock(return_value=raw)
    response = _response(
        [
            {
                "id": "call",
                "call_id": "call-1",
                "type": "function_call",
                "name": "final_answer",
                "arguments": '{"answer":"$10 million"}',
                "status": "completed",
            }
        ]
    )

    result = await server.verify(MagicMock(), _request(response))
    assert result.reward == 1.0
    assert result.final_answer_correct is True
    assert result.rubric_points_earned == 2
    assert result.rubric_points_possible == 3
    assert result.rubric_points_fraction == pytest.approx(2 / 3)
    assert [v.satisfied for v in result.rubric_verdicts] == [True, False]
    server.server_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_passthrough_skips_judge_and_accepts_generation_only_request() -> None:
    server = _server(
        reward_mode="passthrough",
        judge_model_server=None,
        judge_responses_create_params=None,
    )
    response = _response(
        [
            {
                "id": "call",
                "call_id": "call-1",
                "type": "function_call",
                "name": "final_answer",
                "arguments": '{"answer":"Generated answer"}',
                "status": "completed",
            }
        ]
    )
    request = BigFinanceVerifyRequest(
        id="generation-only",
        query="Research this company.",
        reference_answer="",
        rubric=[],
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Research this company."}]
        ),
        response=response,
    )

    judge = AsyncMock()
    with patch.object(server, "_judge", judge):
        result = await server.verify(MagicMock(), request)

    assert result.reward == 1.0
    assert result.responses_create_params == request.responses_create_params
    assert result.response == response
    assert result.reference_answer == ""
    assert result.final_answer == "Generated answer"
    assert result.final_answer_correct is None
    assert result.rubric_verdicts == []
    assert result.judge_text is None
    assert result.judge_error is None
    judge.assert_not_awaited()
    server.server_client.post.assert_not_called()


def test_yaml_reward_mode_can_select_passthrough() -> None:
    config = OmegaConf.merge(
        OmegaConf.load(_CONFIG_FPATH),
        {"big_finance_reward_mode": "passthrough"},
    )

    assert config.big_finance_resources_server.resources_servers.big_finance.reward_mode == "passthrough"


@pytest.mark.asyncio
async def test_rubric_points_reward_mode_uses_weighted_fraction() -> None:
    server = _server(reward_mode="rubric_points")
    judge = {
        "final_answer_correct": False,
        "rubric": [
            {"index": 1, "satisfied": True, "explanation": "present"},
            {"index": 2, "satisfied": False, "explanation": "missing"},
        ],
    }
    raw = MagicMock(ok=True)
    raw.read = AsyncMock(return_value=_judge_response(json.dumps(judge)))
    server.server_client.post = AsyncMock(return_value=raw)

    result = await server.verify(MagicMock(), _request(_response([])))

    assert result.final_answer_correct is False
    assert result.reward == pytest.approx(2 / 3)
    assert result.rubric_points_fraction == pytest.approx(2 / 3)


@pytest.mark.parametrize("failure", ["timeout", "http_500", "unparseable_text"])
def test_judge_failure_is_masked_through_verify_route(failure: str) -> None:
    server = _server()
    if failure == "timeout":
        server.server_client.post = AsyncMock(side_effect=TimeoutError("judge unavailable"))
        expected_reason = "judge unavailable"
    elif failure == "http_500":
        server.server_client.post = AsyncMock(return_value=_provider_error_response(500, b"judge unavailable"))
        expected_reason = "500"
    else:
        raw = MagicMock(ok=True)
        raw.read = AsyncMock(return_value=_judge_response("I cannot return a verdict."))
        server.server_client.post = AsyncMock(return_value=raw)
        expected_reason = "JSON object"
    request = _request(_response([_assistant_message("Revenue was $10 million.")]))

    response = TestClient(server.setup_webserver()).post("/verify", json=request.model_dump(mode="json"))

    assert response.status_code == 200
    result = response.json()
    assert result["reward"] == 0.0
    assert result["mask_sample"] is True
    assert result["instance_config"]["mask_sample"] is True
    assert result["_ng_failure_class"] == result["failure_kind"] == "judge_failed"
    assert expected_reason in result["failure_reason"]
    assert result["response"] == request.response.model_dump(mode="json")
    forwarded = FinanceAgentVerifyResponse.model_validate(result).model_dump(mode="json")
    for key in ("mask_sample", "failure_kind", "failure_reason", "_ng_failure_class", "response", "instance_config"):
        assert forwarded[key] == result[key]


@pytest.mark.parametrize("final_correct", [True, False])
@pytest.mark.parametrize("judge_format", ["schema_alias", "output_text_only"])
def test_judge_valid_verdict_is_measured_despite_unrelated_response_metadata(
    final_correct: bool, judge_format: str
) -> None:
    server = _server()
    grade = json.dumps({"final_answer_correct": final_correct, "rubric": []})
    raw = MagicMock(ok=True)
    raw.read = AsyncMock(
        return_value=(
            _judge_response(grade) if judge_format == "schema_alias" else json.dumps({"output_text": grade}).encode()
        )
    )
    server.server_client.post = AsyncMock(return_value=raw)
    request = _request(_response([_assistant_message("Revenue was $10 million.")]))

    response = TestClient(server.setup_webserver()).post("/verify", json=request.model_dump(mode="json"))

    assert response.status_code == 200
    result = response.json()
    assert result["reward"] == float(final_correct)
    assert result["mask_sample"] is False
    assert result["failure_reason"] is None
    assert result.get("_ng_failure_class") is None
    assert result["response"] == request.response.model_dump(mode="json")


def test_judge_failure_is_excluded_from_aggregate_reward() -> None:
    server = _server()
    request = _request(_response([_assistant_message("Revenue was $10 million.")]))
    raw = MagicMock(ok=True)
    raw.read = AsyncMock(return_value=_judge_response('{"final_answer_correct":true,"rubric":[]}'))
    server.server_client.post = AsyncMock(side_effect=[raw, TimeoutError("judge unavailable")])
    with TestClient(server.setup_webserver()) as client:
        good = client.post("/verify", json=request.model_dump(mode="json")).json()
        failed = client.post("/verify", json=request.model_dump(mode="json")).json()
    rows = [
        {
            **FinanceAgentVerifyResponse.model_validate(row).model_dump(),
            "_ng_task_index": index,
            "_ng_rollout_index": 0,
        }
        for index, row in enumerate((good, failed))
    ]

    mixed = compute_aggregate_metrics(rows)

    assert mixed.agent_metrics["mean/reward"] == 1.0
    assert mixed.agent_metrics["coverage/measured_rollouts"] == 1
    assert mixed.agent_metrics["coverage/masked_rollouts"] == 1
    assert mixed.agent_metrics["coverage/measured_tasks"] == 1
    assert mixed.agent_metrics["coverage/fully_masked_tasks"] == 1

    all_failed = compute_aggregate_metrics(rows[1:])

    assert "mean/reward" not in all_failed.agent_metrics
    assert "mean/reward" not in all_failed.key_metrics
    assert all_failed.group_level_metrics == []
    assert all_failed.agent_metrics["coverage/measured_rollouts"] == 0
    assert all_failed.agent_metrics["coverage/masked_rollouts"] == 1
