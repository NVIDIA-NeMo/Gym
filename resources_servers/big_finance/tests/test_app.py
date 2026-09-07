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
from unittest.mock import AsyncMock, MagicMock

import pytest
from big_finance_harness.tools import (
    EdgarSearchTool,
    FetchUrlTool,
    FinalAnswerTool,
    PythonExecTool,
    WebSearchTool,
)

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.big_finance.app import (
    BigFinanceResourcesServer,
    BigFinanceResourcesServerConfig,
    BigFinanceVerifyRequest,
    extract_final_answer,
    format_trace,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]


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
    return (
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
        )
        .model_dump_json()
        .encode()
    )


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


@pytest.mark.asyncio
async def test_verify_aggregates_points_and_defaults_reward_to_final_answer() -> None:
    server = _server()
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


@pytest.mark.asyncio
async def test_judge_failure_is_exposed_not_raised() -> None:
    server = _server()
    server.server_client.post = AsyncMock(side_effect=TimeoutError("judge unavailable"))
    result = await server.verify(MagicMock(), _request(_response([])))
    assert result.reward == 0.0
    assert "TimeoutError" in result.judge_error
    assert result.rubric_points_possible == 3
