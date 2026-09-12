# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.polar.app import (
    PolarAgent,
    PolarAgentConfig,
    PolarAgentRunRequest,
    PolarInferenceArtifacts,
    _body_to_swegym_row,
    _build_response_from_result,
)


def _config(tmp_path: Path) -> PolarAgentConfig:
    return PolarAgentConfig(
        name="polar",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        prorl_root=str(tmp_path),
        out_dir=str(tmp_path / "out"),
        concurrency=1,
    )


def _sample_body() -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[],
        metadata={
            "instance_id": "sqlfluff__sqlfluff-1625",
            "problem_statement": "Fix a SQLFluff rule bug.",
        },
        model="Qwen/Qwen3.5-4B",
        temperature=0.2,
        max_output_tokens=128,
    )


def _sample_result() -> dict:
    return {
        "instance_id": "sqlfluff__sqlfluff-1625",
        "scenario": "tool_multi_turn",
        "rounds": [
            {
                "name": "inspect_issue",
                "response": {},
                "text": "",
                "tool_calls": [
                    {
                        "id": "call_inspect",
                        "function": {
                            "name": "inspect_issue",
                            "arguments": '{"summary":"bug"}',
                        },
                    }
                ],
                "prompt_token_ids": [1, 2],
                "response_token_ids": [3, 4],
                "logprobs_content": [{"logprob": -0.1}, {"logprob": -0.2}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
            {
                "name": "final_answer",
                "response": {},
                "text": "Diagnosis: the rule is too broad.",
                "tool_calls": [],
                "prompt_token_ids": [5],
                "response_token_ids": [6, 7],
                "logprobs_content": [{"logprob": -0.3}, {"logprob": -0.4}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2},
            },
        ],
        "tool_results": {
            "inspect_issue": {"tool": "inspect_issue", "instance_id": "sqlfluff__sqlfluff-1625"},
        },
        "token_summary": {"round_count": 2},
    }


def test_body_to_swegym_row_uses_metadata_problem_statement() -> None:
    row = _body_to_swegym_row(_sample_body())

    assert row["metadata"]["instance_id"] == "sqlfluff__sqlfluff-1625"
    assert row["prompt"] == [{"role": "user", "content": "Fix a SQLFluff rule bug."}]


def test_build_response_from_tool_multi_turn_result(tmp_path: Path) -> None:
    response = _build_response_from_result(
        _sample_result(),
        result_path=tmp_path / "result.json",
        mode="direct",
        model="Qwen/Qwen3.5-4B",
    )

    assert response.output[0].type == "function_call"
    assert response.output[0].name == "inspect_issue"
    assert response.output[1].type == "function_call_output"
    assert response.output[2].type == "message"
    assert response.output[2].content[0].text.startswith("Diagnosis:")
    assert response.usage.total_tokens == 34


async def test_run_executes_inference_once_and_wraps_verify_response(tmp_path: Path, monkeypatch) -> None:
    server = PolarAgent(config=_config(tmp_path), server_client=MagicMock(spec=ServerClient))
    body = _sample_body()
    artifacts = PolarInferenceArtifacts(
        result=_sample_result(),
        result_path=tmp_path / "result.json",
        data_path=tmp_path / "input.jsonl",
        run_dir=tmp_path,
        row={},
        stdout="ok",
        stderr="",
    )

    async def fake_execute(request_body, run_context=None):
        assert request_body == body
        assert run_context == {}
        return artifacts

    monkeypatch.setattr(server, "_execute_inference", fake_execute)

    result = await server.run(PolarAgentRunRequest(responses_create_params=body))

    assert result.reward == 0.0
    assert result.metadata["instance_id"] == "sqlfluff__sqlfluff-1625"
    assert result.response.output[-1].type == "message"
