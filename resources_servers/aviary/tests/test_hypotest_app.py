# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hypotest submission feedback and verifier metadata across the Gym session lifecycle."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aviary.core import Message, Tool
from fastapi import Request
from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvState
from hypotest.env.kernel_server import NBLanguage

from nemo_gym.openai_utils import NeMoGymResponseFunctionToolCall
from nemo_gym.server_utils import ServerClient
from resources_servers.aviary.hypotest_app import HypotestResourcesServer, HypotestServerConfig
from resources_servers.aviary.schemas import (
    AviaryAgentVerifyRequest,
    AviaryCloseRequest,
    AviarySeedSessionRequest,
    AviaryStepRequest,
)


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HypotestResourcesServer:
    problem = {
        "id": "00000000-0000-0000-0000-000000000000",
        "hypothesis": "One plus one is two",
        "protocol": "Compute the sum",
        "answer": True,
        "rubric": "One point for computing two",
        "max_points": 1,
        "input_data_path": "capsule",
    }
    problems = tmp_path / "problems.jsonl"
    problems.write_text(json.dumps(problem) + "\n")
    (tmp_path / "capsule").mkdir()
    rubric = SimpleNamespace(call_single=AsyncMock(return_value=SimpleNamespace(text="<score>1</score>")))
    monkeypatch.setattr("hypotest.dataset_server.LiteLLMModel", lambda **kwargs: rubric)

    async def reset(env: InterpreterEnv) -> tuple[list[Message], list[Tool]]:
        # No kernel is needed to submit an answer and run the actual Hypotest scorer.
        env.state = InterpreterEnvState(
            work_dir=env.work_dir, language=NBLanguage.PYTHON, use_docker=False, use_ray=False
        )
        env.tools = [Tool.from_function(env.submit_answer)]
        return [Message(content=problem["hypothesis"])], env.tools

    monkeypatch.setattr(InterpreterEnv, "reset", reset)
    monkeypatch.setattr(InterpreterEnv, "close", AsyncMock())
    return HypotestResourcesServer(
        config=HypotestServerConfig(
            host="0.0.0.0",
            port=8080,
            name="hypotest",
            entrypoint="hypotest_app.py",
            dataset={
                "problem_jsonl": str(problems),
                "capsule_dir": str(tmp_path),
                "work_dir": str(tmp_path / "work"),
                "use_enroot": False,
                "use_ray": False,
            },
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.parametrize("suppress_feedback", [False, True])
@pytest.mark.parametrize("score", [0, 1])
async def test_submission_feedback_and_metadata(
    server: HypotestResourcesServer, suppress_feedback: bool, score: int
) -> None:
    request = MagicMock(spec=Request)
    seeded = await server.seed_session(
        request, AviarySeedSessionRequest(task_idx=0, suppress_answer_feedback=suppress_feedback)
    )
    env = server.env_id_to_env[seeded.env_id]
    env.rubric_model.call_single.return_value.text = f"<score>{score}</score>"
    result = await server.step(
        request,
        AviaryStepRequest(
            env_id=seeded.env_id,
            action=[
                NeMoGymResponseFunctionToolCall(call_id="submit", name="submit_answer", arguments='{"answer":"2"}')
            ],
        ),
    )
    expected = "Correct answer!" if score else "Incorrect answer."
    assert result.obs[0].output == ("Answer submitted." if suppress_feedback else expected)
    assert result.obs[0].call_id == "submit"
    assert result.done
    assert result.reward == score

    verify_request = AviaryAgentVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "response": {
                "id": "response",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
                "env_id": seeded.env_id,
                "group_id": "0",
                "contains_transitions": False,
            },
            "instance_config": {"existing": "preserved"},
        }
    )
    before_close = await server.verify(request, verify_request)
    assert (await server.close(request, AviaryCloseRequest(env_id=seeded.env_id))).success
    after_close = await server.verify(request, verify_request)
    assert after_close.model_dump() == before_close.model_dump()
    assert after_close.reward == score
    assert after_close.model_extra["rubric_model_parsed_score"] == score
    assert not after_close.mask_sample
    assert seeded.env_id not in server.env_id_to_result_metadata


async def test_feedback_is_isolated_between_sessions(server: HypotestResourcesServer) -> None:
    request = MagicMock(spec=Request)
    suppressed = await server.seed_session(
        request, AviarySeedSessionRequest(task_idx=0, suppress_answer_feedback=True)
    )
    normal = await server.seed_session(request, AviarySeedSessionRequest(task_idx=0))
    assert not server.env_id_to_env[suppressed.env_id].config.include_answer_feedback
    assert server.env_id_to_env[normal.env_id].config.include_answer_feedback
    assert server.config.dataset.include_answer_feedback


async def test_judge_failure_masks_sample_after_close(server: HypotestResourcesServer) -> None:
    request = MagicMock(spec=Request)
    seeded = await server.seed_session(request, AviarySeedSessionRequest(task_idx=0))
    env = server.env_id_to_env[seeded.env_id]
    env.state.rubric_model_failed = True
    env.state.rubric_model_fail_type = "request_error"
    await server.close(request, AviaryCloseRequest(env_id=seeded.env_id))
    body = AviaryAgentVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "response": {
                "id": "response",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
                "env_id": seeded.env_id,
                "group_id": "0",
                "contains_transitions": False,
            },
            "instance_config": {"existing": "preserved"},
        }
    )
    verified = await server.verify(request, body)
    assert verified.mask_sample
    assert verified.failure_kind == "judge_failed"
    assert verified.model_extra["instance_config"] == {
        "existing": "preserved",
        "mask_sample": True,
        "agent_error_kind": "rubric_model",
    }
    assert verified.model_extra["rubric_model_fail_request_error"]
