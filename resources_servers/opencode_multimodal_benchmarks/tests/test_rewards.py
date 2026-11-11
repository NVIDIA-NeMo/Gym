# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import ValidationError

from resources_servers.opencode_multimodal_benchmarks.rewards import OpenCodeToolRewardConfig, score_opencode_tools


def call(call_id, name="bash", arguments=None):
    return {
        "type": "function_call",
        "call_id": str(call_id),
        "name": name,
        "arguments": json.dumps(arguments or {"command": f"echo {call_id}"}),
    }


def execution_records(calls, *, statuses=None):
    statuses = statuses or {}
    return {item["call_id"]: (item["name"], *statuses.get(item["call_id"], ("completed", 0))) for item in calls}


@pytest.mark.parametrize(
    "count,bonus", [(0, 0), (1, 0.1), (2, 0.1), (3, 0.075), (4, 0.05), (5, 0.025), (6, 0.01), (8, 0.01)]
)
def test_correct_answer_schedule(tmp_path, count, bonus):
    calls = [call(i) for i in range(count)]
    result = score_opencode_tools(execution_records(calls), calls, 1.0, OpenCodeToolRewardConfig())
    assert result.auxiliary_reward == bonus
    assert result.base_reward + result.auxiliary_reward == 1.0 + bonus
    assert result.bash_success_count == count


def test_incorrect_answer_exploration_cap(tmp_path):
    calls = [call(i) for i in range(12)]
    result = score_opencode_tools(execution_records(calls), calls, 0.0, OpenCodeToolRewardConfig())
    assert result.auxiliary_reward == pytest.approx(0.2)


@pytest.mark.parametrize("base_reward,auxiliary", [(0.0, -0.05), (1.0, 0.03)])
def test_only_bash_earns_bonus_but_all_tools_have_penalties(tmp_path, base_reward, auxiliary):
    calls = [
        call(0, "read", {"filePath": "a.png"}),
        call(1, "read", {"filePath": "a.png"}),
        call(2, "edit", {"filePath": "x.py", "oldString": "a", "newString": "b"}),
        call(3),
        call(4),
    ]
    path = execution_records(calls, statuses={"4": ("completed", 1)})
    result = score_opencode_tools(path, calls, base_reward, OpenCodeToolRewardConfig())
    assert result.auxiliary_reward == pytest.approx(auxiliary)
    assert result.bash_success_count == 1
    assert result.tool_call_count == 5
    assert result.tool_error_count == 1
    assert result.duplicate_tool_call_count == 1


@pytest.mark.parametrize("name", ["read", "edit", "todowrite", "bash"])
def test_duplicate_is_consecutive_and_failure_takes_precedence(tmp_path, name):
    arguments = {"command": "echo hello"} if name == "bash" else {"value": "same"}
    calls = [call(i, name, arguments) for i in range(3)]
    path = execution_records(calls, statuses={"2": ("error", None)})
    result = score_opencode_tools(path, calls, 1.0, OpenCodeToolRewardConfig())
    assert result.duplicate_tool_call_count == 1
    assert result.tool_error_count == 1
    assert result.penalty_reward == pytest.approx(-0.07)
    assert result.bash_success_count == int(name == "bash")


def test_bash_description_and_json_key_order_do_not_evade_duplicate(tmp_path):
    calls = [
        call(0, arguments={"command": "echo hello", "description": "first"}),
        call(1, arguments={"description": "different", "command": "  echo hello\n"}),
    ]
    result = score_opencode_tools(execution_records(calls), calls, 1.0, OpenCodeToolRewardConfig())
    assert result.bash_success_count == 1
    assert result.auxiliary_reward == pytest.approx(0.08)


def test_only_calls_in_main_trajectory_are_rewarded(tmp_path):
    calls = [
        call(0, arguments={"command": "echo hello"}),
        call(1, "read", {"filePath": "a"}),
        call(2, arguments={"command": "echo hello"}),
    ]
    path = execution_records([*calls, call("child")])
    result = score_opencode_tools(path, calls, 1.0, OpenCodeToolRewardConfig())
    assert result.tool_call_count == 3
    assert result.duplicate_tool_call_count == 0
    assert result.bash_success_count == 2


@pytest.mark.parametrize("status", ["running", "pending"])
def test_unfinished_calls_cannot_silently_earn_rewards(tmp_path, status):
    calls = [call(0)]
    path = execution_records(calls, statuses={"0": (status, None)})
    with pytest.raises(ValueError, match="unfinished"):
        score_opencode_tools(path, calls, 1.0, OpenCodeToolRewardConfig())


def test_missing_execution_records_fail_closed():
    with pytest.raises(ValueError, match="Missing or unfinished"):
        score_opencode_tools({}, [call(0)], 1.0, OpenCodeToolRewardConfig())


def test_shell_timeout_is_failed_not_successful(tmp_path):
    calls = [call(0)]
    path = execution_records(calls, statuses={"0": ("completed", None)})
    result = score_opencode_tools(path, calls, 0.0, OpenCodeToolRewardConfig())
    assert result.auxiliary_reward == -0.05
    assert result.bash_success_count == 0


@pytest.mark.parametrize("bonuses", [[], [0.1], [0, -0.1], [0, float("nan")]])
def test_invalid_schedule_is_rejected(bonuses):
    with pytest.raises(ValidationError):
        OpenCodeToolRewardConfig(correct_tool_call_bonuses=bonuses)


def test_malformed_tool_arguments_receive_failure_penalty(tmp_path):
    calls = [call(0) | {"arguments": "{invalid"}]
    path = execution_records(calls, statuses={"0": ("error", None)})
    result = score_opencode_tools(path, calls, 1.0, OpenCodeToolRewardConfig())
    assert result.auxiliary_reward == -0.05


@pytest.mark.parametrize("base_reward", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 6, 8])
def test_correct_only_bash_attempt_schedule(tmp_path, base_reward, count):
    calls = [call(i) for i in range(count)]
    config = OpenCodeToolRewardConfig(
        correct_only=True,
        correct_tool_call_bonuses=(0.0,),
        correct_bash_call_penalties=(0.0, 0.0, 0.0, -0.025, -0.05, -0.075, -0.10),
        tool_success_reward=0.0,
    )
    evidence = score_opencode_tools(execution_records(calls), calls, 0.0, config)
    result = type(evidence).model_validate_json(evidence.model_dump_json()).for_answer(base_reward, config)
    expected = -min(max(count - 2, 0) * 0.025, 0.10) if base_reward == 1.0 else 0.0
    assert result.auxiliary_reward == pytest.approx(expected)
    assert result.bash_call_count == count


@pytest.mark.parametrize("base_reward,expected", [(0.0, 0.0), (1.0, -0.095)])
def test_correct_only_counts_failed_and_duplicate_bash_attempts(tmp_path, base_reward, expected):
    calls = [
        call(0, arguments={"command": "echo same"}),
        call(1, arguments={"command": "echo same"}),
        call(2, arguments={"command": "python3 -c 'raise RuntimeError()'"}),
        call(3, "read", {"filePath": "missing"}),
    ]
    config = OpenCodeToolRewardConfig(
        correct_only=True,
        correct_tool_call_bonuses=(0.0,),
        correct_bash_call_penalties=(0.0, 0.0, 0.0, -0.025, -0.05, -0.075, -0.10),
        invalid_tool_call_names=("bash",),
        tool_success_reward=0.0,
    )
    path = execution_records(calls, statuses={"2": ("completed", 1), "3": ("error", None)})
    result = score_opencode_tools(path, calls, base_reward, config)
    assert result.auxiliary_reward == pytest.approx(expected)
    assert result.bash_call_count == 3
    assert result.bash_success_count == 1
    assert result.tool_error_count == 2
    assert result.duplicate_tool_call_count == 1
    assert result.penalty_reward == pytest.approx(-0.07)


@pytest.mark.parametrize("penalties", [[], [-0.1], [0, 0.1], [0, float("nan")]])
def test_invalid_penalty_schedule_is_rejected(penalties):
    with pytest.raises(ValidationError):
        OpenCodeToolRewardConfig(correct_bash_call_penalties=penalties)


def trajectory(status="completed", *, child=False):
    from nemo_gym.trajectory import TrajectoryRecord

    return TrajectoryRecord.model_validate(
        {
            "schema_version": "2.0",
            "task_id": "task",
            "capture_key": "rollout",
            "invocations": [
                {"invocation_id": "root"},
                *([{"invocation_id": "child", "parent_invocation_id": "root"}] if child else []),
            ],
            "tool_calls": [
                {
                    "invocation_id": "child" if child else "root",
                    "tool_call_id": "bash-1",
                    "native_tool_call_id": "bash-1",
                    "span_id": "span-bash-1",
                    "tool_name": "bash",
                    "status": status,
                }
            ],
        }
    )


def tool_response(answer="A"):
    from resources_servers.opencode_multimodal_benchmarks.tests.test_app import completion

    response = completion(answer)
    return type(response).model_validate(
        response.model_dump()
        | {
            "output": [
                call("bash-1"),
                {"type": "function_call_output", "call_id": "bash-1", "output": "anything"},
                *response.model_dump()["output"],
            ]
        }
    )


@pytest.mark.parametrize("base,bonus", [(0.0, 0.02), (1.0, 0.1)])
def test_shared_trajectory_preserves_training_schedule(base, bonus):
    from resources_servers.opencode_multimodal_benchmarks.rewards import trajectory_tool_reward

    reward = trajectory_tool_reward(tool_response(), trajectory(), base, OpenCodeToolRewardConfig())
    assert reward.auxiliary_reward == pytest.approx(bonus)
    assert reward.bash_success_count == 1


@pytest.mark.parametrize("status", ["failed", "timeout", "cancelled"])
def test_shared_tool_failure_is_not_inferred_from_model_visible_output(status):
    from resources_servers.opencode_multimodal_benchmarks.rewards import trajectory_tool_reward

    reward = trajectory_tool_reward(tool_response(), trajectory(status), 1, OpenCodeToolRewardConfig())
    assert reward.auxiliary_reward == -0.05 and reward.bash_success_count == 0


def test_child_tool_cannot_supply_missing_parent_evidence():
    from resources_servers.opencode_multimodal_benchmarks.rewards import trajectory_tool_reward

    with pytest.raises(ValueError, match="Missing or unfinished"):
        trajectory_tool_reward(tool_response(), trajectory(child=True), 1, OpenCodeToolRewardConfig())


@pytest.mark.asyncio
@pytest.mark.parametrize("base", [0.0, 1.0])
async def test_training_verifier_preserves_full_response_and_shapes_from_trajectory(base):
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient
    from nemo_gym.trajectory.trajectory_record import grading_transform_source
    from resources_servers.string_match.app import StringMatchVerifyRequest
    from resources_servers.string_match.tool_rewards import ToolRewardConfig, ToolRewardServer

    server = ToolRewardServer(
        config=ToolRewardConfig(name="test", host="127.0.0.1", port=1, entrypoint="tool_rewards.py"),
        server_client=MagicMock(spec=ServerClient),
    )
    response = tool_response("<think>private</think>A")
    body = StringMatchVerifyRequest(
        responses_create_params={"input": []},
        response=response,
        expected_answer="A" if base else "B",
        extraction_mode="full_response",
        ng_trajectory=trajectory(),
    )
    result = await server.verify(body)
    saved = result.model_dump(mode="json")
    assert saved["reward"] == pytest.approx(base + (0.1 if base else 0.02))
    assert saved["response"] == response.model_dump(mode="json")
    assert saved["opencode_tool_reward"]["bash_success_count"] == 1
    assert grading_transform_source(saved) is not None
    assert server.setup_webserver() is not None


@pytest.mark.parametrize("status,bonus", [("completed", 0.1), ("failed", -0.05)])
def test_raw_shared_observations_supply_evidence_before_trajectory_projection(status, bonus):
    from nemo_gym.trajectory import AgentObservationBundle
    from resources_servers.opencode_multimodal_benchmarks.rewards import trajectory_tool_reward

    evidence = AgentObservationBundle.model_validate(
        {
            "schema_version": "2.0",
            "source": "opencode",
            "records": [
                {"kind": "agent_invocation", "invocation_id": "root"},
                {
                    "kind": "tool_call",
                    "invocation_id": "root",
                    "tool_call_id": "canonical-call",
                    "native_tool_call_id": "bash-1",
                    "tool_name": "bash",
                    "status": status,
                },
            ],
        }
    )
    scored = trajectory_tool_reward(tool_response(), None, 1, OpenCodeToolRewardConfig(), evidence)
    assert scored.auxiliary_reward == pytest.approx(bonus)


@pytest.mark.asyncio
async def test_masked_base_verifier_result_preserves_original_response_without_shaping(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from nemo_gym.client import ServerClient
    from resources_servers.string_match.app import (
        StringMatchResourcesServer,
        StringMatchVerifyRequest,
        StringMatchVerifyResponse,
    )
    from resources_servers.string_match.tool_rewards import ToolRewardConfig, ToolRewardServer

    body = StringMatchVerifyRequest(
        responses_create_params={"input": []}, response=tool_response(), expected_answer="A"
    )
    base = StringMatchVerifyResponse(**body.model_dump(), reward=0, extracted_answer=None, mask_sample=True)
    monkeypatch.setattr(StringMatchResourcesServer, "verify", AsyncMock(return_value=base))
    server = ToolRewardServer(
        config=ToolRewardConfig(name="test", host="127.0.0.1", port=1, entrypoint="tool_rewards.py"),
        server_client=MagicMock(spec=ServerClient),
    )
    result = await server.verify(body)
    assert result.mask_sample and result.reward == 0
    assert result.response == body.response
    assert "opencode_tool_reward" not in result.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("resource,answer", [("mcqa", "A"), ("math_with_judge", "2")])
@pytest.mark.parametrize("correct", [True, False])
async def test_existing_training_graders_shape_real_scores_and_preserve_tool_evidence(resource, answer, correct):
    import importlib
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient

    if resource == "math_with_judge":
        pytest.importorskip("math_verify")
    entrypoint = importlib.import_module(f"resources_servers.{resource}.tool_rewards")
    request_type = entrypoint.MCQAVerifyRequest if resource == "mcqa" else entrypoint.LibraryJudgeMathVerifyRequest
    config = {"name": "test", "host": "127.0.0.1", "port": 1, "entrypoint": "tool_rewards.py"}
    if resource == "math_with_judge":
        config.update(
            should_use_judge=False,
            judge_model_server={"type": "responses_api_models", "name": "unused"},
            judge_responses_create_params={"input": []},
        )
    server = entrypoint.ToolRewardServer(
        config=entrypoint.ToolRewardConfig.model_validate(config), server_client=MagicMock(spec=ServerClient)
    )
    response = tool_response("<think>private reasoning</think>\\boxed{" + answer + "}")
    body = request_type.model_validate(
        {
            "responses_create_params": {"input": []},
            "response": response,
            "expected_answer": answer if correct else ("B" if resource == "mcqa" else "3"),
            "question": "Solve the task",
            "options": [{"A": "first"}, {"B": "second"}],
            "ng_trajectory": trajectory(),
        }
    )
    result = await server.verify(body)
    assert result.reward == pytest.approx(1.1 if correct else 0.02)
    saved = result.model_dump(mode="json")
    assert saved["opencode_tool_reward"]["base_reward"] == float(correct)
    assert saved["response"] == response.model_dump(mode="json")
    assert server.setup_webserver() is not None
