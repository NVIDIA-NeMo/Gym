# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3

import pytest
from pydantic import ValidationError

from responses_api_agents.opencode_multimodal_agent.rewards import OpenCodeToolRewardConfig, score_opencode_tools


def call(call_id, name="bash", arguments=None):
    return {
        "type": "function_call",
        "call_id": str(call_id),
        "name": name,
        "arguments": json.dumps(arguments or {"command": f"echo {call_id}"}),
    }


def database(tmp_path, calls, *, statuses=None):
    path = tmp_path / "opencode.db"
    statuses = statuses or {}
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE part (data TEXT)")
        for item in calls:
            status, exit_code = statuses.get(item["call_id"], ("completed", 0))
            part = {
                "type": "tool",
                "callID": item["call_id"],
                "tool": item["name"],
                "state": {"status": status, "metadata": {"exit": exit_code}},
            }
            connection.execute("INSERT INTO part VALUES (?)", (json.dumps(part),))
    return path


@pytest.mark.parametrize(
    "count,bonus", [(0, 0), (1, 0.1), (2, 0.1), (3, 0.075), (4, 0.05), (5, 0.025), (6, 0.01), (8, 0.01)]
)
def test_correct_answer_schedule(tmp_path, count, bonus):
    calls = [call(i) for i in range(count)]
    result = score_opencode_tools(database(tmp_path, calls), calls, [], 1.0, OpenCodeToolRewardConfig())
    assert result.auxiliary_reward == bonus
    assert result.base_reward + result.auxiliary_reward == 1.0 + bonus
    assert result.bash_success_count == count


def test_incorrect_answer_exploration_cap(tmp_path):
    calls = [call(i) for i in range(12)]
    result = score_opencode_tools(database(tmp_path, calls), calls, [], 0.0, OpenCodeToolRewardConfig())
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
    path = database(tmp_path, calls, statuses={"4": ("completed", 1)})
    result = score_opencode_tools(path, calls, [], base_reward, OpenCodeToolRewardConfig())
    assert result.auxiliary_reward == pytest.approx(auxiliary)
    assert result.bash_success_count == 1
    assert result.tool_call_count == 5
    assert result.tool_error_count == 1
    assert result.duplicate_tool_call_count == 1


@pytest.mark.parametrize("name", ["read", "edit", "todowrite", "bash"])
def test_duplicate_is_consecutive_and_failure_takes_precedence(tmp_path, name):
    arguments = {"command": "echo hello"} if name == "bash" else {"value": "same"}
    calls = [call(i, name, arguments) for i in range(3)]
    path = database(tmp_path, calls, statuses={"2": ("error", None)})
    result = score_opencode_tools(path, calls, [], 1.0, OpenCodeToolRewardConfig())
    assert result.duplicate_tool_call_count == 1
    assert result.tool_error_count == 1
    assert result.penalty_reward == pytest.approx(-0.07)
    assert result.bash_success_count == int(name == "bash")


def test_bash_description_and_json_key_order_do_not_evade_duplicate(tmp_path):
    calls = [
        call(0, arguments={"command": "echo hello", "description": "first"}),
        call(1, arguments={"description": "different", "command": "  echo hello\n"}),
    ]
    result = score_opencode_tools(database(tmp_path, calls), calls, [], 1.0, OpenCodeToolRewardConfig())
    assert result.bash_success_count == 1
    assert result.auxiliary_reward == pytest.approx(0.08)


def test_replay_seeds_duplicate_but_is_not_rewarded_and_child_calls_are_excluded(tmp_path):
    prefix = call("replay", arguments={"command": "echo hello"})
    calls = [
        call(0, arguments={"command": "echo hello"}),
        call(1, "read", {"filePath": "a"}),
        call(2, arguments={"command": "echo hello"}),
    ]
    path = database(tmp_path, [prefix, *calls, call("child")])
    result = score_opencode_tools(path, calls, [prefix], 1.0, OpenCodeToolRewardConfig())
    assert result.tool_call_count == 3
    assert result.duplicate_tool_call_count == 1
    assert result.bash_success_count == 1


@pytest.mark.parametrize("status", ["running", "pending"])
def test_unfinished_calls_cannot_silently_earn_rewards(tmp_path, status):
    calls = [call(0)]
    path = database(tmp_path, calls, statuses={"0": (status, None)})
    with pytest.raises(ValueError, match="unfinished"):
        score_opencode_tools(path, calls, [], 1.0, OpenCodeToolRewardConfig())


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        score_opencode_tools(path, [call(0)], [], 1.0, OpenCodeToolRewardConfig())
    assert not path.exists()


def test_shell_timeout_is_failed_not_successful(tmp_path):
    calls = [call(0)]
    path = database(tmp_path, calls, statuses={"0": ("completed", None)})
    result = score_opencode_tools(path, calls, [], 0.0, OpenCodeToolRewardConfig())
    assert result.auxiliary_reward == -0.05
    assert result.bash_success_count == 0


@pytest.mark.parametrize("bonuses", [[], [0.1], [0, -0.1], [0, float("nan")]])
def test_invalid_schedule_is_rejected(bonuses):
    with pytest.raises(ValidationError):
        OpenCodeToolRewardConfig(correct_tool_call_bonuses=bonuses)


def test_malformed_tool_arguments_receive_failure_penalty(tmp_path):
    calls = [call(0) | {"arguments": "{invalid"}]
    path = database(tmp_path, calls, statuses={"0": ("error", None)})
    result = score_opencode_tools(path, calls, [], 1.0, OpenCodeToolRewardConfig())
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
    evidence = score_opencode_tools(database(tmp_path, calls), calls, [], 0.0, config)
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
    path = database(tmp_path, calls, statuses={"2": ("completed", 1), "3": ("error", None)})
    result = score_opencode_tools(path, calls, [], base_reward, config)
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
