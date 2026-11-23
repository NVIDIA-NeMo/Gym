# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import ValidationError

from resources_servers.opencode_multimodal_benchmarks.rewards import OpenCodeToolRewardConfig, score_opencode_tools


def bonus_config():
    """Explicit opt-in to positive bonuses and exploration rewards."""
    return OpenCodeToolRewardConfig(
        correct_tool_call_bonuses=(0.0, 0.1, 0.1, 0.075, 0.05, 0.025, 0.01),
        tool_success_reward=0.02,
        tool_success_reward_cap=0.20,
        correct_only=False,
        correct_bash_call_penalties=(0.0,),
        invalid_tool_call_names=None,
    )


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
    result = score_opencode_tools(execution_records(calls), calls, 1.0, bonus_config())
    assert result.auxiliary_reward == bonus
    assert result.base_reward + result.auxiliary_reward == 1.0 + bonus
    assert result.bash_success_count == count


def test_incorrect_answer_exploration_cap(tmp_path):
    calls = [call(i) for i in range(12)]
    result = score_opencode_tools(execution_records(calls), calls, 0.0, bonus_config())
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
    result = score_opencode_tools(path, calls, base_reward, bonus_config())
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
    result = score_opencode_tools(path, calls, 1.0, bonus_config())
    assert result.duplicate_tool_call_count == 1
    assert result.tool_error_count == 1
    assert result.penalty_reward == pytest.approx(-0.07)
    assert result.bash_success_count == int(name == "bash")


def test_bash_description_and_json_key_order_do_not_evade_duplicate(tmp_path):
    calls = [
        call(0, arguments={"command": "echo hello", "description": "first"}),
        call(1, arguments={"description": "different", "command": "  echo hello\n"}),
    ]
    result = score_opencode_tools(execution_records(calls), calls, 1.0, bonus_config())
    assert result.bash_success_count == 1
    assert result.auxiliary_reward == pytest.approx(0.08)


def test_only_calls_in_main_trajectory_are_rewarded(tmp_path):
    calls = [
        call(0, arguments={"command": "echo hello"}),
        call(1, "read", {"filePath": "a"}),
        call(2, arguments={"command": "echo hello"}),
    ]
    path = execution_records([*calls, call("child")])
    result = score_opencode_tools(path, calls, 1.0, bonus_config())
    assert result.tool_call_count == 3
    assert result.duplicate_tool_call_count == 0
    assert result.bash_success_count == 2


@pytest.mark.parametrize("status", ["running", "pending"])
def test_unfinished_calls_cannot_silently_earn_rewards(tmp_path, status):
    calls = [call(0)]
    path = execution_records(calls, statuses={"0": (status, None)})
    with pytest.raises(ValueError, match="unfinished"):
        score_opencode_tools(path, calls, 1.0, bonus_config())


def test_missing_execution_records_fail_closed():
    with pytest.raises(ValueError, match="Missing or unfinished"):
        score_opencode_tools({}, [call(0)], 1.0, bonus_config())


def test_shell_timeout_is_failed_not_successful(tmp_path):
    calls = [call(0)]
    path = execution_records(calls, statuses={"0": ("completed", None)})
    result = score_opencode_tools(path, calls, 0.0, bonus_config())
    assert result.auxiliary_reward == -0.05
    assert result.bash_success_count == 0


@pytest.mark.parametrize("bonuses", [[], [0.1], [0, -0.1], [0, float("nan")]])
def test_invalid_schedule_is_rejected(bonuses):
    with pytest.raises(ValidationError):
        OpenCodeToolRewardConfig(correct_tool_call_bonuses=bonuses)


def test_malformed_tool_arguments_receive_failure_penalty(tmp_path):
    calls = [call(0) | {"arguments": "{invalid"}]
    path = execution_records(calls, statuses={"0": ("error", None)})
    result = score_opencode_tools(path, calls, 1.0, bonus_config())
    assert result.auxiliary_reward == -0.05


@pytest.mark.parametrize("base_reward", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 6, 8])
def test_correct_only_bash_attempt_schedule(tmp_path, base_reward, count):
    calls = [call(i) for i in range(count)]
    config = OpenCodeToolRewardConfig()
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
    config = OpenCodeToolRewardConfig()
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


def trajectory(status="completed", *, child=False, root=None):
    from nemo_gym.trajectory import TrajectoryRecord

    return TrajectoryRecord.model_validate(
        {
            "schema_version": "2.0",
            "task_id": "task",
            "capture_key": "rollout",
            "invocations": [
                {"invocation_id": "root", **(root or {})},
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


@pytest.mark.parametrize("shell_name", ["bash", "shell", "exec_command"])
def test_default_failure_filter_follows_shell_name(shell_name):
    config = OpenCodeToolRewardConfig(shell_tool_name=shell_name)
    calls = [call(0, name=shell_name)]
    result = score_opencode_tools(execution_records(calls, statuses={"0": ("error", None)}), calls, 1.0, config)
    assert config.invalid_tool_call_names == (shell_name,)
    assert result.auxiliary_reward == pytest.approx(-0.05)
    assert result.tool_error_count == 1
    assert OpenCodeToolRewardConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("names,penalty", [(None, -0.05), ((), 0.0), (("read",), 0.0), (("shell",), -0.05)])
def test_explicit_failure_filter_is_preserved(names, penalty):
    config = OpenCodeToolRewardConfig(shell_tool_name="shell", invalid_tool_call_names=names)
    calls = [call(0, name="shell")]
    result = score_opencode_tools(execution_records(calls, statuses={"0": ("error", None)}), calls, 1.0, config)
    assert config.invalid_tool_call_names == names
    assert result.auxiliary_reward == pytest.approx(penalty)


def test_shell_name_rejects_empty_string():
    with pytest.raises(ValidationError, match="shell_tool_name"):
        OpenCodeToolRewardConfig(shell_tool_name="")


def test_null_shell_name_preserves_main_behavior():
    config = OpenCodeToolRewardConfig(shell_tool_name=None)
    calls = [call(i) for i in range(6)]
    result = score_opencode_tools(execution_records(calls), calls, 1.0, config)
    assert config.effective_shell_tool_names == ()
    assert config.invalid_tool_call_names == ("bash",)
    assert result.bash_call_count == 0
    assert result.auxiliary_reward == 0


def tool_response(answer="A"):
    from resources_servers.opencode_multimodal_benchmarks.tests.response_fixtures import completion

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

    reward = trajectory_tool_reward(tool_response(), trajectory(), base, bonus_config())
    assert reward.auxiliary_reward == pytest.approx(bonus)
    assert reward.bash_success_count == 1


@pytest.mark.parametrize("status", ["failed", "timeout", "cancelled"])
def test_shared_tool_failure_is_not_inferred_from_model_visible_output(status):
    from resources_servers.opencode_multimodal_benchmarks.rewards import trajectory_tool_reward

    reward = trajectory_tool_reward(tool_response(), trajectory(status), 1, bonus_config())
    assert reward.auxiliary_reward == -0.05 and reward.bash_success_count == 0


_ROLLOUT_TIMEOUT = {"status": "incomplete", "error_type": "timeout"}


def test_tool_cut_by_rollout_timeout_is_a_rollout_timeout():
    """The rollout's wall-clock timeout is the model's budget: not missing evidence, but not scored either."""
    from resources_servers.opencode_multimodal_benchmarks.rewards import (
        RolloutTimeoutToolEvidence,
        trajectory_tool_reward,
    )

    with pytest.raises(RolloutTimeoutToolEvidence, match="rollout timeout"):
        trajectory_tool_reward(tool_response(), trajectory("incomplete", root=_ROLLOUT_TIMEOUT), 0, bonus_config())


@pytest.mark.parametrize(
    "status,root",
    [
        ("incomplete", None),  # no recorded deadline: the evidence is just missing
        ("incomplete", {"status": "incomplete", "error_type": "cancelled"}),
        ("incomplete", {"status": "failed", "error_type": "server_error"}),
        ("unknown", _ROLLOUT_TIMEOUT),  # an unknown outcome is not established by the deadline
    ],
)
def test_unfinished_tool_without_a_rollout_timeout_stays_unusable(status, root):
    from resources_servers.opencode_multimodal_benchmarks.rewards import (
        RolloutTimeoutToolEvidence,
        ToolEvidenceError,
        trajectory_tool_reward,
    )

    with pytest.raises(ToolEvidenceError, match="Unfinished") as raised:
        trajectory_tool_reward(tool_response(), trajectory(status, root=root), 0, bonus_config())
    assert not isinstance(raised.value, RolloutTimeoutToolEvidence)


def test_child_tool_cannot_supply_missing_parent_evidence():
    from resources_servers.opencode_multimodal_benchmarks.rewards import trajectory_tool_reward

    with pytest.raises(ValueError, match="Missing or unfinished"):
        trajectory_tool_reward(tool_response(), trajectory(child=True), 1, bonus_config())


@pytest.mark.asyncio
@pytest.mark.parametrize("base", [0.0, 1.0])
async def test_training_verifier_preserves_full_response_and_shapes_from_trajectory(base):
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient
    from resources_servers.string_match_tool_rewards.app import (
        ToolRewardConfig,
        ToolRewardServer,
        ToolRewardVerifyRequest,
    )

    server = ToolRewardServer(
        config=ToolRewardConfig(
            name="test",
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            tool_rewards=bonus_config(),
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    response = tool_response("<think>private</think>A")
    body = ToolRewardVerifyRequest(
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
    assert "ng_grading_transform" not in saved
    assert server.setup_webserver() is not None


@pytest.mark.asyncio
async def test_training_verifier_masks_a_rollout_cut_mid_tool_by_its_timeout_as_an_agent_timeout():
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient
    from resources_servers.string_match_tool_rewards.app import (
        ToolRewardConfig,
        ToolRewardServer,
        ToolRewardVerifyRequest,
    )

    server = ToolRewardServer(
        config=ToolRewardConfig(
            name="test", host="127.0.0.1", port=1, entrypoint="app.py", tool_rewards=bonus_config()
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    body = ToolRewardVerifyRequest(
        responses_create_params={"input": []},
        response=tool_response(),
        expected_answer="B",
        extraction_mode="full_response",
        ng_trajectory=trajectory("incomplete", root=_ROLLOUT_TIMEOUT),
    )
    saved = (await server.verify(body)).model_dump(mode="json")
    assert saved["mask_sample"] is True
    assert saved["reward"] == 0.0
    # The model's own budget, unlike a verifier error, so a trainer can keep it in group statistics.
    assert saved["failure_kind"] == "agent_timeout"


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
    scored = trajectory_tool_reward(tool_response(), None, 1, bonus_config(), evidence)
    assert scored.auxiliary_reward == pytest.approx(bonus)


@pytest.mark.asyncio
async def test_masked_base_verifier_result_preserves_original_response_without_shaping(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from nemo_gym.client import ServerClient
    from resources_servers.string_match.app import (
        StringMatchResourcesServer,
        StringMatchVerifyResponse,
    )
    from resources_servers.string_match_tool_rewards.app import (
        ToolRewardConfig,
        ToolRewardServer,
        ToolRewardVerifyRequest,
    )

    body = ToolRewardVerifyRequest(
        responses_create_params={"input": []}, response=tool_response(), expected_answer="A"
    )
    base = StringMatchVerifyResponse(**body.model_dump(), reward=0, extracted_answer=None, mask_sample=True)
    monkeypatch.setattr(StringMatchResourcesServer, "verify", AsyncMock(return_value=base))
    server = ToolRewardServer(
        config=ToolRewardConfig(
            name="test",
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            tool_rewards=bonus_config(),
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    result = await server.verify(body)
    assert result.mask_sample and result.reward == 0
    assert result.response == body.response
    assert "opencode_tool_reward" not in result.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("resource,answer", [("mcqa", "A"), ("math_with_judge", "2")])
@pytest.mark.parametrize("correct", [True, False])
@pytest.mark.parametrize("split_blocks", [False, True])
async def test_existing_training_graders_shape_real_scores_and_preserve_tool_evidence(
    resource, answer, correct, split_blocks
):
    import importlib
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient

    if resource == "math_with_judge":
        pytest.importorskip("math_verify")
    component = {"mcqa": "mcqa_tool_rewards", "math_with_judge": "math_tool_rewards"}[resource]
    entrypoint = importlib.import_module(f"resources_servers.{component}.app")
    request_type = entrypoint.ToolRewardVerifyRequest
    config = {"name": "test", "host": "127.0.0.1", "port": 1, "entrypoint": "app.py"}
    config["tool_rewards"] = bonus_config().model_dump()
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
    if split_blocks:
        message = response.output[-1]
        text = message.content[0].text
        split = text.index("boxed") + 2
        message.content = [
            message.content[0].model_copy(update={"text": part}) for part in (text[:split], text[split:])
        ]
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


@pytest.mark.parametrize("verdict,base,calls", [("[[A=B]]", 1.0, 2), ("[[A!=B]]", 0.0, 1), ("unclear", 0.0, 1)])
@pytest.mark.parametrize("reply_state", ["completed", "omitted", "incomplete", "failed", "error", "second_incomplete"])
async def test_math_judge_records_final_shaped_grade(verdict, base, calls, reply_state, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from nemo_gym.client import ServerClient
    from nemo_gym.core import rollout_context
    from nemo_gym.server_runtime import observe_verifier
    from nemo_gym.trajectory import grading_result_binding
    from nemo_gym.verification import judge_failsafe
    from resources_servers.math_tool_rewards.app import (
        ToolRewardConfig,
        ToolRewardServer,
        ToolRewardVerifyRequest,
    )

    client = MagicMock(spec=ServerClient)
    replies = []
    for i in range(calls):
        payload = tool_response(verdict).model_dump(mode="json") | {"id": f"judge-{i}"}
        if reply_state == "omitted":
            payload.pop("status", None)
        elif reply_state in {"incomplete", "failed", "completed"}:
            payload["status"] = reply_state
        elif reply_state == "error":
            payload["error"] = {"code": "server_error", "message": "Judge unavailable"}
        elif reply_state == "second_incomplete" and i == 1:
            payload["status"] = "incomplete"
        http = MagicMock(status=200)
        http.json = AsyncMock(return_value=payload)
        http.read = AsyncMock(return_value=json.dumps(payload).encode())
        replies.append(http)
    client.post = AsyncMock(side_effect=replies)
    server = ToolRewardServer(
        config=ToolRewardConfig(
            name="math",
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            tool_rewards=bonus_config(),
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            judge_responses_create_params={"input": []},
        ),
        server_client=client,
    )
    # Force the library to defer to the real inherited judge implementation.
    monkeypatch.setattr(server, "_verify_answer_with_library_async", AsyncMock(return_value=(0.0, "2")))
    body = ToolRewardVerifyRequest(
        responses_create_params={"input": []},
        response=tool_response("\\boxed{2}"),
        expected_answer="3",
        question="Solve this task",
        ng_trajectory=trajectory(),
    )
    with rollout_context("math-judge-tool-rewards"):
        # The /verify route's own chain: judge_failsafe turns a judge failure into a masked row.
        result = await observe_verifier(judge_failsafe(server.verify), enabled=True, verifier_name="math")(body)
    # A judge reply cut off at its own limit, or without a verdict, masks the row (math_with_judge's judge_failsafe
    # path); only a usable verdict is graded.
    if verdict == "unclear" or reply_state == "incomplete" or (reply_state == "second_incomplete" and calls == 2):
        masked = json.loads(result.body)
        assert masked["mask_sample"] is True
        assert (masked["_ng_failure_class"], masked["failure_kind"]) == ("judge_failed", "judge_unparseable")
        return
    saved = result.model_dump(mode="json")
    assert saved["reward"] == pytest.approx(base + (0.1 if base else 0.02))
    assert saved["response"] == body.response.model_dump(mode="json")
    assert "ng_grading_transform" not in saved
    [invocation] = saved["ng_verifier_invocations"]
    [grade] = invocation["grading_operations"]
    assert grade["kind"] == "llm_judge"
    assert grade["result_binding_sha256"] == grading_result_binding(saved)
    unusable = reply_state in {"incomplete", "failed", "error"} or (reply_state == "second_incomplete" and calls == 2)
    assert grade["outcome"] == ("unknown" if verdict == "unclear" or unusable else "completed")
    assert len(grade["dependencies"]) == calls == client.post.await_count


@pytest.mark.parametrize("base", [0.0, 1.0])
@pytest.mark.parametrize("profile", ["default", "all_tools"])
def test_websearch_rate_limit_retries_have_no_tool_penalty(base, profile):
    calls = [call(i, "websearch", {"query": "same search"}) for i in range(2)]
    outputs = [
        {
            "type": "function_call_output",
            "call_id": str(i),
            "output": "StatusCode: non 2xx status code (429 POST https://mcp.exa.ai/mcp)",
        }
        for i in range(2)
    ]
    records = execution_records(calls, statuses={"0": ("error", None), "1": ("error", None)})
    config = OpenCodeToolRewardConfig() if profile == "default" else bonus_config()
    first = score_opencode_tools(records, calls[:1] + outputs[:1], base, config)
    assert first.tool_error_count == 1
    assert first.rate_limited_search_count == 1
    assert first.bash_success_count == 0
    assert first.auxiliary_reward == 0
    repeated = score_opencode_tools(records, calls + outputs, base, config)
    assert repeated.tool_error_count == 2
    assert repeated.rate_limited_search_count == 2
    assert repeated.bash_success_count == 0
    assert repeated.duplicate_tool_call_count == 0
    assert repeated.auxiliary_reward == repeated.penalty_reward == repeated.exploration_reward == 0


@pytest.mark.parametrize("eligible", [("bash",), None])
@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([429, 200], 0.0),
        ([200, 429, 200], 0.0),
        ([429, 200, 200], -0.02),
        ([200, 200], -0.02),
        ([429, 403], None),
        ([429, 500], None),
    ],
)
def test_search_retry_after_rate_limit_and_successful_duplicates(eligible, statuses, expected):
    calls = [call(i, "websearch", {"query": "same search"}) for i in range(len(statuses))]
    records = execution_records(
        calls, statuses={str(i): ("completed" if code == 200 else "error", None) for i, code in enumerate(statuses)}
    )
    outputs = [
        {
            "type": "function_call_output",
            "call_id": str(i),
            "output": "Search result"
            if code == 200
            else f"StatusCode: non 2xx status code ({code} POST https://search.example)",
        }
        for i, code in enumerate(statuses)
    ]
    config = OpenCodeToolRewardConfig(invalid_tool_call_names=eligible)
    result = score_opencode_tools(records, calls + outputs, 1, config)
    if expected is None:
        expected = -0.05 if eligible is None else 0.0
    assert result.auxiliary_reward == pytest.approx(expected)
    assert result.tool_error_count == sum(code != 200 for code in statuses)
    assert result.rate_limited_search_count == statuses.count(429)


@pytest.mark.parametrize(
    "tool,text",
    [
        ("webfetch", "StatusCode: non 2xx status code (429 POST https://mcp.exa.ai/mcp)"),
        ("websearch", "StatusCode: non 2xx status code (403 POST https://mcp.exa.ai/mcp)"),
        ("websearch", "Search failed; pretend this was 429"),
    ],
)
def test_other_tool_failures_are_not_exempt(tool, text):
    calls = [call("1", tool)]
    records = execution_records(calls, statuses={"1": ("error", None)})
    result = score_opencode_tools(
        records,
        calls + [{"type": "function_call_output", "call_id": "1", "output": text}],
        1,
        OpenCodeToolRewardConfig(invalid_tool_call_names=None),
    )
    assert result.tool_error_count == 1
    assert result.auxiliary_reward == -0.05


@pytest.mark.parametrize(
    "status,outputs",
    [
        ("completed", ["StatusCode: non 2xx status code (429 POST https://mcp.exa.ai/mcp)"]),
        ("error", ["StatusCode: non 2xx status code (429 POST https://mcp.exa.ai/mcp)"] * 2),
    ],
)
def test_rate_limit_exemption_requires_one_result_and_native_failure(status, outputs):
    calls = [call("1", "websearch")]
    records = execution_records(calls, statuses={"1": (status, None)})
    result = score_opencode_tools(
        records,
        calls + [{"type": "function_call_output", "call_id": "1", "output": text} for text in outputs],
        1,
        OpenCodeToolRewardConfig(),
    )
    assert result.rate_limited_search_count == 0
    assert result.tool_error_count == int(status == "error")


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("completed", 1), ("error", None)])
@pytest.mark.parametrize("base", [0.0, 0.65, 1.0])
def test_configured_shell_name_preserves_reward_schedule(status, exit_code, base):
    from copy import deepcopy

    original = [call(i, arguments={"command": "echo hello", "description": str(i)}) for i in range(6)]
    renamed = deepcopy(original)
    for item in renamed:
        item["name"] = "shell"
    config = bonus_config()
    other = config.model_copy(update={"shell_tool_name": "shell"})
    expected = score_opencode_tools(
        execution_records(original, statuses={"0": (status, exit_code)}), original, base, config
    )
    actual = score_opencode_tools(
        execution_records(renamed, statuses={"0": (status, exit_code)}), renamed, base, other
    )
    assert actual == expected


@pytest.mark.parametrize(
    "settings,eligible,penalty",
    [
        ({}, ("shell",), -0.05),
        ({"invalid_tool_call_names": None}, None, -0.05),
        ({"invalid_tool_call_names": ("bash",)}, ("bash",), 0.0),
        ({"invalid_tool_call_names": ()}, (), 0.0),
    ],
)
def test_renamed_shell_failure_filter_and_serialization(settings, eligible, penalty):
    config = OpenCodeToolRewardConfig(shell_tool_name="shell", **settings)
    calls = [call("1", "shell")]
    records = execution_records(calls, statuses={"1": ("error", 7)})
    for parsed in [config, OpenCodeToolRewardConfig.model_validate_json(config.model_dump_json())]:
        assert parsed.invalid_tool_call_names == eligible
        result = score_opencode_tools(records, calls, 1.0, parsed)
        assert result.tool_error_count == 1
        assert result.bash_call_count == 1
        assert result.auxiliary_reward == penalty


@pytest.mark.parametrize("settings", [{}, {"shell_tool_name": None}])
def test_default_failure_filter_retains_bash_without_a_renamed_shell(settings):
    assert OpenCodeToolRewardConfig(**settings).invalid_tool_call_names == ("bash",)


def test_configured_search_name_and_error_pattern_preserve_exemption():
    original = [
        call("1", name="websearch"),
        {
            "type": "function_call_output",
            "call_id": "1",
            "output": "StatusCode: non 2xx status code (429 POST https://search.example)",
        },
    ]
    renamed = [call("1", name="search"), {"type": "function_call_output", "call_id": "1", "output": "HTTP 429"}]
    config = bonus_config()
    expected = score_opencode_tools({"1": ("websearch", "error", None)}, original, 1, config)
    other = config.model_copy(update={"search_tool_name": "search", "rate_limit_error_pattern": r"HTTP 429"})
    actual = score_opencode_tools({"1": ("search", "error", None)}, renamed, 1, other)
    assert actual == expected
    assert actual.rate_limited_search_count == 1
    renamed[-1]["output"] = "HTTP 500"
    assert score_opencode_tools({"1": ("search", "error", None)}, renamed, 1, other).auxiliary_reward == -0.05


def test_invalid_rate_limit_pattern_rejected_at_configuration():
    with pytest.raises(ValidationError, match="rate-limit error pattern"):
        OpenCodeToolRewardConfig(rate_limit_error_pattern="[")


@pytest.mark.parametrize("base,expected", [(0.0, 0.0), (0.5, 0.45), (0.99, 0.891), (1.0, 0.9)])
def test_proportional_shell_penalties_preserve_correctness_order(base, expected):
    calls = [call(i) for i in range(6)]
    config = OpenCodeToolRewardConfig(penalty_mode="proportional")
    result = score_opencode_tools(execution_records(calls), calls, base, config)
    assert result.base_reward + result.auxiliary_reward == pytest.approx(expected)
    assert result.bash_call_count == result.bash_success_count == 6
    saved = type(result).model_validate_json(result.model_dump_json())
    assert saved.for_answer(base, config) == result


@pytest.mark.parametrize("correct_only", [True, False])
def test_proportional_penalties_include_failures_duplicates_and_usage(correct_only):
    calls = [call(0, arguments={"command": "same"}), call(1, arguments={"command": "same"}), call(2)]
    config = OpenCodeToolRewardConfig(penalty_mode="proportional", correct_only=correct_only)
    result = score_opencode_tools(execution_records(calls, statuses={"2": ("error", None)}), calls, 0.5, config)
    # Failed call 0.05 + duplicate 0.02 + third shell attempt 0.025; scale the whole cost by 0.5.
    assert result.penalty_reward == pytest.approx(-0.07)
    assert result.auxiliary_reward == pytest.approx(-0.0475)
    assert result.base_reward + result.auxiliary_reward == pytest.approx(0.4525)


@pytest.mark.parametrize("base", [0.0, 0.5, 0.99, 1.0])
def test_proportional_penalty_cannot_exceed_correctness(base):
    calls = [call(i) for i in range(2)]
    config = OpenCodeToolRewardConfig(penalty_mode="proportional", invalid_tool_call_penalty=-0.8)
    result = score_opencode_tools(
        execution_records(calls, statuses={str(i): ("error", None) for i in range(2)}), calls, base, config
    )
    assert result.penalty_reward == pytest.approx(-1.6)
    assert result.auxiliary_reward == -base
    assert result.base_reward + result.auxiliary_reward == 0


@pytest.mark.parametrize("base", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("correct_only", [True, False])
def test_proportional_mode_does_not_scale_or_ungate_positive_bonuses(base, correct_only):
    calls = [call(0)]
    config = bonus_config().model_copy(update={"correct_only": correct_only})
    original = score_opencode_tools(execution_records(calls), calls, base, config)
    proportional = score_opencode_tools(
        execution_records(calls), calls, base, config.model_copy(update={"penalty_mode": "proportional"})
    )
    assert proportional == original


@pytest.mark.parametrize("correct_only,expected", [(True, -0.025), (False, -0.005)])
def test_proportional_mode_separates_exploration_bonuses_from_penalties(correct_only, expected):
    calls = [call(0), call(1)]
    config = bonus_config().model_copy(update={"penalty_mode": "proportional", "correct_only": correct_only})
    result = score_opencode_tools(execution_records(calls, statuses={"1": ("error", None)}), calls, 0.5, config)
    assert result.auxiliary_reward == pytest.approx(expected)


@pytest.mark.parametrize("failure_index", [0, 10, 13])
@pytest.mark.parametrize("correct_only", [True, False])
@pytest.mark.parametrize("base", [0.0, 0.5, 1.0])
def test_proportional_success_bonus_cap_is_independent_of_failure_order(failure_index, correct_only, base):
    calls = [call(i) for i in range(14)]
    config = bonus_config().model_copy(update={"penalty_mode": "proportional", "correct_only": correct_only})
    result = score_opencode_tools(
        execution_records(calls, statuses={str(failure_index): ("error", None)}), calls, base, config
    )
    assert result.bash_success_count == 13
    assert result.tool_error_count == 1
    assert result.success_reward == pytest.approx(0.2)
    assert result.penalty_reward == pytest.approx(-0.05)
    # Correct answers use the existing schedule; other answers retain the correct_only gate.
    bonus = 0.01 if base == 1.0 else (0.0 if correct_only else 0.2)
    assert result.auxiliary_reward == pytest.approx(bonus - base * 0.05)
    saved = type(result).model_validate_json(result.model_dump_json())
    assert saved.for_answer(base, config) == result
    assert saved.for_answer(0.5, config).auxiliary_reward == pytest.approx((0.0 if correct_only else 0.2) - 0.025)


@pytest.mark.parametrize("failure_index,expected", [(0, 0.2), (10, 0.2), (13, 0.15)])
def test_default_mode_retains_existing_exploration_clipping(failure_index, expected):
    calls = [call(i) for i in range(14)]
    result = score_opencode_tools(
        execution_records(calls, statuses={str(failure_index): ("error", None)}), calls, 0.5, bonus_config()
    )
    assert result.exploration_reward == pytest.approx(expected)
    assert result.auxiliary_reward == pytest.approx(expected)


def test_proportional_mode_preserves_negative_verifier_scores():
    calls = [call(0)]
    result = score_opencode_tools(
        execution_records(calls, statuses={"0": ("error", None)}),
        calls,
        -1.0,
        OpenCodeToolRewardConfig(penalty_mode="proportional"),
    )
    assert result.base_reward == -1.0 and result.auxiliary_reward == 0.0


def test_penalty_mode_requires_explicit_valid_opt_in():
    assert OpenCodeToolRewardConfig().penalty_mode == "default"
    with pytest.raises(ValidationError, match="penalty_mode"):
        OpenCodeToolRewardConfig(penalty_mode="proportonal")


@pytest.mark.parametrize("names", [[], [""], ["bash", ""], [None], "bash"])
def test_shell_names_reject_invalid_lists(names):
    with pytest.raises(ValidationError, match="shell_tool_names"):
        OpenCodeToolRewardConfig(shell_tool_names=names)


@pytest.mark.parametrize("singular,plural", [("shell", ["terminal"]), ("bash", ["terminal"]), ("shell", ["shell"])])
def test_shell_names_reject_an_explicit_singular_name(singular, plural):
    with pytest.raises(ValidationError, match="Set either shell_tool_name or shell_tool_names, not both"):
        OpenCodeToolRewardConfig(shell_tool_name=singular, shell_tool_names=plural)


def test_shell_names_select_default_failure_filter():
    config = OpenCodeToolRewardConfig(shell_tool_names=["terminal", "exec"])
    calls = [call(0, "terminal"), call(1, "exec"), call(2, "unselected_shell")]
    records = execution_records(calls, statuses={str(i): ("error", None) for i in range(3)})
    result = score_opencode_tools(records, calls, 1.0, config)
    assert config.invalid_tool_call_names == ("terminal", "exec")
    assert result.bash_call_count == 2
    assert result.tool_error_count == 3
    assert result.auxiliary_reward == pytest.approx(-0.10)
    assert OpenCodeToolRewardConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("names", [None, ["shell"]])
def test_null_or_singleton_shell_names_preserve_singular_behavior(names):
    calls = [call(i, "shell") for i in range(4)]
    records = execution_records(calls, statuses={"0": ("completed", 1)})
    singular = OpenCodeToolRewardConfig(shell_tool_name="shell")
    plural = (
        OpenCodeToolRewardConfig(shell_tool_name="shell", shell_tool_names=None)
        if names is None
        else OpenCodeToolRewardConfig(shell_tool_names=names)
    )
    assert score_opencode_tools(records, calls, 1.0, plural) == score_opencode_tools(records, calls, 1.0, singular)


@pytest.mark.parametrize("settings", [{}, {"shell_tool_name": "shell"}, {"shell_tool_names": ["terminal", "exec"]}])
@pytest.mark.parametrize("failure_filter", [{}, {"invalid_tool_call_names": None}, {"invalid_tool_call_names": []}])
def test_shell_selection_survives_config_serialization(settings, failure_filter):
    from resources_servers.string_match_tool_rewards.app import ToolRewardConfig

    config = ToolRewardConfig(
        name="training_string", host="127.0.0.1", port=1, entrypoint="app.py", tool_rewards=settings | failure_filter
    )
    restored = ToolRewardConfig.model_validate_json(config.model_dump_json())
    assert restored == config
    assert restored.tool_rewards.effective_shell_tool_names == config.tool_rewards.effective_shell_tool_names
    assert restored.tool_rewards.invalid_tool_call_names == config.tool_rewards.invalid_tool_call_names
    assert ToolRewardConfig.model_validate(config.model_dump()) == config
    if "shell_tool_names" in settings:
        assert "shell_tool_name" not in config.model_dump()["tool_rewards"]


@pytest.mark.parametrize("name", ["terminal", "exec"])
@pytest.mark.parametrize("exit_code", [None, 0.0])
def test_plural_shell_without_integer_exit_code_is_a_failure(name, exit_code):
    config = OpenCodeToolRewardConfig(shell_tool_names=["terminal", "exec"])
    calls = [call(0, name)]
    # A non-integer zero must not bypass the type check through equality with integer zero.
    result = score_opencode_tools(
        execution_records(calls, statuses={"0": ("completed", exit_code)}), calls, 1.0, config
    )
    assert result.bash_call_count == 1
    assert result.bash_success_count == 0
    assert result.tool_error_count == 1
    assert result.auxiliary_reward == pytest.approx(-0.05)


@pytest.mark.parametrize("base,penalty", [(0.0, 0.0), (1.0, -0.175)])
def test_multiple_shell_names_share_count_schedule_and_all_tool_failures(base, penalty):
    names = ["terminal", "bash", "shell", "exec", "exec_command"]
    config = OpenCodeToolRewardConfig(shell_tool_names=names, invalid_tool_call_names=None)
    calls = [call(i, name) for i, name in enumerate(names)] + [call(5, "read")]
    records = execution_records(calls, statuses={"2": ("completed", 7), "5": ("error", None)})
    result = score_opencode_tools(records, calls, base, config)
    assert config.invalid_tool_call_names is None
    assert result.bash_call_count == 5
    assert result.bash_success_count == 4
    assert result.tool_call_count == 6
    assert result.tool_error_count == 2
    # Five total shell attempts cost .075, plus .05 for each failed shell/read call.
    assert result.auxiliary_reward == pytest.approx(penalty)


@pytest.mark.parametrize("names,penalty", [((), 0.0), (("read",), -0.05), (("terminal", "exec"), -0.10)])
def test_multiple_shell_names_preserve_explicit_failure_filter(names, penalty):
    config = OpenCodeToolRewardConfig(shell_tool_names=["terminal", "exec"], invalid_tool_call_names=names)
    calls = [call(0, "terminal"), call(1, "exec"), call(2, "read")]
    records = execution_records(calls, statuses={str(i): ("error", None) for i in range(3)})
    assert config.invalid_tool_call_names == names
    assert score_opencode_tools(records, calls, 1.0, config).auxiliary_reward == pytest.approx(penalty)


def test_multiple_shell_names_share_bonus_cap_and_keep_duplicate_identity():
    config = OpenCodeToolRewardConfig(
        shell_tool_names=["terminal", "bash"],
        correct_only=False,
        tool_success_reward=0.02,
        tool_success_reward_cap=0.03,
        correct_tool_call_bonuses=(0.0, 0.1, 0.15),
        correct_bash_call_penalties=(0.0,),
    )
    calls = [
        call(0, "bash", {"command": "echo hello"}),
        call(1, "terminal", {"command": "echo hello"}),
        call(2, "terminal", {"command": " echo hello ", "description": "same command"}),
    ]
    records = execution_records(calls)
    exploration = score_opencode_tools(records, calls, 0.0, config)
    correct = score_opencode_tools(records, calls, 1.0, config)
    assert correct.bash_call_count == 3
    assert correct.bash_success_count == 2
    assert correct.duplicate_tool_call_count == 1
    assert exploration.auxiliary_reward == pytest.approx(0.03 - 0.02)
    assert correct.auxiliary_reward == pytest.approx(0.15 - 0.02)
