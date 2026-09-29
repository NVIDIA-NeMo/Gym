# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise OSWorld's producer through the shared collector and health checks."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletion

from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_collection import _attach_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from responses_api_agents.osworld_agent.app import OSWorldRunRequest, _build_messages_model_fn, _build_response


MODEL_REF = ModelServerRef(type="responses_api_models", name="policy_model")


@pytest.fixture
def observed_run():
    body = OSWorldRunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "verifier_metadata": {"task_id": "desktop-task"},
            "_ng_task_index": 7,
            "_ng_rollout_index": 2,
            "_ng_rollout_id": "canary-observability",
        }
    )
    calls = []
    captures = []
    for index in range(3):
        response_id = f"completion-{index}"
        answer = "invalid action" if index == 0 else "pyautogui.click(10, 20)"
        messages = [{"role": "user", "content": "Click the button"}]
        calls.append(
            {
                "parse_attempt": index + 1,
                "prompt_messages": messages,
                "response": {
                    "content": answer,
                    "raw_content": answer,
                    "reasoning_content": "",
                    "observation": {
                        "response_id": response_id,
                        "started_at": 1000.0 + index,
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    },
                },
                "accepted": index == 2,
                "parse_error": None if index == 2 else "invalid action",
                "parsed_actions": [answer] if index == 2 else [],
            }
        )
        captures.append(
            {
                "model_call_id": f"capture-{index}",
                "model_ref": MODEL_REF.model_dump(),
                "response_id": response_id,
                "status_code": 200,
                "finish_reason": "stop",
                "tokens_in": 3,
                "tokens_out": 2,
                "request": {"messages": messages},
                "response": {"choices": [{"message": {"role": "assistant", "content": answer}}]},
            }
        )
    result = {
        "reward": 1.0,
        "score": 1.0,
        "finished": True,
        "evaluation_completed": True,
        "mask_sample": False,
        "steps": [
            {
                "step": 0,
                "model_text": "pyautogui.click(10, 20)",
                "actions": ["pyautogui.click(10, 20)"],
                "reward": 1.0,
                "done": True,
                "info": {"agent": {"model_calls": calls}},
            }
        ],
    }
    return body, result, captures


def _collect_and_check(tmp_path, body, result, captures):
    response = _build_response(body, result, "super", 0.6, 0.95, model_ref=MODEL_REF)
    source = body.model_dump() | {"_ng_rollout_id": body.capture_rollout_id}
    row = source | response.model_dump(mode="json")
    row["ng_model_call_capture"] = {"calls": captures}
    _attach_trajectory_record(source, row)
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return row, run_health_checks(path, workers=1).rollouts[0]


def test_observability_reaches_shared_health_checks_without_changing_rollout(tmp_path, observed_run):
    body, result, captures = observed_run
    before = _build_response(body, result, "super", 0.6, 0.95).model_dump(mode="json")
    row, digest = _collect_and_check(tmp_path, body, result, captures)

    assert digest.verdict == "healthy"
    assert digest.unobserved == []
    assert digest.model_calls == 3
    assert digest.successful_model_calls == 3
    assert digest.transcript_prompt_tokens == digest.capture_prompt_tokens == 9
    assert digest.transcript_completion_tokens == digest.capture_completion_tokens == 6
    trajectory = row["ng_trajectory"]
    assert trajectory["task_id"] == "7"
    assert trajectory["rollout_id"] == "canary-observability"
    assert len(trajectory["turns"]) == 3  # Includes both parser retries.
    assert [turn["timestamp"] for turn in trajectory["turns"]] == [1000.0, 1001.0, 1002.0]
    assert [turn["model_calls"][0]["response_id"] for turn in trajectory["turns"]] == [
        f"completion-{i}" for i in range(3)
    ]
    assert trajectory["invocations"][0]["conversation"][0]["content"] == "Click the button"
    for key in ("reward", "mask_sample", "runtime_eligible", "evaluation_completed", "verifier_metadata"):
        assert row[key] == before[key]
    assert row["response"] | {"usage": None} == before["response"]


@pytest.mark.parametrize("missing", ["response_id", "started_at", "usage"])
def test_missing_evidence_is_not_invented(tmp_path, observed_run, missing):
    body, result, captures = observed_run
    result["steps"][0]["info"]["agent"]["model_calls"][0]["response"]["observation"].pop(missing)
    row, digest = _collect_and_check(tmp_path, body, result, captures)
    assert digest.verdict == "unobserved"
    assert digest.unobserved
    if missing == "usage":
        assert {"code": "model_call_usage_unavailable", "invocation_id": None, "detail": "call:0"} in row[
            "ng_trajectory"
        ]["gaps"]


def test_missing_step_call_record_cannot_pass_health_checks(tmp_path, observed_run):
    body, result, captures = observed_run
    result["steps"].append({"step": 1, "model_text": "", "actions": [], "reward": 0.0, "done": True, "info": {}})
    orphan = copy.deepcopy(captures[0])
    orphan["model_call_id"] = "capture-orphan"
    orphan["response_id"] = "completion-orphan"
    orphan["status_code"] = 500
    captures.append(orphan)
    row, digest = _collect_and_check(tmp_path, body, result, captures)
    assert digest.verdict == "unobserved"
    assert "model_call_failed" in digest.unobserved
    assert "rollout_token_count_mismatch" in digest.unobserved
    assert {gap["code"] for gap in row["ng_trajectory"]["gaps"]} >= {
        "model_calls_unavailable",
        "turn_model_call_scope_incomplete",
    }
    assert row["response"]["usage"] is None


def test_final_multimodal_context_is_preserved_without_mutating_prompts(tmp_path, observed_run):
    body, result, captures = observed_run
    image_url = "data:image/png;base64,aA=="
    calls = result["steps"][0]["info"]["agent"]["model_calls"]
    calls[-1]["prompt_messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Current screen"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        }
    ]
    original = copy.deepcopy(result)
    row, digest = _collect_and_check(tmp_path, body, result, captures)
    assert digest.verdict == "healthy"
    content = row["ng_trajectory"]["invocations"][0]["conversation"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Current screen"}
    assert content[1]["image_url"] == image_url
    assert content[1]["detail"] == "high"
    assert result == original


def test_ambiguous_response_ids_are_reported(tmp_path, observed_run):
    body, result, captures = observed_run
    captures[1]["response_id"] = captures[0]["response_id"]
    _, digest = _collect_and_check(tmp_path, body, result, captures)
    assert digest.verdict == "unhealthy"
    assert "trajectory_capture_mismatch" in {finding.check for finding in digest.findings}


def test_collectors_attempt_identity_is_preserved(tmp_path, observed_run):
    body, result, captures = observed_run
    body = body.model_copy(update={"_ng_attempt_index": 1})
    row, digest = _collect_and_check(tmp_path, body, result, captures)
    assert digest.verdict == "healthy"
    assert row["ng_trajectory"]["rollout_id"] == "canary-observability-a1"
    assert not row["ng_trajectory"]["gaps"]


def test_caller_owned_trajectory_identity_matches_collector(tmp_path, observed_run):
    body, result, captures = observed_run
    body = body.model_copy(
        update={
            "trajectory_identity": {
                "schema_version": 1,
                "group_id": "caller-group",
                "task_id": "desktop-task",
                "rollout_id": "caller-rollout",
                "rollout_index": 2,
                "attempt_index": 0,
            }
        }
    )
    row, digest = _collect_and_check(tmp_path, body, result, captures)
    assert digest.verdict == "healthy"
    assert row["ng_trajectory"]["task_id"] == "desktop-task"
    assert row["ng_trajectory"]["rollout_id"] == "caller-rollout"
    assert not row["ng_trajectory"]["gaps"]


@pytest.mark.parametrize(
    "fault,check",
    [("empty", "agent_turn_hollow"), ("failed", "model_call_failed"), ("tokens", "rollout_token_count_mismatch")],
)
def test_real_findings_are_not_hidden(tmp_path, observed_run, fault, check):
    body, result, captures = observed_run
    response = result["steps"][0]["info"]["agent"]["model_calls"][0]["response"]
    if fault == "empty":
        response["content"] = response["raw_content"] = ""
    elif fault == "failed":
        captures[0]["status_code"] = 500
    else:
        response["observation"]["usage"]["completion_tokens"] += 1
    _, digest = _collect_and_check(tmp_path, body, result, captures)
    assert digest.verdict == "unhealthy"
    assert check in {finding.check for finding in digest.findings}


@pytest.mark.parametrize("logging_enabled", [False, True])
def test_model_wrapper_preserves_real_identity_timing_usage_and_request(monkeypatch, tmp_path, logging_enabled):
    if logging_enabled:
        monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(tmp_path / "model.jsonl"))
    else:
        monkeypatch.delenv("OSWORLD_MODEL_IO_LOG", raising=False)
    completion = ChatCompletion.model_validate(
        {
            "id": "completion-real",
            "created": 100,
            "model": "super",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "action"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
    messages = [{"role": "user", "content": "Click the button"}]
    original_messages = copy.deepcopy(messages)
    with (
        patch("responses_api_agents.osworld_agent.app._build_policy_openai_client", return_value=client),
        patch.object(client.chat.completions, "create", return_value=completion) as create,
    ):
        caller = _build_messages_model_fn(base_url="http://policy/v1", model_name="super", api_key="unused")
        response = caller(messages, {"_nemo_gym_return_message": True, "temperature": 0.6, "max_tokens": 4096})
    assert response["content"] == "action"
    assert response["observation"]["response_id"] == "completion-real"
    assert response["observation"]["started_at"] > 0
    assert response["observation"]["usage"]["total_tokens"] == 5
    assert messages == original_messages
    assert create.call_args.kwargs["messages"] == original_messages
    assert set(create.call_args.kwargs) == {"model", "messages", "max_tokens", "temperature", "timeout"}
