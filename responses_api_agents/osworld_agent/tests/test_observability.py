# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise OSWorld's producer through the shared collector and health checks."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import APIConnectionError, InternalServerError
from openai.types.chat import ChatCompletion

from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_collection import _attach_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from responses_api_agents.osworld_agent.adapter_agents import NemotronV3NanoOmniAgent
from responses_api_agents.osworld_agent.app import (
    OSWorldRunRequest,
    _build_messages_model_fn,
    _build_response,
    _PolicyUsage,
)


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
                "call_index": index,
                "started_at": 1000.0 + index,
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
        "usage": {
            "input_tokens": 9,
            "output_tokens": 6,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": None},
            "output_tokens_details": {"reasoning_tokens": None},
        },
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
    assert row["response"] == before["response"]


@pytest.mark.parametrize("records_complete", [True, False])
@pytest.mark.parametrize("usage_known", [True, False])
def test_trajectory_preserves_policy_usage(tmp_path, observed_run, records_complete, usage_known):
    body, result, captures = observed_run
    if usage_known:
        # Gym also counts finished responses the adapter rejected before recording a turn.
        result["usage"]["input_tokens"] += 4
        result["usage"]["output_tokens"] += 1
        result["usage"]["total_tokens"] += 5
        result["usage"]["input_tokens_details"]["cached_tokens"] = 2
        result["usage"]["output_tokens_details"]["reasoning_tokens"] = 1
    else:
        # Missing policy usage must stay unknown even when all recorded turns have usage.
        result["usage"] = None
    if not records_complete:
        result["steps"].append({"step": 1, "model_text": "", "actions": [], "reward": 0.0, "done": True, "info": {}})
    row, _ = _collect_and_check(tmp_path, body, result, captures)
    assert row["response"]["usage"] == result["usage"]


@pytest.mark.parametrize(
    "missing,verdict", [("response_id", "unobserved"), ("started_at", "unobserved"), ("usage", "healthy")]
)
def test_missing_evidence_is_not_invented(tmp_path, observed_run, missing, verdict):
    body, result, captures = observed_run
    result["steps"][0]["info"]["agent"]["model_calls"][0]["response"]["observation"].pop(missing)
    row, digest = _collect_and_check(tmp_path, body, result, captures)
    assert digest.verdict == verdict
    if missing == "usage":
        # Captured calls and the independent policy total still supply usage to health checks.
        assert digest.unobserved == []
        assert row["response"]["usage"] == result["usage"]
        assert {"code": "model_call_usage_unavailable", "invocation_id": None, "detail": "call:0"} in row[
            "ng_trajectory"
        ]["gaps"]
    else:
        assert digest.unobserved


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
    assert "rollout_ended_on_failed_model_call" in digest.unobserved
    assert "rollout_token_count_mismatch" in digest.unobserved
    assert {gap["code"] for gap in row["ng_trajectory"]["gaps"]} >= {
        "model_calls_unavailable",
        "turn_model_call_scope_incomplete",
    }
    assert row["response"]["usage"] == result["usage"]


@pytest.mark.parametrize("failed_request_recorded", [False, True])
def test_missing_completion_records_are_unobserved_turns(tmp_path, observed_run, failed_request_recorded):
    body, result, captures = observed_run
    result["steps"][0]["info"]["agent"].pop("model_calls")
    if failed_request_recorded:
        failed_step = copy.deepcopy(result["steps"][0])
        failed_step["info"]["agent"]["model_calls"] = [
            {"parse_attempt": 1, "prompt_messages": [], "response": None, "completion_received": False}
        ]
        result["steps"].insert(0, failed_step)
        captures.insert(
            0,
            {
                "model_call_id": "failed-request",
                "started_at": 999.0,
                "model_ref": MODEL_REF.model_dump(),
                "status_code": 503,
            },
        )
    row, digest = _collect_and_check(tmp_path, body, result, captures)

    assert digest.verdict == "unobserved"
    assert "rollout_missing_agent_turns" in digest.unobserved
    assert not digest.findings
    assert row["ng_trajectory"]["turns"] == []
    assert row["response"]["usage"] == result["usage"]


def test_complete_zero_call_observation_still_reports_missing_turns(tmp_path, observed_run):
    body, result, _ = observed_run
    result["steps"] = []
    result["usage"] = None
    row, digest = _collect_and_check(tmp_path, body, result, [])

    assert row["ng_trajectory"]["turns"] == []
    assert "turns_unavailable" not in {gap["code"] for gap in row["ng_trajectory"]["gaps"]}
    assert "rollout_missing_agent_turns" in {finding.check for finding in digest.findings}


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
    [
        ("empty", "agent_turn_hollow"),
        ("failed", "rollout_ended_on_failed_model_call"),
        ("tokens", "rollout_token_count_mismatch"),
    ],
)
def test_real_findings_are_not_hidden(tmp_path, observed_run, fault, check):
    body, result, captures = observed_run
    response = result["steps"][0]["info"]["agent"]["model_calls"][0]["response"]
    if fault == "empty":
        response["content"] = response["raw_content"] = ""
    elif fault == "failed":
        captures[-1]["status_code"] = 500
    else:
        result["usage"]["output_tokens"] += 1
        result["usage"]["total_tokens"] += 1
    _, digest = _collect_and_check(tmp_path, body, result, captures)
    assert digest.verdict == "unhealthy"
    assert check in {finding.check for finding in digest.findings}


@pytest.mark.parametrize("logging_enabled", [False, True])
@pytest.mark.parametrize(
    "failure,terminal,verdict",
    [
        ("connection", False, "healthy"),
        ("http", False, "healthy"),
        ("connection", True, "unhealthy"),
        ("http", True, "unhealthy"),
        ("ambiguous", False, "unobserved"),
        ("post_response", False, "unobserved"),
        ("post_response_unknown_usage", False, "unobserved"),
    ],
)
def test_request_outcome_reaches_health_checks(
    monkeypatch, tmp_path, observed_run, logging_enabled, failure, terminal, verdict
):
    if logging_enabled:
        monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(tmp_path / "model.jsonl"))
    else:
        monkeypatch.delenv("OSWORLD_MODEL_IO_LOG", raising=False)
    body, result, _ = observed_run
    captures = []
    answer = "## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```"
    request = httpx.Request("POST", "http://policy/v1/chat/completions")

    def create(**kwargs):
        index = len(captures)
        capture = {
            "model_call_id": f"capture-{index}",
            "call_index": index,
            "started_at": 1000.0 + index,
            "model_ref": MODEL_REF.model_dump(),
            "request": {"messages": copy.deepcopy(kwargs["messages"])},
        }
        captures.append(capture)
        if index == 0 and failure in {"connection", "http", "ambiguous"}:
            if failure == "connection":
                error = APIConnectionError(message="policy endpoint unreachable", request=request)
            elif failure == "http":
                error = InternalServerError(
                    "policy endpoint unavailable", response=httpx.Response(503, request=request), body=None
                )
            else:
                error = ValueError("could not decode the upstream response")
            capture.update(
                status_code=503 if failure == "http" else None,
                error_category="capture_parse_error" if failure == "ambiguous" else "client_error",
            )
            raise error
        completion = ChatCompletion.model_validate(
            {
                "id": f"completion-{index}",
                "created": 1000 + index,
                "model": "super",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        )
        if index == 0 and failure.startswith("post_response"):
            # The wrapper receives and counts this completion, then rejects it before
            # returning a response to the adapter. It must not look like a failed request.
            completion.choices = []
            if failure == "post_response_unknown_usage":
                completion.usage = None
        capture.update(
            status_code=200,
            response_id=completion.id,
            finish_reason="stop",
            tokens_in=completion.usage.prompt_tokens if completion.usage is not None else None,
            tokens_out=completion.usage.completion_tokens if completion.usage is not None else None,
            response=completion.model_dump(mode="json"),
        )
        return completion

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    usage = _PolicyUsage()
    with patch("responses_api_agents.osworld_agent.app._build_policy_openai_client", return_value=client):
        caller = _build_messages_model_fn(
            base_url="http://policy/v1", model_name="super", api_key="unused", policy_usage=usage
        )
        agent = NemotronV3NanoOmniAgent(model="super", max_steps=1, parse_retries=1 if terminal else 2)
        agent.call_llm = lambda payload, model: caller(payload["messages"], payload)
        text, actions, info = agent.predict("Click.", {"screenshot": b"fake-png"})
    result["steps"][0].update(model_text=text, actions=actions, info={"agent": info})
    result["usage"] = usage.reported().model_dump(mode="json") if usage.reported() is not None else None
    row, digest = _collect_and_check(tmp_path, body, result, captures)

    assert digest.verdict == verdict
    assert row["response"]["usage"] == result["usage"]
    assert digest.model_calls == len(captures) == (1 if terminal else 2)
    assert len(row["ng_trajectory"]["turns"]) == (0 if terminal else 1)
    assert info["model_calls"][0]["response"] is None
    if terminal:
        assert "rollout_ended_on_failed_model_call" in {finding.check for finding in digest.findings}
    elif verdict == "healthy":
        assert digest.unobserved == []
        assert not digest.findings
        assert digest.transcript_prompt_tokens == digest.capture_prompt_tokens == 3
        assert digest.transcript_completion_tokens == digest.capture_completion_tokens == 2
        assert row["ng_trajectory"]["turns"][0]["turn_no"] == 1
    else:
        assert "rollout_token_count_mismatch" in digest.unobserved
        assert "turns_unavailable" in {gap["code"] for gap in row["ng_trajectory"]["gaps"]}
        if failure == "post_response":
            assert row["response"]["usage"]["input_tokens"] == 6
            assert row["response"]["usage"]["output_tokens"] == 4
        elif failure == "post_response_unknown_usage":
            assert row["response"]["usage"] is None


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
