# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the native message and terminal protocols with real shell executions."""

import json
import subprocess
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from scripts.harness_conformance.provider import ANSWER, Probe
from scripts.harness_conformance.scenarios import SCENARIOS


SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}
BASH = {"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scenario_name", ["tool_success", "tool_failure", "usage_omitted", "retry_429"])
def test_messages_protocol_keeps_raw_exchanges_and_executes_tools(tmp_path, stream, scenario_name):
    probe = Probe(SCENARIOS_BY_NAME[scenario_name], tmp_path, harness="claude_code")
    body = {
        "model": "probe",
        "messages": [{"role": "user", "content": "Run checks"}],
        "tools": [BASH],
        "stream": stream,
    }
    with TestClient(probe.model_app()) as client:
        for status in probe.scenario.http_errors:
            assert client.post("/ng-rollout/0-0/v1/messages", json=body).status_code == status
        for _ in range(2):
            response = client.post("/ng-rollout/0-0/v1/messages", json=body)
            assert response.status_code == 200
            call = probe.attempts[-1]["response"]["content"][0]
            execution = subprocess.run(call["input"]["command"], shell=True, capture_output=True, text=True)
            body["messages"].extend(
                [
                    {"role": "assistant", "content": [call]},
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": execution.stdout}],
                    },
                ]
            )
        assert client.post("/ng-rollout/0-0/v1/messages", json=body).status_code == 200
    assert probe.finished and not probe.violations
    assert all(t["executed"] and t["result_seen"] for t in probe.tool_calls)
    captures = [json.loads(line) for line in (tmp_path / "capture/0-0.capture.jsonl").read_text().splitlines()]
    assert [c["response"] for c in captures] == [a["response"] for a in probe.attempts]
    assert probe.attempts[-1]["response"]["content"] == [{"type": "text", "text": ANSWER}]
    usage = probe.attempts[-1]["response"].get("usage")
    if probe.scenario.usage:
        assert usage == {
            "input_tokens": 22 + len(probe.scenario.http_errors),
            "output_tokens": 8,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        }
    else:
        assert all("usage" not in c["response"] for c in captures)


def test_auxiliary_title_does_not_consume_retry_sequence_but_is_captured(tmp_path):
    probe = Probe(SCENARIOS_BY_NAME["retry_429"], tmp_path, harness="claude_code")
    title = {
        "model": "probe",
        "messages": [],
        "tools": [],
        "stream": True,
        "output_config": {"format": {"type": "json_schema", "schema": {"properties": {"title": {"type": "string"}}}}},
    }
    policy = {"model": "probe", "messages": [], "tools": [BASH]}
    with TestClient(probe.model_app()) as client:
        assert client.post("/ng-rollout/0-0/v1/messages", json=title).status_code == 200
        assert client.post("/ng-rollout/0-0/v1/messages", json=policy).status_code == 429
        assert client.post("/ng-rollout/0-0/v1/messages", json=policy).status_code == 429
        assert client.post("/ng-rollout/0-0/v1/messages", json=policy).status_code == 200
    witness = json.loads((tmp_path / "witness.json").read_text())
    assert len(witness["auxiliary_calls"]) == 1
    assert len(witness["attempts"]) == 3
    assert len((tmp_path / "capture/0-0.capture.jsonl").read_text().splitlines()) == 4
    assert not probe.finished and not probe.violations


def test_terminus_executes_json_commands_and_confirms_completion_twice(tmp_path):
    probe = Probe(SCENARIOS_BY_NAME["tool_failure"], tmp_path, harness="terminus_2_sandboxed")
    body = {"model": "probe", "input": [{"role": "user", "content": "Run checks"}]}
    with TestClient(probe.model_app()) as client:
        for index in range(4):
            response = client.post("/v1/responses", json=body)
            assert response.status_code == 200
            message = response.json()["output"][0]
            instruction = json.loads(message["content"][0]["text"])
            body["input"].append(message)
            if index < 2:
                execution = subprocess.run(
                    instruction["commands"][0]["keystrokes"], shell=True, capture_output=True, text=True
                )
                assert execution.returncode == (7 if index == 0 else 0)
                body["input"].append({"role": "user", "content": execution.stdout})
                assert instruction["task_complete"] is False
            else:
                assert instruction["task_complete"] is True and not instruction["commands"]
                assert probe.finished is (index == 3)
                body["input"].append({"role": "user", "content": "Confirm completion"})
    assert not probe.violations
    assert all(t["executed"] and t["result_seen"] for t in probe.tool_calls)


def test_miniswe_finishes_using_native_submission_sentinel(tmp_path):
    probe = Probe(replace(SCENARIOS_BY_NAME["tool_success"], tool_steps=0), tmp_path, harness="miniswe_sandboxed")
    body = {
        "model": "probe",
        "input": [],
        "tools": [{"type": "function", "name": "bash", "parameters": BASH["input_schema"]}],
    }
    with TestClient(probe.model_app()) as client:
        response = client.post("/v1/responses", json=body)
    call = response.json()["output"][0]
    execution = subprocess.run(json.loads(call["arguments"])["command"], shell=True, capture_output=True, text=True)
    assert execution.returncode == 0
    assert execution.stdout == f"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{ANSWER}\n"
    assert probe.finished and probe.finish_call["id"] == call["call_id"]
