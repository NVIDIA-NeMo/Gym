# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the native message and terminal protocols with real shell executions."""

import json
import subprocess
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from scripts.harness_conformance.episode import _settings
from scripts.harness_conformance.probes import ANSWER, PiProbe, Probe
from scripts.harness_conformance.scenarios import SCENARIOS


SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}
BASH = {"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scenario_name", ["tool_success", "tool_failure", "usage_omitted", "retry_429"])
def test_messages_protocol_keeps_raw_exchanges_and_executes_tools(tmp_path, stream, scenario_name):
    probe = Probe(SCENARIOS_BY_NAME[scenario_name], tmp_path)
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


@pytest.mark.parametrize("scenario_name", ["tool_success", "tool_failure"])
@pytest.mark.parametrize("stdout_visible", [False, True])
def test_terminus_requires_stdout_not_just_echoed_keystrokes(tmp_path, scenario_name, stdout_visible):
    probe = Probe(SCENARIOS_BY_NAME[scenario_name], tmp_path, tool_protocol="terminal_json")
    body = {"model": "probe", "input": [{"role": "user", "content": "Run checks"}]}
    with TestClient(probe.model_app()) as client:
        response = client.post("/v1/responses", json=body)
        assert response.status_code == 200
        message = response.json()["output"][0]
        command = json.loads(message["content"][0]["text"])["commands"][0]["keystrokes"]
        execution = subprocess.run(command, shell=True, capture_output=True, text=True)
        call = probe.tool_calls[0]
        assert execution.returncode == probe.scenario.tool_exit_code
        assert execution.stdout == call["token"]
        assert (tmp_path / "tool-0.txt").read_text() == call["token"]
        terminal_content = command + (execution.stdout if stdout_visible else "")
        body["input"].extend([message, {"role": "user", "content": terminal_content}])
        response = client.post("/v1/responses", json=body)
    assert response.status_code == (200 if stdout_visible else 409)
    assert call["executed"] is True
    assert call["result_seen"] is stdout_visible
    assert call["outputs"] == ([terminal_content] if stdout_visible else [])
    assert len(probe.tool_calls) == (2 if stdout_visible else 1)
    assert call["token"] not in command


@pytest.mark.parametrize("stream", [False, True])
def test_messages_compaction_keeps_summary_and_usage_separate_from_policy(tmp_path, stream):
    probe = Probe(
        replace(SCENARIOS_BY_NAME["context_compaction"], tool_steps=1),
        tmp_path,
        responses=[
            {"kind": "tool", "input_tokens": 27000, "output_tokens": 8},
            {"kind": "summary", "input_tokens": 500, "output_tokens": 11},
            {"kind": "final", "input_tokens": 101, "output_tokens": 8},
        ],
    )
    body = {"model": "probe", "messages": [], "tools": [BASH], "stream": stream}
    with TestClient(probe.model_app()) as client:
        assert client.post("/ng-rollout/0-0/v1/messages", json=body).status_code == 200
        first = probe.attempts[-1]["response"]
        call = first["content"][0]
        execution = subprocess.run(call["input"]["command"], shell=True, capture_output=True, text=True, check=True)
        body["messages"].extend(
            [
                {"role": "assistant", "content": [call]},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": execution.stdout}],
                },
            ]
        )
        summary_request = {**body, "messages": [{"role": "user", "content": "Any summary instruction"}]}
        assert client.post("/ng-rollout/0-0/v1/messages", json=summary_request).status_code == 200
        helper = probe.attempts[-1]["response"]
        assert helper["content"] == [{"type": "text", "text": probe.compaction_summary}]
        assert helper["usage"]["input_tokens"] == 500
        assert helper["usage"]["output_tokens"] == 11
        assert probe.compaction_response_ids == [helper["id"]]
        assert not probe.finished and len(probe.tool_calls) == 1
        body["messages"].append({"role": "assistant", "content": helper["content"]})
        assert client.post("/ng-rollout/0-0/v1/messages", json=body).status_code == 200
    final = probe.attempts[-1]["response"]
    assert final["content"] == [{"type": "text", "text": ANSWER}]
    assert first["usage"]["input_tokens"] == 27000
    assert final["usage"]["input_tokens"] == 101
    assert probe.finished and not probe.violations
    captures = [json.loads(line) for line in (tmp_path / "capture/0-0.capture.jsonl").read_text().splitlines()]
    assert [c["response"] for c in captures] == [a["response"] for a in probe.attempts]


@pytest.mark.parametrize("scenario_name", ["tool_failure", "context_compaction"])
def test_terminus_executes_json_commands_and_confirms_completion_twice(tmp_path, scenario_name):
    scenario = replace(SCENARIOS_BY_NAME[scenario_name], tool_steps=2)
    settings = _settings("terminus_2_sandboxed", tmp_path, 30, scenario)
    probe = Probe(scenario, tmp_path, **settings.probe_overrides)
    body = {"model": "probe", "input": [{"role": "user", "content": "Run checks"}]}
    with TestClient(probe.model_app()) as client:
        for index, reply in enumerate(probe.responses):
            response = client.post("/v1/responses", json=body)
            assert response.status_code == 200
            message = response.json()["output"][0]
            body["input"].append(message)
            if reply["kind"] == "summary":
                assert message["content"][0]["text"] == probe.compaction_summary
                assert not probe.finished
                continue
            instruction = json.loads(message["content"][0]["text"])
            if reply["kind"] == "tool":
                assert probe.tool_calls[index]["token"] not in instruction["commands"][0]["keystrokes"]
                execution = subprocess.run(
                    instruction["commands"][0]["keystrokes"], shell=True, capture_output=True, text=True
                )
                assert execution.returncode == (scenario.tool_exit_code if index == 0 else 0)
                body["input"].append({"role": "user", "content": execution.stdout})
                assert instruction["task_complete"] is False
            else:
                assert instruction["task_complete"] is True and not instruction["commands"]
                assert probe.finished is (index == len(probe.responses) - 1)
                body["input"].append({"role": "user", "content": "Confirm completion"})
    assert not probe.violations
    assert len(probe.compaction_response_ids) == (3 if scenario.compaction else 0)
    assert all(t["executed"] and t["result_seen"] for t in probe.tool_calls)


def test_miniswe_finishes_using_native_submission_sentinel(tmp_path):
    probe = Probe(replace(SCENARIOS_BY_NAME["tool_success"], tool_steps=0), tmp_path, finish_protocol="bash_marker")
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


def test_pi_overflow_preserves_error_and_resumes_scripted_replies(tmp_path):
    scenario = replace(SCENARIOS_BY_NAME["context_compaction"], tool_steps=2)
    settings = _settings("pi", tmp_path, 30, scenario)
    probe = PiProbe(scenario, tmp_path, **settings.probe_overrides)
    body = {
        "model": "probe",
        "messages": [{"role": "user", "content": "Run checks"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "bash", "parameters": {"properties": {"command": {"type": "string"}}}},
            }
        ],
    }
    with TestClient(probe.model_app()) as client:
        for _ in range(scenario.tool_steps):
            response = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
            assert response.status_code == 200
            message = response.json()["choices"][0]["message"]
            call = message["tool_calls"][0]
            execution = subprocess.run(
                json.loads(call["function"]["arguments"])["command"],
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            assert execution.stdout == "x" * 20000 + probe.tool_calls[-1]["token"]
            body["messages"].extend(
                [message, {"role": "tool", "tool_call_id": call["id"], "content": execution.stdout}]
            )
        overflow = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
        assert overflow.status_code == 400
        assert overflow.json()["error"]["code"] == "context_length_exceeded"
        assert not probe.finished and not probe.compaction_response_ids
        assert all(t["executed"] and t["result_seen"] for t in probe.tool_calls)
        summary = client.post("/ng-rollout/0-0/v1/chat/completions", json={**body, "messages": []})
        assert summary.json()["choices"][0]["message"]["content"] == probe.compaction_summary
        body["messages"] = [{"role": "user", "content": probe.compaction_summary}]
        final = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
        assert final.json()["choices"][0]["message"]["content"] == ANSWER
        assert client.post("/ng-rollout/0-0/v1/chat/completions", json=body).status_code == 409
    captures = [json.loads(line) for line in (tmp_path / "capture/0-0.capture.jsonl").read_text().splitlines()]
    assert [c["response"] for c in captures] == [a["response"] for a in probe.attempts]
    assert [a["status_code"] for a in probe.attempts] == [200, 200, 400, 200, 200, 409]
