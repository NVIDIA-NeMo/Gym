# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Messages wire exchanges, without installing or launching a harness runtime."""

import json
import subprocess

import pytest
from fastapi.testclient import TestClient
from scripts.harness_conformance.provider import Probe
from scripts.harness_conformance.runner import _fingerprint
from scripts.harness_conformance.scenarios import SCENARIOS

from nemo_gym.rollouts.harness_capabilities import inspect_behavior, json_rows
from nemo_gym.trajectory import build_model_call_record


def messages_body(stream):
    return {
        "model": "prepared-policy-alias",
        "max_tokens": 1024,
        "stream": stream,
        "messages": [{"role": "user", "content": "Run the prescribed checks"}],
        "tools": [
            {
                "name": "Bash",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}, "description": {"type": "string"}},
                    "required": ["command", "description"],
                },
            }
        ],
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scenario_name", ["tool_success", "tool_failure", "retry_429", "retry_500", "usage_omitted"])
def test_messages_tools_retries_and_usage_match_raw_capture(tmp_path, stream, scenario_name):
    scenario = next(s for s in SCENARIOS if s.name == scenario_name)
    probe = Probe(scenario, tmp_path)
    body = messages_body(stream)
    with TestClient(probe.model_app()) as client:
        for status in scenario.http_errors:
            reply = client.post("/ng-rollout/0-0/v1/messages", json=body)
            assert reply.status_code == status
            assert reply.json()["error"]["type"] == "conformance_error"
        for step in range(scenario.tool_steps + 1):
            reply = client.post("/ng-rollout/0-0/v1/messages", json=body)
            assert reply.status_code == 200
            payload = probe.attempts[-1]["response"]
            assert payload["model"] == body["model"]
            assert payload["type"] == "message"
            if stream:
                events = [json.loads(line[6:]) for line in reply.text.splitlines() if line.startswith("data: ")]
                assert events[-1] == {"type": "message_stop"}
                assert events[-2]["delta"]["stop_reason"] == payload["stop_reason"]
                if not scenario.usage:
                    assert "usage" not in events[0]["message"]
                    assert all("usage" not in event for event in events)
            else:
                assert reply.json() == payload
            if step == scenario.tool_steps:
                assert payload["content"] == [{"type": "text", "text": "CONFORMANCE_DONE"}]
                assert payload["stop_reason"] == "end_turn"
                break
            [tool] = payload["content"]
            assert tool["type"] == "tool_use" and tool["name"] == "Bash"
            assert tool["input"]["description"] == "Run conformance shell check"
            assert payload["stop_reason"] == "tool_use"
            # Execute the requested command independently; its marker must be witnessed by the probe.
            result = subprocess.run(["sh", "-c", tool["input"]["command"]], capture_output=True, text=True)
            assert result.returncode == (scenario.tool_exit_code if step == 0 else 0)
            body["messages"].extend(
                [
                    {"role": "assistant", "content": payload["content"]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool["id"],
                                "content": result.stdout,
                                "is_error": result.returncode != 0,
                            }
                        ],
                    },
                ]
            )
    assert probe.finished and not probe.violations
    assert all(call["executed"] and call["result_seen"] for call in probe.tool_calls)
    captures = [row for _, row in json_rows(tmp_path / "capture/0-0.capture.jsonl")]
    assert len(captures) == len(probe.attempts)
    canonical = []
    for index, (capture, witness) in enumerate(zip(captures, probe.attempts)):
        assert capture["request"] == witness["request"]
        assert capture["response"] == witness["response"]
        assert "messages" in capture["request"] and "input" not in capture["request"]
        assert "token_ids" not in json.dumps(capture)
        call = build_model_call_record(capture, call_index=index)
        assert call.tokens_reasoning is None  # Messages does not report a reasoning count.
        if not scenario.usage or witness["status_code"] != 200:
            assert call.tokens_in is call.tokens_out is call.tokens_total is None
            assert "usage" not in witness["response"]
        else:
            assert call.tokens_in == 20 + index and call.tokens_out == 8
            assert call.tokens_total == 28 + index
        canonical.append(
            {
                "request": call.request,
                "response": call.response,
                "response_metadata": {
                    "status_code": call.status_code,
                    "response_id": call.response_id,
                    "response_status": call.response_status,
                    "finish_reason": call.finish_reason,
                },
                "token_stats": {
                    "prompt_tokens": call.tokens_in,
                    "completion_tokens": call.tokens_out,
                    "reasoning_tokens": call.tokens_reasoning,
                    "total_tokens": call.tokens_total,
                    "cached_tokens": call.cached_tokens,
                },
            }
        )
    checks = inspect_behavior(
        json.loads((tmp_path / "witness.json").read_text()),
        {"ng_trajectory": {"model_calls": canonical}},
        http_errors=scenario.http_errors,
        terminal_error=scenario.terminal_error,
        tool_steps=scenario.tool_steps,
        expected_reward=scenario.expected_reward,
        fingerprint=_fingerprint,
    )
    expected = {"model.saved_exchanges": "pass", "model.response_metadata": "pass", "model.token_counts": "pass"}
    assert {check["id"]: check["status"] for check in checks if check["id"] in expected} == expected


def test_messages_requires_actual_tool_execution_and_matching_result(tmp_path):
    probe = Probe(next(s for s in SCENARIOS if s.name == "tool_success"), tmp_path)
    body = messages_body(False)
    with TestClient(probe.model_app()) as client:
        assert client.post("/v1/messages", json=body).status_code == 200
        assert client.post("/v1/messages", json=body).status_code == 409
    assert probe.violations and not probe.finished
