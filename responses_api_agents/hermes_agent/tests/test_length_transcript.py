# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real pinned Hermes and SDK over deterministic HTTP; no live model is claimed."""

import hashlib
import inspect
import json

import pytest
from run_agent import AIAgent

from nemo_gym.models.token_capture.completion import completion_items, output_item_fingerprint
from responses_api_agents.hermes_agent import length_transcript
from responses_api_agents.hermes_agent.app import _trajectory_to_output_items
from responses_api_agents.hermes_agent.sandbox_runner import _run
from responses_api_agents.hermes_agent.tests.compat_fixtures import (
    restore_length_transcript_state as restore_length_transcript_state,
)
from responses_api_agents.hermes_agent.tests.test_sandbox_runner import (
    _completion,
    _ModelServer,
    _payload,
)
from responses_api_agents.hermes_agent.tests.test_sandbox_runner import (
    restore_process_globals as restore_process_globals,
)


def run_provider_reply(tmp_path, message, *, finish_reason="length", history=None):
    """Run the real native entrypoint via Gym's actual sandbox runner."""
    answer = _completion(message)
    answer["choices"][0]["finish_reason"] = finish_reason
    history = history or []
    config = "model: policy_model\nprovider: auto\ncompression:\n  enabled: false\n"
    with _ModelServer([answer]) as model_server:
        payload = _payload(
            model_server.base_url,
            config_yaml=config,
            history=history,
            max_turns=3,
            temperature=1.0,
            enabled_toolsets=["terminal"],
        )
        output = _run(payload, tmp_path)
    assert len(model_server.requests) == 1
    assert model_server.requests[0].get("stream", False) is False
    assert model_server.requests[0]["temperature"] == 1.0
    result = output["result"]
    converted = [
        item.model_dump(mode="json") for item in _trajectory_to_output_items(result["messages"], len(history) + 1)
    ]
    captured = completion_items(answer)
    return answer, result, converted, captured


@pytest.mark.parametrize(
    "content",
    [
        "<think>unfinished reasoning",
        " \n<think>unfinished reasoning  ",
        "<think>reasoning</think>partial answer",
        "<think>" + "x" * 37302,
    ],
)
def test_length_keeps_actual_accepted_thinking(tmp_path, restore_process_globals, content):
    _, result, returned, captured = run_provider_reply(tmp_path, {"content": content})
    assert result["partial"] is True
    assert result["completed"] is False
    assert result["error"] == "Response truncated due to output length limit"
    assert result["final_response"] is None
    assert result["messages"][-1]["content"] == content
    assert returned[0]["content"][0]["text"] == content
    assert [output_item_fingerprint(item) for item in returned] == [output_item_fingerprint(item) for item in captured]


def test_length_preserves_adjacent_native_tool_fields(tmp_path, restore_process_globals):
    tool = {
        "id": "call-truncated",
        "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps({"command": "echo not-executed"})},
    }
    _, result, returned, captured = run_provider_reply(
        tmp_path, {"content": "<think>tool-adjacent partial", "tool_calls": [tool]}
    )
    assert result["partial"] is True
    assert [item["type"] for item in returned] == ["message", "function_call"]
    assert returned[-1]["call_id"] == tool["id"]
    assert returned[-1]["name"] == tool["function"]["name"]
    assert returned[-1]["arguments"] == tool["function"]["arguments"]
    assert [output_item_fingerprint(item) for item in returned] == [output_item_fingerprint(item) for item in captured]


def test_length_appends_once_after_existing_tool_history(tmp_path, restore_process_globals):
    history = [
        {"role": "user", "content": "earlier request"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-earlier", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-earlier", "content": "fixture tool result"},
    ]
    _, result, returned, captured = run_provider_reply(tmp_path, {"content": "<think>partial"}, history=history)
    assert result["messages"][: len(history)] == history
    assert len(result["messages"]) == len(history) + 2
    assert len(returned) == len(captured) == 1
    assert output_item_fingerprint(returned[0]) == output_item_fingerprint(captured[0])


@pytest.mark.parametrize("content", ["done", "<think>reasoning</think>done"])
def test_normal_stop_preserves_ordinary_native_result(tmp_path, restore_process_globals, content):
    _, result, returned, captured = run_provider_reply(tmp_path, {"content": content}, finish_reason="stop")
    assert result["completed"] is True
    assert not result.get("partial")
    assert not result.get("error")
    assert len(result["messages"]) == 2
    assert result["messages"][-1]["content"] == content
    assert [output_item_fingerprint(item) for item in returned] == [output_item_fingerprint(item) for item in captured]


def test_actual_source_pin_change_refuses_before_execution(monkeypatch, restore_process_globals):
    monkeypatch.setattr(length_transcript, "RUN_AGENT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="exact declared 26bb847a source"):
        length_transcript.install_length_transcript_compat(AIAgent)


def test_genuine_declared_dependency_identity(restore_process_globals):
    from pathlib import Path

    native_method = AIAgent.run_conversation
    assert native_method.__code__.co_firstlineno == 6392
    assert AIAgent._interruptible_api_call.__code__.co_firstlineno == 4075
    assert hashlib.sha256(Path(inspect.getsourcefile(AIAgent)).read_bytes()).hexdigest() == (
        length_transcript.RUN_AGENT_SHA256
    )
    # This validates real distribution direct_url.json, including the exact Git/archive pin.
    length_transcript.install_length_transcript_compat(AIAgent)
    installed = AIAgent.run_conversation
    assert installed.__wrapped__ is native_method
    length_transcript.install_length_transcript_compat(AIAgent)
    assert AIAgent.run_conversation is installed
