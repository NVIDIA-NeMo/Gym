# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import pytest
from run_agent import AIAgent
from tools.delegate_tool import MAX_CONCURRENT_CHILDREN, _build_child_agent

from responses_api_agents.hermes_agent.model_kwargs import install_summary_compat
from responses_api_agents.hermes_agent.sandbox_runner import _run
from responses_api_agents.hermes_agent.tests.test_sandbox_runner import _completion, _ModelServer, _payload


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(AIAgent, "__init__", AIAgent.__init__)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    dispatched = []

    def execute(_agent, message, messages, _task_id, _api_call_count=0):
        for call in message.tool_calls:
            dispatched.append(call.model_dump())
            messages.append({"role": "tool", "tool_call_id": call.id, "content": "executed"})

    monkeypatch.setattr(AIAgent, "_execute_tool_calls_sequential", execute)
    monkeypatch.setattr(AIAgent, "_execute_tool_calls_concurrent", execute)
    return dispatched


def tool(name: str, *, index: int = 0, arguments: str = "{}") -> dict:
    return {"id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": arguments}}


def conversation(tmp_path, calls: list[dict], *, training: bool, max_turns: int = 4, content: str = "work"):
    first = _completion({"content": content, "tool_calls": calls})
    first["choices"][0]["finish_reason"] = "tool_calls"
    with _ModelServer([first, _completion({"content": "done"})]) as server:
        output = _run(
            _payload(
                server.base_url,
                max_turns=max_turns,
                preserve_training_history=training,
                enabled_toolsets=["terminal", "delegation"],
            ),
            tmp_path,
        )
    assert len(server.requests) == 2
    return server.requests[1]["messages"], output


@pytest.mark.parametrize("training", [False, True])
def test_invalid_tool_name_is_reported_without_rewriting_training_history(tmp_path, runtime, training):
    messages, output = conversation(tmp_path, [tool("termnal")], training=training)
    assistant = next(message for message in messages if message.get("tool_calls"))
    assert assistant["tool_calls"][0]["function"]["name"] == ("termnal" if training else "terminal")
    results = [message for message in messages if message["role"] == "tool"]
    assert len(results) == 1
    assert ("does not exist" in results[0]["content"]) is training
    assert len(runtime) == (0 if training else 1)
    assert output["result"]["completed"]


@pytest.mark.parametrize("training", [False, True])
def test_duplicate_calls_stay_in_history_but_execute_once(tmp_path, runtime, training):
    calls = [tool("terminal", index=index) for index in range(2)]
    messages, _ = conversation(tmp_path, calls, training=training)
    assistant = next(message for message in messages if message.get("tool_calls"))
    assert assistant["tool_calls"] == (calls if training else calls[:1])
    assert len(runtime) == 1
    results = [message for message in messages if message["role"] == "tool"]
    assert len(results) == (2 if training else 1)
    if training:
        assert results[-1]["tool_call_id"] == "call-1"
        assert "No execution occurred" in results[-1]["content"]


@pytest.mark.parametrize("training", [False, True])
def test_delegation_limit_keeps_all_served_calls_and_reports_skipped_dispatch(tmp_path, runtime, training):
    calls = [
        tool("delegate_task", index=index, arguments=json.dumps({"goal": str(index)}))
        for index in range(MAX_CONCURRENT_CHILDREN + 2)
    ]
    messages, _ = conversation(tmp_path, calls, training=training)
    assistant = next(message for message in messages if message.get("tool_calls"))
    assert assistant["tool_calls"] == (calls if training else calls[:MAX_CONCURRENT_CHILDREN])
    assert len(runtime) == MAX_CONCURRENT_CHILDREN
    results = [message for message in messages if message["role"] == "tool"]
    assert len(results) == (len(calls) if training else MAX_CONCURRENT_CHILDREN)
    if training:
        assert all("No execution occurred" in result["content"] for result in results[-2:])


@pytest.mark.parametrize("training", [False, True])
def test_argument_normalization_only_changes_dispatch_in_training(tmp_path, runtime, training):
    messages, _ = conversation(tmp_path, [tool("terminal", arguments="")], training=training)
    assistant = next(message for message in messages if message.get("tool_calls"))
    assert assistant["tool_calls"][0]["function"]["arguments"] == ("" if training else "{}")
    assert runtime[0]["function"]["arguments"] == "{}"


@pytest.mark.parametrize("training", [False, True])
def test_iteration_summary_preserves_reasoning_echo_in_training(tmp_path, runtime, training):
    messages, _ = conversation(
        tmp_path, [tool("terminal")], training=training, max_turns=1, content="<think>check</think>work"
    )
    assistant = next(message for message in messages if message.get("tool_calls"))
    assert assistant["content"] == "<think>check</think>work"
    assert assistant.get("reasoning_content") == ("check" if training else None)


def test_training_adapter_reaches_upstream_delegated_children(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("run_agent.get_tool_definitions", return_value=[]), patch("run_agent.OpenAI"):
        parent = AIAgent(model="model", quiet_mode=True, skip_context_files=True, skip_memory=True)
        install_summary_compat(parent, preserve_reasoning_history=True, preserve_training_history=True)
        child = _build_child_agent(
            task_index=0,
            goal="inspect",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=2,
            parent_agent=parent,
        )
    child.valid_tool_names = {"terminal"}
    assert child._repair_tool_call("termnal") is None
    assert child._gym_summary_compat_installed
