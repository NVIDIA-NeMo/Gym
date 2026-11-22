# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from responses_api_agents.hermes_agent.model_kwargs import _model_api_kwargs


@pytest.mark.parametrize("source", ["extra_body", "metadata"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("preserve_history", [False, True])
def test_thinking_is_owned_by_model_server(source, thinking, preserve_history):
    template = {"enable_thinking": thinking, "other_template_setting": "kept"}
    original = {
        "model": "policy",
        "metadata": {"trace_id": "episode"},
        "extra_body": {"other_provider_setting": 42},
    }
    original[source]["chat_template_kwargs"] = json.dumps(template) if source == "metadata" else template
    before = deepcopy(original)
    result = _model_api_kwargs(original, preserve_reasoning_history=preserve_history)
    assert original == before
    assert result["model"] == "policy"
    assert result["extra_body"] == {"other_provider_setting": 42}
    assert result["metadata"]["trace_id"] == "episode"
    expected = {"other_template_setting": "kept"}
    if preserve_history:
        expected["truncate_history_thinking"] = False
    assert json.loads(result["metadata"]["chat_template_kwargs"]) == expected


@pytest.mark.parametrize("preserve_history", [False, True])
def test_absent_thinking_does_not_inject_a_default(preserve_history):
    result = _model_api_kwargs({}, preserve_reasoning_history=preserve_history)
    expected = {"metadata": {"chat_template_kwargs": json.dumps({"truncate_history_thinking": False})}}
    assert result == (expected if preserve_history else {})


def test_template_settings_merge_without_mutating_hermes_input():
    original = {
        "metadata": {"chat_template_kwargs": json.dumps({"from_metadata": 1, "shared": "old"})},
        "extra_body": {"chat_template_kwargs": {"from_extra_body": 2, "shared": "new"}},
    }
    before = deepcopy(original)
    result = _model_api_kwargs(original, preserve_reasoning_history=False)
    assert "extra_body" not in result
    assert json.loads(result["metadata"]["chat_template_kwargs"]) == {
        "from_metadata": 1,
        "from_extra_body": 2,
        "shared": "new",
    }
    assert original == before


def test_standalone_runner_loads_shared_helper_without_gym_on_path(tmp_path):
    source = Path(__file__).parents[1]
    for name in ("sandbox_runner.py", "sandbox_observer.py", "model_kwargs.py"):
        shutil.copyfile(source / name, tmp_path / name)
    # -S excludes site-packages (including Gym); only the uploaded runtime files are importable.
    completed = subprocess.run(
        [sys.executable, "-S", str(tmp_path / "sandbox_runner.py")],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert completed.returncode == 2
    assert "usage: sandbox_runner.py" in completed.stderr
    assert "ImportError" not in completed.stderr


class _ReplayAgent:
    """The Hermes agent surface configure_model_calls wraps, replaying one assistant turn with reasoning."""

    api_mode = "chat_completions"
    max_tokens = None

    def __init__(self, assistant_turn: dict):
        self.assistant_turn = assistant_turn

    def _build_api_kwargs(self, api_messages, tools_for_api=None):
        return {"model": "policy", "messages": [dict(message) for message in api_messages]}

    def _build_assistant_message(self, assistant_message, finish_reason):
        return {"role": "assistant", "content": ""}

    def _handle_max_iterations(self, messages, api_call_count):
        return ""

    def _get_transport(self, api_mode=None):
        return None


@pytest.mark.parametrize("preserve_history", [False, True])
@pytest.mark.parametrize(
    ("turn", "expected_reasoning"),
    [
        ({"reasoning": "think", "reasoning_content": "think"}, {"reasoning_content": "think"}),
        ({"reasoning": "think", "reasoning_content": " "}, {"reasoning_content": "think"}),
        ({"reasoning_content": "think"}, {"reasoning_content": "think"}),
        ({"reasoning": "think"}, {"reasoning": "think"}),
    ],
)
def test_replayed_assistant_turn_carries_one_reasoning_field(preserve_history, turn, expected_reasoning):
    """A replayed turn names its reasoning once: a body with both fields is rejected by servers that read them
    as one field."""
    from responses_api_agents.hermes_agent.model_kwargs import configure_model_calls

    assistant = {"role": "assistant", "content": None, "tool_calls": [{"id": "t", "type": "function"}], **turn}
    messages = [{"role": "user", "content": "q"}, assistant, {"role": "tool", "tool_call_id": "t", "content": "1"}]
    agent = _ReplayAgent(assistant)
    configure_model_calls(agent, preserve_reasoning_history=preserve_history)
    sent = agent._build_api_kwargs(messages)["messages"]
    replayed = sent[1]
    assert {key: replayed[key] for key in ("reasoning", "reasoning_content") if key in replayed} == expected_reasoning
    assert replayed["tool_calls"] == assistant["tool_calls"]
    assert sent[0] == messages[0] and sent[2] == messages[2]
    assert messages[1] == assistant  # Hermes's own history is left as it was
