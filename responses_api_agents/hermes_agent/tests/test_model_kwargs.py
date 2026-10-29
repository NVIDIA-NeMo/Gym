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


def test_sampling_controls_apply_to_calls_summaries_and_children():
    from types import SimpleNamespace

    from responses_api_agents.hermes_agent.model_kwargs import install_summary_compat

    requests = []

    def agent():
        return SimpleNamespace(
            _ensure_primary_openai_client=lambda **kwargs: None,
            _handle_max_iterations=lambda *args: None,
            _interruptible_api_call=lambda kwargs: requests.append(kwargs),
            _active_children=[],
            _sanitize_tool_calls_for_strict_api=lambda message: message,
        )

    root = agent()
    install_summary_compat(
        root,
        preserve_reasoning_history=False,
        sampling_overrides={"temperature": 0.7, "top_p": 0.8, "max_tokens": 64},
    )
    original = {"temperature": 1.0, "max_completion_tokens": 128}
    root._interruptible_api_call(original)
    summary = root._ensure_primary_openai_client(reason="iteration_limit_summary")
    summary.chat.completions.create(messages=[], temperature=0.0, max_tokens=32)
    child = agent()
    root._active_children.append(child)
    child._interruptible_api_call({"max_tokens": 256})
    grandchild = agent()
    child._active_children.append(grandchild)
    grandchild._interruptible_api_call({})
    assert original == {"temperature": 1.0, "max_completion_tokens": 128}
    assert [request["max_tokens"] for request in requests] == [64, 32, 64, 64]
    assert all(request["temperature"] == 0.7 and request["top_p"] == 0.8 for request in requests)
    assert all("max_completion_tokens" not in request for request in requests)
