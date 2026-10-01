# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from responses_api_agents.hermes_agent.model_kwargs import _model_api_kwargs


@pytest.mark.parametrize("source", ["extra_body", "metadata"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("configured", [False, True, None])
@pytest.mark.parametrize("preserve_history", [False, True])
def test_thinking_is_owned_by_model_server(caplog, source, thinking, configured, preserve_history):
    template = {"enable_thinking": thinking, "other_template_setting": "kept"}
    original = {
        "model": "policy",
        "metadata": {"trace_id": "episode"},
        "extra_body": {"other_provider_setting": 42},
    }
    original[source]["chat_template_kwargs"] = json.dumps(template) if source == "metadata" else template
    before = deepcopy(original)
    with caplog.at_level(logging.WARNING):
        result = _model_api_kwargs(
            original, preserve_reasoning_history=preserve_history, model_enable_thinking=configured
        )
    assert original == before
    assert result["model"] == "policy"
    assert result["extra_body"] == {"other_provider_setting": 42}
    assert result["metadata"]["trace_id"] == "episode"
    expected = {"other_template_setting": "kept"}
    if preserve_history:
        expected["truncate_history_thinking"] = False
    assert json.loads(result["metadata"]["chat_template_kwargs"]) == expected
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == int(configured is None or thinking != configured)
    if warnings:
        assert f"enable_thinking={thinking}" in warnings[0].message
        if configured is not None:
            assert f"Model Server enable_thinking={configured}" in warnings[0].message


@pytest.mark.parametrize("preserve_history", [False, True])
def test_absent_thinking_does_not_inject_a_default_or_warn(caplog, preserve_history):
    with caplog.at_level(logging.WARNING):
        result = _model_api_kwargs({}, preserve_reasoning_history=preserve_history, model_enable_thinking=False)
    expected = {"metadata": {"chat_template_kwargs": json.dumps({"truncate_history_thinking": False})}}
    assert result == (expected if preserve_history else {})
    assert not caplog.records


def test_template_settings_merge_without_mutating_hermes_input():
    original = {
        "metadata": {"chat_template_kwargs": json.dumps({"from_metadata": 1, "shared": "old"})},
        "extra_body": {"chat_template_kwargs": {"from_extra_body": 2, "shared": "new"}},
    }
    before = deepcopy(original)
    result = _model_api_kwargs(original, preserve_reasoning_history=False, model_enable_thinking=None)
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
