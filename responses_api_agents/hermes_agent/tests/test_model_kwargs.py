# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from responses_api_agents.hermes_agent.model_kwargs import (
    ModelRequestRejected,
    _model_api_kwargs,
    install_model_retry_policy,
)


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


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code} - {{'error': {{'message': 'controlled'}}}}")
        self.status_code = status_code
        self.body = {"message": "controlled"}
        self.response = object()


class _Agent:
    def __init__(self, error: Exception) -> None:
        self._client_kwargs = {"api_key": "gym", "base_url": "http://model/v1"}
        self.error = error
        self.calls = 0

    def _interruptible_api_call(self, api_kwargs: dict) -> dict:
        self.calls += 1
        raise self.error


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_retry_policy_turns_a_rejection_into_an_error_hermes_does_not_resend(status: int) -> None:
    error = _StatusError(status)
    agent = _Agent(error)
    install_model_retry_policy(agent)
    install_model_retry_policy(agent)  # installing twice wraps once

    with pytest.raises(ModelRequestRejected) as caught:
        agent._interruptible_api_call({"messages": []})

    # Hermes ends the run on a ValueError; Gym's classifier still reads the SDK's text and status.
    assert isinstance(caught.value, ValueError)
    assert str(caught.value) == str(error)
    assert caught.value.status_code == status
    assert caught.value.body == error.body and caught.value.response is error.response
    assert caught.value.__cause__ is error
    assert agent.calls == 1
    assert agent._client_kwargs["max_retries"] == 0


@pytest.mark.parametrize(
    "error",
    [
        _StatusError(408),
        _StatusError(413),
        _StatusError(425),
        _StatusError(429),
        _StatusError(500),
        _StatusError(503),
        ConnectionError("refused"),
    ],
)
def test_retry_policy_leaves_failures_that_may_clear_to_hermes(error: Exception) -> None:
    agent = _Agent(error)
    install_model_retry_policy(agent)

    with pytest.raises(type(error)) as caught:
        agent._interruptible_api_call({"messages": []})

    assert caught.value is error
