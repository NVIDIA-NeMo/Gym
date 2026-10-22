# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a trial's failures are named for evaluation's retry rule."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.rollout_collection import is_terminal_failure
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.models import AgentTermination


class _Env:
    def __init__(self, start_delay: float = 0.0) -> None:
        self.main = object()
        self.closed = False
        self.start_delay = start_delay

    async def quiesce_agent(self, session_id):
        return None

    async def stop(self):
        self.closed = True

    async def start(self):
        await asyncio.sleep(self.start_delay)


@pytest.mark.parametrize("eval_mode", [False, True], ids=["training", "evaluation"])
async def test_a_verifier_sandbox_that_misses_its_start_deadline_is_named_in_evaluation(
    tmp_path, monkeypatch, eval_mode
):
    """Terminal-Bench retries a trial whose verifier sandbox did not start in time; evaluation names it so."""
    monkeypatch.setattr(lifecycle, "is_eval_mode", lambda: eval_mode)
    monkeypatch.setattr(lifecycle, "download_dir", AsyncMock())
    monkeypatch.setattr(lifecycle, "collect", AsyncMock())
    session = SimpleNamespace(
        termination=AgentTermination(reason="completed"),
        environment=_Env(),
        verifier_environment=_Env(start_delay=1.0),
        shared_logs=None,
        session_id="session-1",
        directory=tmp_path,
        diagnostics=[],
        result={},
        subphase=None,
        phase="verifying",
        expiry_task=None,
        owns_slot=False,
        task=SimpleNamespace(config=SimpleNamespace(environment=SimpleNamespace(build_timeout_sec=0.01))),
        persist=lambda: None,
    )

    await lifecycle.finalize_session(session, grade=True)

    exception_type = session.result["exception_info"]["exception_type"]
    assert exception_type == ("VerifierStartTimeoutError" if eval_mode else "TimeoutError")
    row = {"_ng_failure_class": "infrastructure_error", "_ng_failure_type": exception_type}
    # The rule itself runs only in evaluation scoring.
    assert is_terminal_failure(row) is not eval_mode


class _FailingStartEnv(_Env):
    async def start(self):
        raise RuntimeError("SandboxApiException: create rejected (401 invalid api key)")


@pytest.mark.parametrize("eval_mode", [False, True], ids=["training", "evaluation"])
async def test_a_sandbox_that_could_not_be_created_is_named_as_a_create_failure(tmp_path, monkeypatch, eval_mode):
    """A create or start failure comes before the agent's first command: evaluation names it so it is retried."""
    monkeypatch.setattr(lifecycle, "is_eval_mode", lambda: eval_mode)

    class _Loader:
        async def load(self, name, ref):
            return SimpleNamespace(config=SimpleNamespace(environment=SimpleNamespace(build_timeout_sec=5)))

    session = SimpleNamespace(
        directory=tmp_path,
        result={},
        diagnostics=[],
        phase="preparing",
        subphase=None,
        slots=asyncio.Semaphore(1),
        owns_slot=False,
        request=SimpleNamespace(task_name="t", task_ref=None),
        config=SimpleNamespace(environment=SimpleNamespace(efs_logs_host_path=None)),
        session_id="s",
        shared_logs=None,
        persist=lambda: None,
    )
    monkeypatch.setattr(lifecycle, "Environment", lambda *args, **kwargs: _FailingStartEnv())

    with pytest.raises(RuntimeError):
        await lifecycle.prepare_session(session, _Loader())

    if not eval_mode:
        # Training keeps its behaviour: the seed handler records the raw exception type afterwards.
        assert "exception_info" not in session.result
        return
    exception_type = session.result["exception_info"]["exception_type"]
    assert exception_type == "SandboxCreateError"
    row = {"_ng_failure_class": "infrastructure_error", "_ng_failure_type": exception_type}
    assert not is_terminal_failure(row)


class _FailingVerifierStartEnv(_Env):
    async def start(self):
        raise RuntimeError("SandboxApiException: create rejected")


async def test_a_verifier_sandbox_that_could_not_be_created_is_named_as_a_create_failure(tmp_path, monkeypatch):
    """The verifier sandbox never started, so it ran none of the model's work: retried as infra."""
    monkeypatch.setattr(lifecycle, "is_eval_mode", lambda: True)
    monkeypatch.setattr(lifecycle, "download_dir", AsyncMock())
    monkeypatch.setattr(lifecycle, "collect", AsyncMock())
    session = SimpleNamespace(
        termination=AgentTermination(reason="completed"),
        environment=_Env(),
        verifier_environment=_FailingVerifierStartEnv(),
        shared_logs=None,
        session_id="session-1",
        directory=tmp_path,
        diagnostics=[],
        result={},
        subphase=None,
        phase="verifying",
        expiry_task=None,
        owns_slot=False,
        task=SimpleNamespace(config=SimpleNamespace(environment=SimpleNamespace(build_timeout_sec=5))),
        persist=lambda: None,
    )

    await lifecycle.finalize_session(session, grade=True)

    exception_type = session.result["exception_info"]["exception_type"]
    assert exception_type == "SandboxCreateError"
    assert not is_terminal_failure({"_ng_failure_class": "infrastructure_error", "_ng_failure_type": exception_type})
