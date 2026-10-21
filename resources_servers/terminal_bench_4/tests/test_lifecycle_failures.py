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
