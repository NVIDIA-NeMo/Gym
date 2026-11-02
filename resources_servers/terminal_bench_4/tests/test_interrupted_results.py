# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An attempt cut short (a killed agent, a resources shutdown) is marked in its result, never read as graded."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.app import TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.models import AgentTermination


def _session(termination=None, *, phase="verifying", directory=Path("trial"), session_id="session-1", **extra):
    environment = SimpleNamespace(main=object(), closed=False, quiesce_agent=AsyncMock(), stop=AsyncMock())
    return SimpleNamespace(
        termination=termination,
        environment=environment,
        result={},
        diagnostics=[],
        phase=phase,
        subphase=None,
        persist=lambda: None,
        directory=directory,
        session_id=session_id,
        **extra,
    )


async def test_a_killed_agent_marks_its_zero_as_interrupted(monkeypatch):
    for step in ("download_dir", "collect", "run_verifier", "cleanup"):
        monkeypatch.setattr(lifecycle, step, AsyncMock())
    session = _session(AgentTermination(reason="killed", exit_code=9, detail="mini-SWE command exited 9: "))

    await lifecycle.finalize_session(session, grade=True)

    # The attempt still scores its zero; the mark says no later reader may reuse it as a graded result.
    assert session.result["verifier_result"]["source"] == "agent_killed"
    assert session.result["interrupted"]["reason"] == "agent_killed"


async def test_a_shutdown_during_evaluation_marks_the_result_interrupted(monkeypatch):
    seen = {}

    async def cleanup(session):
        seen.update(session.result)

    monkeypatch.setattr(lifecycle, "download_dir", AsyncMock(side_effect=asyncio.CancelledError))
    monkeypatch.setattr(lifecycle, "cleanup", cleanup)
    session = _session(AgentTermination(reason="completed"))

    await lifecycle.finalize_session(session, grade=True)

    assert seen["exception_info"]["exception_type"] == "CancelledError"
    # Marked before cleanup, which persists result.json and the state record.
    assert seen["interrupted"]["reason"] == "resources_shutdown"
    assert seen["interrupted"]["subphase"] == "collect"


async def test_a_shutdown_with_an_attempt_in_flight_marks_it_and_leaves_a_closed_one(monkeypatch):
    persisted = {}

    async def cleanup(session):
        persisted[session.session_id] = dict(session.result)
        session.phase = "closed"

    monkeypatch.setattr(lifecycle, "cleanup", cleanup)
    in_flight = _session(phase="ready", session_id="running", expiry_task=None, execution=None, finalization=None)
    closed = _session(phase="closed", session_id="done", expiry_task=None, execution=None, finalization=None)

    await lifecycle.shutdown([in_flight, closed], timeout=0)

    assert persisted["running"]["interrupted"] == {
        "reason": "resources_shutdown",
        "phase": "ready",
        "subphase": None,
        "at": persisted["running"]["interrupted"]["at"],
    }
    assert "done" not in persisted and "interrupted" not in closed.result


async def test_a_preparation_cancelled_by_shutdown_is_marked_interrupted(monkeypatch):
    async def cancelled(session, loader):
        raise asyncio.CancelledError

    monkeypatch.setattr(lifecycle, "prepare_session", cancelled)
    monkeypatch.setattr(lifecycle, "cleanup", AsyncMock())
    session = _session(
        session_id="session-1", started=asyncio.Event(), request=SimpleNamespace(capture_rollout_id=None)
    )

    await TerminalBench4ResourcesServer._prepare_session(SimpleNamespace(_loader=None), session)

    assert session.termination.reason == "cancelled"
    assert session.result["interrupted"]["reason"] == "resources_shutdown"


async def test_a_completed_evaluation_is_not_marked(monkeypatch):
    for step in ("download_dir", "collect", "restore", "cleanup"):
        monkeypatch.setattr(lifecycle, step, AsyncMock())
    monkeypatch.setattr(lifecycle, "run_verifier", AsyncMock(return_value={"rewards": {"reward": 1.0}}))
    config = SimpleNamespace(
        environment=SimpleNamespace(verifier_timeout_sec=lambda task: 60),
    )
    task = SimpleNamespace(
        config=SimpleNamespace(
            environment=SimpleNamespace(build_timeout_sec=60), verifier=SimpleNamespace(timeout_sec=60)
        )
    )
    session = _session(
        AgentTermination(reason="completed"),
        shared_logs=None,
        verifier_environment=SimpleNamespace(start=AsyncMock()),
        config=config,
        task=task,
    )

    await lifecycle.finalize_session(session, grade=True)

    assert session.result["verifier_result"] == {"rewards": {"reward": 1.0}}
    assert "interrupted" not in session.result
