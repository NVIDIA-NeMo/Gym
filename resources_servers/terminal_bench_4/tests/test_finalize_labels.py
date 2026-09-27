# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A killed run is a termination, not the reported failure: whatever breaks the evaluation afterwards is."""

from types import SimpleNamespace

import pytest

from resources_servers.terminal_bench_4 import lifecycle


class FakeEnvironment:
    def __init__(self):
        self.main = object()
        self.closed = False
        self.quiesced = []

    async def quiesce_agent(self, session_id):
        self.quiesced.append(session_id)

    async def stop(self):
        self.closed = True


def make_session(tmp_path, reason="timeout", detail="OpenCode run exceeded its 21600 s per-task budget"):
    return SimpleNamespace(
        session_id="tb4-test",
        phase="verifying",
        subphase=None,
        result={},
        diagnostics=[],
        termination=SimpleNamespace(reason=reason, detail=detail),
        environment=FakeEnvironment(),
        verifier_environment=None,
        shared_logs=None,
        archive_workers=None,
        directory=tmp_path,
        persist=lambda: None,
    )


@pytest.fixture
def quiet_cleanup(monkeypatch):
    async def no_cleanup(session):
        session.subphase = "cleanup"

    async def no_logs(*args, **kwargs):
        return None

    monkeypatch.setattr(lifecycle, "cleanup", no_cleanup)
    monkeypatch.setattr(lifecycle, "download_dir", no_logs)


async def test_collect_failure_after_a_budget_kill_is_the_reported_failure(tmp_path, monkeypatch, quiet_cleanup):
    async def failing_collect(*args, **kwargs):
        raise RuntimeError("Timed out during OpenSandbox read_file(/tmp/download.tar.gz) after 300s")

    monkeypatch.setattr(lifecycle, "collect", failing_collect)
    session = make_session(tmp_path)
    await lifecycle.finalize_session(session, grade=True)
    assert session.result["exception_info"]["exception_type"] == "RuntimeError"
    assert "read_file" in session.result["exception_info"]["exception_message"]
    notes = [d for d in session.diagnostics if d.get("exception_type") == "AgentTimeoutError"]
    assert len(notes) == 1 and notes[0]["subphase"] == "quiesce"  # the kill stays visible in the diagnostics


async def test_a_clean_budget_kill_records_no_failure(tmp_path, monkeypatch, quiet_cleanup):
    async def ok_collect(*args, **kwargs):
        return None

    class Verifier:
        async def start(self):
            return None

    async def ok_restore(*args, **kwargs):
        return None

    async def ok_verifier(*args, **kwargs):
        return {"rewards": {"reward": 0.0}}

    monkeypatch.setattr(lifecycle, "collect", ok_collect)
    monkeypatch.setattr(lifecycle, "restore", ok_restore)
    monkeypatch.setattr(lifecycle, "run_verifier", ok_verifier)
    session = make_session(tmp_path)
    session.verifier_environment = Verifier()
    session.task = SimpleNamespace(config=SimpleNamespace(environment=SimpleNamespace(build_timeout_sec=5)))
    await lifecycle.finalize_session(session, grade=True)
    assert "exception_info" not in session.result
    assert session.result["verifier_result"] == {"rewards": {"reward": 0.0}}
    assert any(d.get("exception_type") == "AgentTimeoutError" for d in session.diagnostics)


async def test_non_zero_exit_is_noted_the_same_way(tmp_path, monkeypatch, quiet_cleanup):
    async def failing_collect(*args, **kwargs):
        raise ValueError("artifact metadata is missing")

    monkeypatch.setattr(lifecycle, "collect", failing_collect)
    session = make_session(tmp_path, reason="nonzero_exit", detail="OpenCode run exited 2")
    await lifecycle.finalize_session(session, grade=True)
    assert session.result["exception_info"]["exception_type"] == "ValueError"
    assert [d["exception_type"] for d in session.diagnostics if "exception_type" in d] == [
        "NonZeroAgentExitCodeError",
        "ValueError",
    ]
