# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only, bounded scheduling diagnostics for slow participant prepares."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from pydantic import JsonValue

from nemo_gym._checkpoint import control as control_module
from nemo_gym._checkpoint.agent import AgentSessionParticipant, RestoredAgentSession
from nemo_gym._checkpoint.control import (
    MAX_PREPARE_DIAGNOSTIC_BLOCKERS,
    CheckpointPhase,
    CheckpointRequest,
    ParticipantControlPlane,
)
from nemo_gym._checkpoint.diagnostics import task_wait_diagnostics
from nemo_gym._checkpoint.steps import EpisodeSteps
from nemo_gym.episode_types import EpisodeId


class Hooks:
    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        raise AssertionError("diagnostics must not export agent sessions")

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        raise AssertionError("diagnostics must not restore agent sessions")

    async def retire_agent_session(self, session_key: str) -> None:
        raise AssertionError("diagnostics must not retire agent sessions")


def participant_control() -> tuple[AgentSessionParticipant, ParticipantControlPlane]:
    participant = AgentSessionParticipant(Hooks())
    controller = ParticipantControlPlane(participant, instance_name="lc_judge_simple_agent", lease_grace_seconds=60)
    return participant, controller


def reports(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        json.loads(record.message.split("checkpoint_prepare_wait ", 1)[1])
        for record in caplog.records
        if record.message.startswith("checkpoint_prepare_wait {")
    ]


async def test_wait_step_reports_verify_mode_without_payload_or_cookie() -> None:
    participant, controller = participant_control()
    episode_id = EpisodeId(rollout_id="r", attempt=2)
    async with participant.legacy_run("COOKIE_SECRET", episode_id) as run:
        await run.boundary(
            {
                "next": "verify",
                "verify_mode": "wait",
                "response": "MODEL_OUTPUT_SECRET",
                "cookies": {"x": "AUTH_SECRET"},
            }
        )
        async with run.step("wait"):
            await participant.close_admission(CheckpointRequest(checkpoint_id="c", deadline_ts=time.time() + 60))
            before = participant.readiness()
            status = controller.status()
            [blocker] = status["prepare_diagnostics"]["blockers"]
            assert blocker["capture_key"] == episode_id.capture_key
            [session] = blocker["sessions"]
            assert session["state"] == "idle"
            legacy = session["legacy_episode"]
            assert legacy["state"] == "wait_step"
            assert legacy["next"] == "verify" and legacy["verify_mode"] == "wait"
            assert legacy["blocking"] is True
            assert legacy["state_age_s"] >= 0
            serialized = json.dumps(status)
            for secret in ("COOKIE_SECRET", "MODEL_OUTPUT_SECRET", "AUTH_SECRET"):
                assert secret not in serialized
            assert participant.readiness() == before
            await participant.open_admission()


async def test_running_activation_reports_await_chain_and_cleanup_flags() -> None:
    participant, controller = participant_control()
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("COOKIE_SECRET", episode_id)
    entered, release = asyncio.Event(), asyncio.Event()

    async def active_work() -> None:
        private_payload = "PROMPT_SECRET"
        async with participant.activation("COOKIE_SECRET", episode_id, index=7):
            entered.set()
            await release.wait()
        assert private_payload

    task = asyncio.create_task(active_work(), name="TASK_NAME_SECRET")
    await entered.wait()
    try:
        await participant.close_admission(CheckpointRequest(checkpoint_id="c", deadline_ts=time.time() + 60))
        session = participant._sessions["COOKIE_SECRET"]
        session.state_since -= 12

        # A diagnostic must never evaluate a lazy export-boundary callback.
        def forbidden_boundary() -> dict[str, JsonValue]:
            raise AssertionError("must not evaluate boundary")

        session.boundary = forbidden_boundary
        session.closing = True
        session.retired = True
        [detail] = controller.status()["prepare_diagnostics"]["blockers"]
        [state] = detail["sessions"]
        assert state["state"] == "running" and state["state_age_s"] >= 12
        assert state["park_requested"] is True
        assert state["closing"] is True and state["retired"] is True
        assert state["activation_index"] == 7 and state["activation_done"] is False
        assert any(frame["function"].endswith("active_work") for frame in state["task"]["await_chain"])
        assert any(frame["function"] == "Event.wait" for frame in state["task"]["await_chain"])
        serialized = json.dumps(detail)
        for secret in ("COOKIE_SECRET", "PROMPT_SECRET", "TASK_NAME_SECRET"):
            assert secret not in serialized
    finally:
        release.set()
        await task
        await participant.open_admission()


async def test_diagnostics_distinguish_seed_without_registered_session() -> None:
    participant, controller = participant_control()
    async with participant.seeding("COOKIE_SECRET", EpisodeId(rollout_id="seeding")):
        [detail] = controller.status()["prepare_diagnostics"]["blockers"]
        assert detail["capture_key"] == "seeding"
        assert detail["seeding_count"] == 1
        assert detail["session_count"] == 0 and detail["sessions"] == []
    assert controller.status()["prepare_diagnostics"] == {}


async def test_episode_diagnostics_show_finishing_and_unknown_labels_are_not_logged() -> None:
    async def notify() -> None:
        pass

    steps = EpisodeSteps(notify)
    steps.begin("COOKIE_SECRET")
    await steps.boundary("COOKIE_SECRET", {"next": "PRIVATE_STAGE_SECRET", "verify_mode": "PRIVATE_MODE_SECRET"})
    steps.finishing("COOKIE_SECRET")
    diagnostic = steps.diagnostic_state("COOKIE_SECRET")
    assert diagnostic["finishing"] is True
    assert diagnostic["next"] is None and diagnostic["verify_mode"] is None
    assert "SECRET" not in json.dumps(diagnostic)
    await steps.end("COOKIE_SECRET")
    assert steps.diagnostic_state("COOKIE_SECRET") is None


async def test_blocker_and_per_episode_session_samples_are_capped() -> None:
    participant, controller = participant_control()
    for index in range(20):
        participant.open_session(f"COOKIE_{index}", EpisodeId(rollout_id=f"r{index:02}"))
        participant._sessions[f"COOKIE_{index}"].set_state("running")
    status = controller.status()
    assert status["report"]["blocker_count"] == 20
    assert len(status["prepare_diagnostics"]["blockers"]) == MAX_PREPARE_DIAGNOSTIC_BLOCKERS
    for index in range(3):
        participant.open_session(f"EXTRA_COOKIE_{index}", EpisodeId(rollout_id="r00"))
    [detail] = participant.prepare_wait_diagnostics(["r00"])["blockers"]
    assert detail["session_count"] == 4
    assert len(detail["sessions"]) == 2 and detail["sessions_truncated"] is True


@pytest.mark.parametrize("ends_ready", [True, False])
async def test_slow_wait_is_rate_limited_and_logs_final_outcome(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, ends_ready: bool
) -> None:
    participant, controller = participant_control()
    participant.open_session("cookie", EpisodeId(rollout_id="r"))
    session = participant._sessions["cookie"]
    session.set_state("running")
    request = CheckpointRequest(checkpoint_id="c", deadline_ts=1006.0)
    controller.checkpoint_id = request.checkpoint_id
    controller.phase = CheckpointPhase.PREPARING
    await participant.close_admission(request)
    clock = {"now": 0.0, "notifications": 0}
    monkeypatch.setattr(
        control_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock["now"], time=lambda: 1000.0 + clock["now"]),
    )
    monkeypatch.setattr(control_module, "_PREPARE_DIAGNOSTIC_INITIAL_DELAY_S", 1.0)
    monkeypatch.setattr(control_module, "_PREPARE_DIAGNOSTIC_INTERVAL_S", 2.0)

    async def changed(timeout: float) -> None:
        clock["notifications"] += 1
        clock["now"] += 0.25
        if ends_ready and clock["now"] >= 5:
            session.set_state("idle")

    monkeypatch.setattr(participant, "wait_changed", changed)
    caplog.set_level("WARNING")
    await controller._wait_ready(request)
    emitted = reports(caplog)
    assert [record["event"] for record in emitted] == (
        ["waiting", "waiting", "ready"] if ends_ready else ["waiting", "waiting", "waiting", "deadline"]
    )
    assert [record["elapsed_s"] for record in emitted] == ([1, 3, 5] if ends_ready else [1, 3, 5, 6])
    assert clock["notifications"] >= 20
    assert emitted[-1]["blocker_count"] == (0 if ends_ready else 1)
    assert controller.phase == CheckpointPhase.PREPARING and participant.accepting is False
    if not ends_ready:
        [detail] = emitted[-1]["details"]["blockers"]
        assert detail["sessions"][0]["state"] == "running"


async def test_fast_ready_wait_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    _, controller = participant_control()
    request = CheckpointRequest(checkpoint_id="c", deadline_ts=time.time() + 60)
    controller.checkpoint_id = "c"
    await controller._wait_ready(request)
    assert reports(caplog) == []


async def test_expired_short_wait_records_blocker_before_return(caplog: pytest.LogCaptureFixture) -> None:
    participant, controller = participant_control()
    participant.open_session("cookie", EpisodeId(rollout_id="r"))
    participant._sessions["cookie"].set_state("running")
    request = CheckpointRequest(checkpoint_id="c", deadline_ts=time.time() - 1)
    controller.checkpoint_id = "c"
    await controller._wait_ready(request)
    [record] = reports(caplog)
    assert record["event"] == "deadline" and record["blockers"] == ["r"]
    assert participant.readiness().ready is False


async def test_diagnostic_failure_does_not_change_status_or_prepare(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    participant, controller = participant_control()
    participant.open_session("cookie", EpisodeId(rollout_id="r"))
    participant._sessions["cookie"].set_state("running")

    def broken(blockers: list[str]) -> dict[str, JsonValue]:
        raise ValueError("SECRET_EXCEPTION_TEXT")

    monkeypatch.setattr(participant, "prepare_wait_diagnostics", broken)
    assert controller.status()["prepare_diagnostics"] == {"diagnostic_error": "ValueError"}
    request = CheckpointRequest(checkpoint_id="c", deadline_ts=time.time() - 1)
    controller.checkpoint_id = "c"
    await controller._wait_ready(request)
    [record] = reports(caplog)
    assert record["details"] == {"diagnostic_error": "ValueError"}
    assert "SECRET_EXCEPTION_TEXT" not in caplog.text
    assert participant.readiness().blockers == ["r"]


async def test_task_locations_are_bounded_and_cancelled_task_is_safe() -> None:
    release, entered = asyncio.Event(), asyncio.Event()

    async def nested(depth: int) -> None:
        secret = "FRAME_LOCAL_SECRET"
        if depth:
            await nested(depth - 1)
        else:
            entered.set()
            await release.wait()
        assert secret

    task = asyncio.create_task(nested(30), name="TASK_NAME_SECRET")
    await entered.wait()
    try:
        result = task_wait_diagnostics(task)
        assert len(result["await_chain"]) == 12
        assert result["chain_truncated"] is True
        assert result["scan_truncated"] is False
        assert result["await_chain"][-1]["function"] == "Event.wait"
        assert "SECRET" not in json.dumps(result)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    result = task_wait_diagnostics(task)
    assert result["done"] is True and result["cancelled"] is True
    assert task_wait_diagnostics(None) == {"present": False}


async def test_state_ages_reset_only_on_actual_state_changes() -> None:
    participant, _ = participant_control()
    participant.open_session("s", EpisodeId(rollout_id="r"))
    session = participant._sessions["s"]
    session.state_since -= 20
    before = session.state_since
    session.set_state("idle")
    assert session.state_since == before
    session.set_state("running")
    assert session.state_since > before + 19

    steps = participant.legacy_episodes
    steps.begin("s")
    episode = steps._episodes["s"]
    episode.state_since -= 20
    before = episode.state_since
    steps._set_state(episode, "running")
    assert episode.state_since == before
    steps._set_state(episode, "wait_step")
    assert episode.state_since > before + 19
    await steps.end("s")
