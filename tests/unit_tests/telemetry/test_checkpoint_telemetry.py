# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial-rollout checkpoint spans and metrics, against a real in-memory exporter and metric reader."""

import asyncio
import time
from pathlib import Path

import pytest

from nemo_gym._checkpoint import coordination
from nemo_gym._checkpoint.store import participant_dir, read_participant_state, write_participant_state
from nemo_gym.telemetry import GymSpanGroup, gym_metrics
from nemo_gym.telemetry import setup as telemetry_setup
from tests.unit_tests.test_checkpoint_control import FakeParticipant, body, make_client
from tests.unit_tests.test_checkpoint_resources import AUTH, SEED, control, make_server


pytest.importorskip("opentelemetry.sdk.metrics")
pytest.importorskip("nemo.lens")


@pytest.fixture
def spans(monkeypatch):
    import nemo.lens.helpers as lens_helpers
    from nemo.lens.state import set_enabled_span_groups
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(lens_helpers, "get_tracer", lambda *a, **k: tracer)
    monkeypatch.setattr("opentelemetry.trace.get_tracer", lambda *a, **k: tracer)
    set_enabled_span_groups(GymSpanGroup.resolve("default,checkpoint"))
    yield exporter.get_finished_spans
    set_enabled_span_groups(frozenset())


@pytest.fixture
def metrics(monkeypatch):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    gym_metrics._reset_for_testing()

    def collect() -> dict[str, list[tuple[dict, float]]]:
        data = reader.get_metrics_data()
        out: dict[str, list[tuple[dict, float]]] = {}
        for resource_metric in data.resource_metrics if data is not None else ():
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    for point in metric.data.data_points:
                        value = getattr(point, "value", None)
                        if value is None:
                            value = point.count
                        out.setdefault(metric.name, []).append((dict(point.attributes), value))
        return out

    yield collect
    gym_metrics._reset_for_testing()


def by_name(finished) -> dict[str, object]:
    return {span.name: span for span in finished}


def points(recorded: dict, instrument: str, *names: str) -> dict[tuple, float]:
    """Each data point of ``instrument`` keyed by the values of the attributes ``names``."""
    return {tuple(attributes.get(name) for name in names): value for attributes, value in recorded.get(instrument, [])}


async def checkpoint_and_restore(tmp_path: Path) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/increment")
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        cookies = dict(client.cookies)
    _, fresh = make_server()
    async with fresh:
        fresh.cookies.update(cookies)
        restored = await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        assert restored.status_code == 200, restored.text


async def test_each_operation_has_a_span_with_its_slow_steps_nested_under_it(tmp_path: Path, spans) -> None:
    await checkpoint_and_restore(tmp_path)

    finished = by_name(spans())
    commit, restore = finished["gym.checkpoint.commit"], finished["gym.checkpoint.restore"]
    assert finished["gym.checkpoint.wait_ready"].parent.span_id == finished["gym.checkpoint.prepare"].context.span_id
    assert finished["gym.checkpoint.export"].parent.span_id == commit.context.span_id
    assert finished["gym.checkpoint.write"].parent.span_id == commit.context.span_id
    assert finished["gym.checkpoint.read"].parent.span_id == restore.context.span_id
    assert finished["gym.checkpoint.install"].parent.span_id == restore.context.span_id
    assert commit.attributes["nemo.gym.checkpoint.participant_kind"] == "resources"
    assert commit.attributes["nemo.gym.checkpoint.id"] == "c1"
    assert commit.attributes["nemo.gym.checkpoint.records"] == 1
    assert finished["gym.checkpoint.write"].attributes["nemo.gym.checkpoint.bytes"] > 0


async def test_operations_record_duration_records_and_bytes(tmp_path: Path, metrics) -> None:
    await checkpoint_and_restore(tmp_path)

    recorded = metrics()
    operations = {
        (attributes["nemo.gym.checkpoint.operation"], attributes["outcome"])
        for attributes, _ in recorded[gym_metrics.CHECKPOINT_OPERATION_INSTRUMENT]
    }
    records = {
        attributes["nemo.gym.checkpoint.operation"]: value
        for attributes, value in recorded[gym_metrics.CHECKPOINT_RECORDS_INSTRUMENT]
    }
    assert {("prepare", "ok"), ("commit", "ok"), ("restore", "ok")} <= operations
    assert records == {"commit": 1, "restore": 1}
    assert all(value > 0 for _, value in recorded[gym_metrics.CHECKPOINT_BYTES_INSTRUMENT])


async def test_a_failed_control_call_is_counted_and_a_refused_data_request_is_not(tmp_path: Path, metrics) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post(
            "/ng-control/v1/checkpoint/retire",
            json=control("retire", episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        # The session belongs to a retired attempt, so its late request is refused.
        refused = await client.post("/increment")
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        # Restoring a participant that is already in a checkpoint is the wrong phase.
        failed = await client.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("c1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )

    recorded = metrics()
    codes = points(recorded, gym_metrics.CHECKPOINT_CONTROL_ERRORS_INSTRUMENT, "nemo.gym.checkpoint.error_code")
    outcomes = points(
        recorded, gym_metrics.CHECKPOINT_OPERATION_INSTRUMENT, "nemo.gym.checkpoint.operation", "outcome"
    )
    assert (
        refused.json()["error"]["code"] == "checkpoint_stale_attempt"
        and failed.json()["error"]["code"] == "checkpoint_invalid_phase"
    )
    assert codes == {("checkpoint_invalid_phase",): 1}
    assert ("restore", "error") in outcomes


async def test_a_lease_that_expires_is_counted(metrics) -> None:
    participant = FakeParticipant()
    async with make_client(participant, lease_grace=0.1) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=0.05))
        await asyncio.sleep(0.4)

    events = {
        attributes["nemo.gym.checkpoint.event"]: value
        for attributes, value in metrics().get(gym_metrics.CHECKPOINT_EVENTS_INSTRUMENT, [])
    }
    assert events.get("lease_expired") == 1


async def test_coordination_spans_each_prepare_stage(monkeypatch, spans) -> None:
    async def prepared(participants, members, operation, body, deadline_ts):
        return {member.server_name: {"phase": "prepared"} for member in members}

    monkeypatch.setattr(coordination, "_fan_out", prepared)
    participants = coordination.Participants(
        client=None,
        auth_token="t",
        members=(
            coordination.Participant("env", "environment", "exported"),
            coordination.Participant("policy", "model", "exported"),
            coordination.Participant("agent", "agent", "exported"),
        ),
    )

    result = await coordination.prepare(participants, "c1", deadline_ts=time.time() + 5)

    finished = by_name(spans())
    top = finished["gym.checkpoint.coordinate.prepare"]
    stages = [name for name in finished if name.startswith("gym.checkpoint.coordinate.prepare.")]
    assert result.prepared
    assert sorted(stages) == [
        f"gym.checkpoint.coordinate.prepare.{kind}" for kind in sorted(coordination.PREPARE_ORDER)
    ]
    assert all(finished[name].parent.span_id == top.context.span_id for name in stages)
    assert top.attributes["nemo.gym.checkpoint.participant_kind"] == "controller"


async def test_retire_and_forget_each_have_a_span(spans) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post(
            "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/forget", json=control("forget", rollout_ids=["r"]), headers=AUTH)

    finished = by_name(spans())
    assert finished["gym.checkpoint.retire"].attributes["nemo.gym.checkpoint.episodes"] == 1
    assert finished["gym.checkpoint.forget"].attributes["nemo.gym.checkpoint.rollouts"] == 1


async def test_a_resume_that_fails_on_a_participant_records_an_error_and_ends_its_span_with_it(
    monkeypatch, spans, metrics
) -> None:
    async def one_stage_fails(participants, members, operation, body, deadline_ts):
        if any(member.kind == "agent" for member in members):
            raise coordination.CoordinationError(operation, {"agent": "unreachable"})
        return {member.server_name: {"phase": "idle"} for member in members}

    monkeypatch.setattr(coordination, "_fan_out", one_stage_fails)
    participants = coordination.Participants(
        client=None,
        auth_token="t",
        members=(
            coordination.Participant("env", "environment", "exported"),
            coordination.Participant("agent", "agent", "exported"),
        ),
    )

    with pytest.raises(coordination.CoordinationError):
        await coordination.resume(participants, "c1", deadline_ts=time.time() + 5)

    from opentelemetry.trace import StatusCode

    resume = by_name(spans())["gym.checkpoint.coordinate.resume"]
    outcomes = points(
        metrics(), gym_metrics.CHECKPOINT_OPERATION_INSTRUMENT, "nemo.gym.checkpoint.operation", "outcome"
    )
    assert resume.status.status_code == StatusCode.ERROR
    assert outcomes == {("coordinate.resume", "error"): 1}


async def test_a_lease_expiry_is_counted_once_and_each_reopen_attempt_with_its_outcome(
    tmp_path: Path, monkeypatch, metrics
) -> None:
    import nemo_gym._checkpoint.control as control_module
    from nemo_gym.episode_types import EpisodeId
    from tests.unit_tests.test_checkpoint_control import _past_deadline, _until

    monkeypatch.setattr(control_module, "_LEASE_REOPEN_WAIT_SECONDS", 0.05)
    release = asyncio.Event()
    deleting: list[None] = []
    reopens: list[None] = []
    real_reopen = control_module.ParticipantControlPlane._reopen

    async def counted_reopen(self, **kwargs):
        reopens.append(None)
        return await real_reopen(self, **kwargs)

    monkeypatch.setattr(control_module.ParticipantControlPlane, "_reopen", counted_reopen)

    class SlowCleanup(FakeParticipant):
        async def delete_restored(self, episode_id: EpisodeId) -> None:
            deleting.append(None)
            await release.wait()
            await super().delete_restored(episode_id)

    participant = SlowCleanup()
    participant.executions["r-a1"] = {"parked": True, "value": 1, "restored": True}
    async with make_client(participant, lease_grace=0.2) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        # The cleanup of the restored state the scope leaves out is held,
        # so every reopen after the lease expires runs out of time waiting for it.
        await _past_deadline(
            client, "commit", lambda: deleting, "the cleanup started", checkpoint_dir=str(tmp_path), episode_ids=[]
        )
        await _until(lambda: len(reopens) >= 3, "the lease recovery tried to reopen three times")
        release.set()
        await _until(lambda: participant.accepting, "the lease recovery reopened admission")

    recorded = metrics()
    events = points(recorded, gym_metrics.CHECKPOINT_EVENTS_INSTRUMENT, "nemo.gym.checkpoint.event")
    reopened = points(
        recorded, gym_metrics.CHECKPOINT_OPERATION_INSTRUMENT, "nemo.gym.checkpoint.operation", "outcome"
    )
    assert events[("lease_expired",)] == 1
    assert reopened[("lease_reopen", "error")] >= 2
    assert reopened[("lease_reopen", "ok")] == 1


async def test_a_commit_retried_after_its_cleanup_ran_out_of_time_counts_its_records_once(
    tmp_path: Path, metrics
) -> None:
    from nemo_gym.episode_types import EpisodeId
    from tests.unit_tests.test_checkpoint_control import _past_deadline

    release = asyncio.Event()
    deleting: list[None] = []

    class SlowCleanup(FakeParticipant):
        async def delete_restored(self, episode_id: EpisodeId) -> None:
            deleting.append(None)
            await release.wait()
            await super().delete_restored(episode_id)

    participant = SlowCleanup()
    participant.executions["r-a1"] = {"parked": True, "value": 1, "restored": True}
    participant.executions["live"] = {"parked": True, "value": 2}
    commit = {"checkpoint_dir": str(tmp_path), "episode_ids": [{"rollout_id": "live"}]}
    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        # The write finished; the cleanup after it did not.
        await _past_deadline(client, "commit", lambda: deleting, "the cleanup started", **commit)
        release.set()
        retried = await client.post("/ng-control/v1/checkpoint/commit", json=body(**commit))

    recorded = metrics()
    size = (
        participant_dir(tmp_path, kind="fake", instance="fake-1") / retried.json()["manifest"]["records_file"]
    ).stat()
    assert retried.status_code == 200
    assert points(recorded, gym_metrics.CHECKPOINT_RECORDS_INSTRUMENT, "nemo.gym.checkpoint.operation") == {
        ("commit",): 1
    }
    assert points(recorded, gym_metrics.CHECKPOINT_BYTES_INSTRUMENT, "nemo.gym.checkpoint.operation") == {
        ("commit",): size.st_size
    }


async def test_a_restore_retried_after_its_install_failed_counts_its_records_once(tmp_path: Path, metrics) -> None:
    write_participant_state(
        tmp_path,
        kind="fake",
        instance="fake-1",
        checkpoint_id="c1",
        records=[{"episode_id": {"rollout_id": "r"}, "value": 1}],
    )

    class InstallFailsOnce(FakeParticipant):
        failed = False

        async def install(self, records, scope) -> None:
            if not self.failed:
                self.failed = True
                raise RuntimeError("install failed")
            await super().install(records, scope)

    participant = InstallFailsOnce()
    restore = {"checkpoint_dir": str(tmp_path), "episode_ids": [{"rollout_id": "r"}]}
    async with make_client(participant) as client:
        with pytest.raises(RuntimeError):
            await client.post("/ng-control/v1/checkpoint/restore", json=body("r1", **restore))
        retried = await client.post("/ng-control/v1/checkpoint/restore", json=body("r1", **restore))

    recorded = metrics()
    assert retried.status_code == 200 and retried.json()["restored"] == ["r-a1"]
    assert points(recorded, gym_metrics.CHECKPOINT_RECORDS_INSTRUMENT, "nemo.gym.checkpoint.operation") == {
        ("restore",): 1
    }


def test_the_store_returns_the_bytes_it_wrote_and_read(tmp_path: Path) -> None:
    records = [{"episode_id": {"rollout_id": f"r{index}"}, "value": index} for index in range(3)]

    manifest, written = write_participant_state(
        tmp_path, kind="fake", instance="f", checkpoint_id="c1", records=records
    )
    again, rewritten = write_participant_state(
        tmp_path, kind="fake", instance="f", checkpoint_id="c1", records=records
    )
    _, kept, read = read_participant_state(
        tmp_path, kind="fake", instance="f", select=lambda record: record if record["value"] == 0 else None
    )

    size = (participant_dir(tmp_path, kind="fake", instance="f") / manifest["records_file"]).stat().st_size
    assert again == manifest
    # The bytes cover the whole records file, whatever ``select`` keeps.
    assert written == rewritten == read == size
    assert len(kept) == 1


async def test_commit_and_restore_measure_their_records_without_a_stat_on_the_event_loop(
    tmp_path: Path, monkeypatch, metrics
) -> None:
    import threading

    loop_thread = threading.get_ident()
    stats_on_loop: list[str] = []
    real_stat = Path.stat

    def watched_stat(self, *args, **kwargs):
        if self.name.startswith("records") and threading.get_ident() == loop_thread:
            stats_on_loop.append(self.name)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", watched_stat)

    await checkpoint_and_restore(tmp_path)

    sizes = points(metrics(), gym_metrics.CHECKPOINT_BYTES_INSTRUMENT, "nemo.gym.checkpoint.operation")
    [records_file] = participant_dir(tmp_path, kind="resources", instance="counter").glob("records-*.jsonl")
    assert stats_on_loop == []
    assert sizes == {("commit",): real_stat(records_file).st_size, ("restore",): real_stat(records_file).st_size}


async def test_a_prepare_span_counts_every_blocker_not_only_the_listed_ones(spans) -> None:
    from nemo_gym._checkpoint.control import MAX_REPORTED_BLOCKERS

    participant = FakeParticipant()
    for index in range(MAX_REPORTED_BLOCKERS + 50):
        participant.executions[f"r{index}"] = {"parked": False, "value": index}
    async with make_client(participant) as client:
        missed = (await client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=0.1))).json()

    prepare = by_name(spans())["gym.checkpoint.prepare"]
    assert missed["report"]["blocker_count"] == MAX_REPORTED_BLOCKERS + 50
    assert prepare.attributes["nemo.gym.checkpoint.blockers"] == MAX_REPORTED_BLOCKERS + 50


async def test_each_failed_control_call_counts_its_error_code_once(tmp_path: Path, metrics) -> None:
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": False, "value": 1}
    async with make_client(participant) as client:
        # Commit before prepare is the wrong phase.
        wrong_phase = await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(tmp_path)))
        unauthorized = await client.post(
            "/ng-control/v1/checkpoint/prepare", json=body(), headers={"authorization": "Bearer wrong"}
        )
        late = await client.post(
            "/ng-control/v1/checkpoint/retire",
            json={"checkpoint_id": "retire", "deadline_ts": time.time() - 1, "episode_ids": [{"rollout_id": "r"}]},
        )
        succeeded = await client.post("/ng-control/v1/checkpoint/prepare", json=body(timeout=0.1))

    codes = points(metrics(), gym_metrics.CHECKPOINT_CONTROL_ERRORS_INSTRUMENT, "nemo.gym.checkpoint.error_code")
    assert [wrong_phase.status_code, unauthorized.status_code, late.status_code] == [409, 401, 409]
    assert succeeded.status_code == 200
    assert codes == {
        ("checkpoint_invalid_phase",): 1,
        ("checkpoint_unauthorized",): 1,
        ("checkpoint_deadline_exceeded",): 1,
    }


def test_building_an_error_reply_records_nothing(metrics) -> None:
    from nemo_gym._checkpoint.errors import InvalidPhaseError

    error = InvalidPhaseError("not valid in phase idle")
    first, second = error.response(), error.response()

    assert first.body == second.body and first.status_code == 409
    assert metrics() == {}
