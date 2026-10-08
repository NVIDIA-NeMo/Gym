# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial-rollout checkpoint spans and metrics, against a real in-memory exporter and metric reader."""

import asyncio
import time
from pathlib import Path

import pytest

from nemo_gym._checkpoint import coordination
from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry import setup as telemetry_setup
from nemo_gym.telemetry.span_groups import GymSpanGroup
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
    monkeypatch.setattr(lens_helpers.trace, "get_tracer", lambda *a, **k: tracer)
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


async def test_a_refusal_and_a_failed_operation_are_counted(tmp_path: Path, metrics) -> None:
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
    events = {
        (attributes["nemo.gym.checkpoint.event"], attributes.get("nemo.gym.checkpoint.code")): value
        for attributes, value in recorded[gym_metrics.CHECKPOINT_EVENTS_INSTRUMENT]
    }
    outcomes = {
        (attributes["nemo.gym.checkpoint.operation"], attributes["outcome"])
        for attributes, _ in recorded[gym_metrics.CHECKPOINT_OPERATION_INSTRUMENT]
    }
    assert refused.status_code == 409 and failed.status_code == 409
    assert events[("refused", "stale_attempt")] == 1
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
            coordination.Participant("env", "environment"),
            coordination.Participant("policy", "model"),
            coordination.Participant("agent", "agent"),
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
