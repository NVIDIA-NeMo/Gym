# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry import setup as telemetry_setup


@pytest.fixture
def collected(monkeypatch):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("run-summary-test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    gym_metrics._reset_for_testing()

    def collect():
        data = reader.get_metrics_data()
        return {
            metric.name: {frozenset(point.attributes.items()): point.value for point in metric.data.data_points}
            for resource_metric in (data.resource_metrics if data is not None else ())
            for scope_metric in resource_metric.scope_metrics
            for metric in scope_metric.metrics
        }

    yield collect
    provider.shutdown()
    gym_metrics._reset_for_testing()


def test_each_run_summary_key_becomes_a_gauge_under_the_run_id(collected) -> None:
    gym_metrics.record_run_summary(
        {"coverage/scored": 8, "retries/at_cap": 1, "coverage/failed/judge_failed": 2}, run_id="run-1"
    )

    metrics = collected()
    run = (gym_metrics.RUN_ID_ATTRIBUTE, "run-1")
    assert metrics["gym.run.coverage.scored"] == {frozenset({run}): 8}
    assert metrics["gym.run.retries.at_cap"] == {frozenset({run}): 1}
    assert metrics["gym.run.info"] == {frozenset({run}): 1}
    assert metrics["gym.run.coverage.failed"] == {
        frozenset({run, (gym_metrics.RUN_METRIC_BUCKET_ATTRIBUTE, "judge_failed")}): 2
    }


def test_a_merge_of_several_shards_is_recorded_without_a_run_id(collected) -> None:
    gym_metrics.record_run_summary({"coverage/scored": 3}, run_id=None)

    metrics = collected()
    assert metrics["gym.run.coverage.scored"] == {frozenset(): 3}
    assert metrics["gym.run.info"] == {frozenset(): 1}


def test_unregistered_failure_classes_share_one_bucket(collected) -> None:
    gym_metrics.record_run_summary(
        {"coverage/failed/judge_failed": 1, "coverage/failed/task 17 died": 2, "coverage/failed/1234": 3},
        run_id="run-1",
    )

    run = (gym_metrics.RUN_ID_ATTRIBUTE, "run-1")
    assert collected()["gym.run.coverage.failed"] == {
        frozenset({run, (gym_metrics.RUN_METRIC_BUCKET_ATTRIBUTE, "judge_failed")}): 1,
        frozenset({run, (gym_metrics.RUN_METRIC_BUCKET_ATTRIBUTE, "unregistered")}): 5,
    }


def test_nothing_is_recorded_without_a_metrics_exporter(collected, monkeypatch) -> None:
    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    gym_metrics.record_run_summary({"coverage/scored": 8}, run_id="run-1")

    assert collected() == {}
