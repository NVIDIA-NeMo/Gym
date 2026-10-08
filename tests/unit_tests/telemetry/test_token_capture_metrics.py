# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token-capture metrics against a real in-memory reader, and the cost of the disabled path."""

import asyncio
import gc
import logging
import os
import subprocess
import sys
import threading
import timeit
import weakref
from contextlib import nullcontext

import pytest

from nemo_gym.telemetry import setup as telemetry_setup
from nemo_gym.telemetry import token_capture_metrics as recorders
from nemo_gym.token_id_capture import metrics
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.token_id_capture.protocols import LineageResolution
from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry
from nemo_gym.token_id_capture.sink import (
    CaptureContext,
    commit_entry,
    record_ledger_failure,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from nemo_gym.token_id_capture.staging import CaptureAdmission, StageResult
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
from nemo_gym.token_id_capture.store import TokenCaptureStore


pytest.importorskip("opentelemetry.sdk.metrics")

OPERATION = recorders.OPERATION_ATTRIBUTE
IMPLEMENTATION = recorders.IMPLEMENTATION_ATTRIBUTE
SIZE = recorders.SIZE_CLASS_ATTRIBUTE
ERROR = recorders.ERROR_ATTRIBUTE
SEEDED_HISTORY = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "seed"}]


@pytest.fixture
def collected(monkeypatch):
    """A live meter on an in-memory reader, installed as the process telemetry handle."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    recorders._reset_for_testing()
    metrics._reset_for_testing()

    def collect() -> dict:
        data = reader.get_metrics_data()
        out: dict = {}
        for resource_metric in data.resource_metrics if data is not None else ():
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    out[metric.name] = list(metric.data.data_points)
        return out

    # Observable instruments are registered with the synchronous ones; make sure they exist.
    recorders.instruments()
    yield collect
    recorders._reset_for_testing()
    metrics._reset_for_testing()


def _durations(collected) -> dict[tuple[str, str, str, str], int]:
    """Sample count per (operation, implementation, size class, error)."""
    points = collected().get(recorders.OPERATION_DURATION_INSTRUMENT, [])
    return {
        (
            p.attributes[OPERATION],
            p.attributes[IMPLEMENTATION],
            p.attributes.get(SIZE, ""),
            p.attributes.get(ERROR, ""),
        ): p.count
        for p in points
    }


def _outcomes(collected) -> dict[tuple[str, str], int]:
    return {
        (p.attributes[recorders.OUTCOME_ATTRIBUTE], p.attributes[recorders.REASON_ATTRIBUTE]): p.value
        for p in collected().get(recorders.OUTCOME_INSTRUMENT, [])
    }


class _RecordingSink:
    def stage(self, record, *, attachments=None):
        return StageResult(ok=True, staging_key=f"{record.rollout_id}/{record.model_call_id}")


def _complete(sink) -> None:
    capture = RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 1)
    call = capture.begin_call(CaptureAdmission(rollout_id="r1", model_call_id="c1", mode="text"))
    capture.complete_call(
        call, prompt_token_ids=[1, 2, 3], generated_token_ids=[4, 5], generated_logprobs=[-0.1, -0.2]
    )


def _entry(rollout_id: str, model_call_id: str) -> TokenEntry:
    return TokenEntry(
        rollout_id=rollout_id,
        model_call_id=model_call_id,
        prompt_token_ids=[1, 2, 3],
        generation_token_ids=[4, 5],
        generation_log_probs=[-0.1, -0.2],
    )


def test_staging_times_build_and_sink_and_counts_digest_work(collected):
    _complete(_RecordingSink())

    durations = _durations(collected)
    assert durations[("stage.build_record", "", "lt_4k", "")] == 1
    assert durations[("stage.sink", "_RecordingSink", "lt_4k", "")] == 1
    tokens = {p.attributes[OPERATION]: p.value for p in collected()[recorders.OPERATION_TOKENS_INSTRUMENT]}
    assert tokens["stage.build_record"] == 5
    calls = {p.attributes[recorders.DIGEST_ATTRIBUTE]: p.value for p in collected()[recorders.DIGEST_CALLS_INSTRUMENT]}
    # The record validator recomputes the staging digest the builder already computed: two per record.
    assert calls["staging"] == 2
    assert calls["chain"] >= 1 and calls["cumulative"] >= 1


def test_a_failing_sink_is_timed_as_an_error_and_counted(collected):
    class _Broken:
        def stage(self, record, *, attachments=None):
            raise RuntimeError("store is down")

    _complete(_Broken())
    assert _durations(collected)[("stage.sink", "_Broken", "lt_4k", "error")] == 1
    assert _outcomes(collected)[("stage_failed", "sink_error")] == 1


def test_a_rejecting_sink_is_counted(collected):
    class _Rejecting:
        def stage(self, record, *, attachments=None):
            return StageResult(ok=False, error="full")

    _complete(_Rejecting())
    assert _outcomes(collected)[("stage_failed", "sink_rejected")] == 1


@pytest.mark.asyncio
async def test_model_server_writes_are_timed_by_implementation(collected, tmp_path):
    store = TokenCaptureStore(tmp_path)
    context = CaptureContext(
        rollout_id="r1", model_call_id="c1", token_sink=store, lineage_store=FileLineageStore(tmp_path)
    )
    token = set_token_sink(context)
    try:
        await resolve_parent(SEEDED_HISTORY)
        await commit_entry(_entry("r1", "c1"))
    finally:
        reset_token_sink(token)

    durations = _durations(collected)
    assert durations[("lineage.resolve", "FileLineageStore", "", "")] == 1
    # The protocol call site and the store's own work: their difference is time spent waiting for a thread.
    assert durations[("sink.put", "TokenCaptureStore", "lt_4k", "")] == 1
    assert durations[("token_store.append", "TokenCaptureStore", "lt_4k", "")] == 1
    outcomes = _outcomes(collected)
    assert outcomes[(ParentResolutionStatus.UNRESOLVED.value, "no_match")] == 1
    assert outcomes[("lock_acquired", "exclusive")] >= 1


@pytest.mark.asyncio
async def test_a_failing_put_is_timed_as_an_error(collected):
    class _FailingSink:
        async def put(self, entry):
            raise OSError("disk full")

        async def mark_incomplete(self, rollout_id, model_call_id=""):
            pass

    context = CaptureContext(rollout_id="r1", model_call_id="c1", token_sink=_FailingSink(), lineage_store=None)
    token = set_token_sink(context)
    try:
        await commit_entry(_entry("r1", "c1"))
    finally:
        reset_token_sink(token)
    assert _durations(collected)[("sink.put", "_FailingSink", "lt_4k", "error")] == 1
    assert _outcomes(collected)[("capture_failed", "write")] == 1


@pytest.mark.asyncio
async def test_a_cancelled_operation_is_labelled_cancelled(collected):
    async def never():
        await asyncio.Event().wait()

    task = asyncio.ensure_future(_timed_wait(never()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _durations(collected)[("sink.put", "TokenCaptureStore", "", "cancelled")] == 1


async def _timed_wait(awaitable) -> None:
    with metrics.timed("sink.put", implementation_name="TokenCaptureStore"):
        await awaitable


@pytest.mark.asyncio
async def test_unknown_resolver_reasons_are_counted_as_other(collected):
    class _Resolver:
        async def resolve(self, rollout_id, request_items):
            return LineageResolution(ParentResolutionStatus.UNRESOLVED, reason=f"custom-{rollout_id}")

    for index in range(3):
        context = CaptureContext(
            rollout_id=f"r{index}", model_call_id="c1", token_sink=None, lineage_store=_Resolver()
        )
        token = set_token_sink(context)
        try:
            await resolve_parent(SEEDED_HISTORY)
        finally:
            reset_token_sink(token)
    assert _outcomes(collected)[(ParentResolutionStatus.UNRESOLVED.value, "other")] == 3


@pytest.mark.asyncio
async def test_poisoned_calls_are_counted_by_reason(collected):
    await record_ledger_failure(InMemoryLineageStore(), "r1", "c1", "unresolved_parent")
    assert _outcomes(collected)[("poisoned", "unresolved_parent")] == 1
    assert _durations(collected)[("ledger.record_failure", "InMemoryLineageStore", "", "")] == 1


def test_a_lock_wait_below_the_threshold_is_not_recorded(collected):
    with metrics.timed("file_lock.wait", implementation_name="TokenCaptureStore", threshold_ms=10_000):
        pass
    assert ("file_lock.wait", "TokenCaptureStore", "", "") not in _durations(collected)


@pytest.mark.asyncio
async def test_cache_sizes_are_observed_at_export(collected, tmp_path):
    store = FileLineageStore(tmp_path)
    await store.record_failure("r1", "c1", "worker_capture_failed")
    sizes = {p.attributes[recorders.CACHE_ATTRIBUTE]: p.value for p in collected()[recorders.CACHE_SIZE_INSTRUMENT]}
    assert sizes["ledger_rollouts"] >= 1
    assert "lineage_rollouts" in sizes


def test_duration_buckets_reach_five_minutes_and_size_classes_split_long_sequences(collected):
    with metrics.timed("stage.build_record", tokens=200_000):
        pass
    (point,) = collected()[recorders.OPERATION_DURATION_INSTRUMENT]
    assert point.explicit_bounds[-1] == 300_000
    assert point.attributes[SIZE] == "ge_128k"


def test_attribute_sets_are_capped_per_instrument(collected, monkeypatch):
    monkeypatch.setattr(recorders, "MAX_ATTRIBUTE_SETS", 2)
    for index in range(5):
        with metrics.timed(f"op-{index}"):
            pass
    operations = {p.attributes[OPERATION] for p in collected()[recorders.OPERATION_DURATION_INSTRUMENT]}
    assert recorders.OVERFLOW_LABEL in operations
    assert len(operations) <= 3


def test_outcome_keys_are_capped_for_new_outcomes_as_well_as_new_reasons(collected, monkeypatch):
    monkeypatch.setattr(metrics, "MAX_OUTCOME_KEYS", 2)
    for index in range(5):
        metrics.count(f"outcome-{index}", f"reason-{index}")
    outcomes = _outcomes(collected)
    assert len(outcomes) <= 3
    assert outcomes[(metrics.OTHER_REASON, metrics.OTHER_REASON)] == 3


def test_concurrent_counts_are_not_lost(collected, monkeypatch):
    """Capture counts from many worker threads; a lost read-modify-write would undercount."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as possible to expose a race
    try:
        threads = [
            threading.Thread(target=lambda: [metrics.count("lock_acquired", "exclusive") for _ in range(20_000)])
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(previous)
    assert _outcomes(collected)[("lock_acquired", "exclusive")] == 8 * 20_000


def test_implementation_labels_keep_no_reference_to_the_class(collected):
    cls = type("ShortLivedSink", (), {})
    assert metrics.implementation(cls()) == "ShortLivedSink"
    reference = weakref.ref(cls)
    del cls
    gc.collect()
    assert reference() is None


def test_one_failing_cache_owner_does_not_hide_the_others(collected):
    class _Broken:
        def cache_sizes(self):
            raise RuntimeError("dictionary changed size during iteration")

    class _Healthy:
        def cache_sizes(self):
            return [("healthy_cache", 7)]

    broken, healthy = _Broken(), _Healthy()
    metrics.track_cache_owner(broken)
    metrics.track_cache_owner(healthy)
    sizes = {p.attributes[recorders.CACHE_ATTRIBUTE]: p.value for p in collected()[recorders.CACHE_SIZE_INSTRUMENT]}
    assert sizes["healthy_cache"] == 7


def test_a_slow_operation_logs_once_per_interval(collected, monkeypatch, caplog):
    monkeypatch.setattr(metrics, "SLOW_OPERATION_MS", 0.0)
    with caplog.at_level(logging.WARNING, logger=metrics.__name__):
        for _ in range(3):
            with metrics.timed("sink.put", implementation_name="TokenCaptureStore", tokens=10):
                pass
    assert sum("Slow token-capture operation sink.put" in r.getMessage() for r in caplog.records) == 1


def test_nothing_is_recorded_without_an_exporting_handle(monkeypatch):
    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    metrics._reset_for_testing()
    with metrics.timed("sink.put") as timer:
        timer.tokens = 5
    metrics.count("poisoned", "x")
    metrics.digest("staging", 10)
    assert metrics._Totals().outcomes() == [] and metrics._Totals().digests() == []


def test_importing_capture_does_not_import_the_telemetry_stack():
    script = (
        "import sys; import nemo_gym.token_id_capture.staging.capture; "
        "import nemo_gym.token_id_capture.sink; "
        "print('nemo_gym.telemetry.setup' in sys.modules)"
    )
    # Without coverage's subprocess hooks, which can import modules and print of their own.
    env = {key: value for key, value in os.environ.items() if not key.startswith("COV_CORE")}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    assert result.stdout.strip().splitlines()[-1] == "False"


#: The same bound Gym's span-gate overhead test uses against an empty ``with nullcontext()``.
MAX_RATIO_VS_NULLCONTEXT = 4.0


def test_disabled_site_costs_about_an_empty_with_block(monkeypatch):
    """Ratio between two timings in the same process, never an absolute time."""
    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)

    def empty():
        with nullcontext():
            pass

    def site():
        with metrics.timed("sink.put"):
            pass

    empty_ns = min(timeit.repeat(empty, number=50_000, repeat=5)) / 50_000 * 1e9
    site_ns = min(timeit.repeat(site, number=50_000, repeat=5)) / 50_000 * 1e9
    assert site_ns < empty_ns * MAX_RATIO_VS_NULLCONTEXT
