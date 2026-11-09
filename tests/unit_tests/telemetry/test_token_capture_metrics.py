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
import fcntl
import gc
import logging
import os
import subprocess
import sys
import threading
import timeit
import weakref
from contextlib import contextmanager, nullcontext

import pytest

from nemo_gym.models.token_capture import capture_metrics
from nemo_gym.models.token_capture.consumer import trajectories_from_source
from nemo_gym.models.token_capture.delivery import finalize_rollout_token_capture, retire_rollout_token_capture
from nemo_gym.models.token_capture.external_capture import VLLMWorkerCaptureHandler, publish_deferred_commit
from nemo_gym.models.token_capture.lineage import (
    FileLineageStore,
    InMemoryLineageStore,
    LineageIndex,
    stamp_continuation,
)
from nemo_gym.models.token_capture.protocols import LineageResolution
from nemo_gym.models.token_capture.records import ParentResolutionStatus, TokenEntry, stamp_lineage
from nemo_gym.models.token_capture.sink import (
    CaptureContext,
    capture_tokens,
    commit_entry,
    record_ledger_failure,
    register_call_intent,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from nemo_gym.models.token_capture.staging.capture import RolloutTokenCapture
from nemo_gym.models.token_capture.staging.digest import encode_token_ids
from nemo_gym.models.token_capture.staging.records import WORKER_CAPTURE_FAILED_REASON, CaptureAdmission, StageResult
from nemo_gym.models.token_capture.store import TokenCaptureStore
from nemo_gym.telemetry import setup as telemetry_setup
from nemo_gym.telemetry import token_capture_metrics as recorders


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
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", True)
    recorders._reset_for_testing()
    capture_metrics._reset_for_testing()

    def collect() -> dict:
        data = reader.get_metrics_data()
        out: dict = {}
        for resource_metric in data.resource_metrics if data is not None else ():
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    out[metric.name] = list(metric.data.data_points)
        return out

    # Bind the in-process totals the observable instruments read, as the first captured call would.
    assert capture_metrics.active()
    # Observable instruments are registered with the synchronous ones; make sure they exist.
    recorders.instruments()
    yield collect
    recorders._reset_for_testing()
    capture_metrics._reset_for_testing()


@pytest.fixture
def no_meter_provider(monkeypatch):
    """Pin the process's meter provider to the API's no-op one.

    An earlier test in the same worker may have installed a real SDK provider globally.
    Capture metrics record through any such provider, so tests of the off path must rule it out.
    """
    from opentelemetry import metrics

    monkeypatch.setattr(metrics, "get_meter_provider", metrics.NoOpMeterProvider)


@pytest.fixture
def framework_meter_provider(monkeypatch):
    """Stand in for a training framework that set OpenTelemetry up itself, without Gym's ``init_telemetry``.

    NeMo RL's inference workers do this with nemo-lens before running Gym's staging capture code.
    """
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", False)
    monkeypatch.setattr(metrics, "get_meter_provider", lambda: provider)
    monkeypatch.setattr(metrics, "get_meter", lambda name, *args, **kwargs: provider.get_meter(name))
    recorders._reset_for_testing()
    capture_metrics._reset_for_testing()

    def collect() -> dict:
        data = reader.get_metrics_data()
        out: dict = {}
        for resource_metric in data.resource_metrics if data is not None else ():
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    out[metric.name] = list(metric.data.data_points)
        return out

    yield collect
    provider.shutdown()
    recorders._reset_for_testing()
    capture_metrics._reset_for_testing()


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
        return StageResult(ok=True, staging_key=f"{record.capture_key}/{record.model_call_id}")


def _complete(sink) -> None:
    capture = RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 1)
    call = capture.begin_call(CaptureAdmission(capture_key="r1", model_call_id="c1", mode="text"))
    capture.complete_call(
        call, prompt_token_ids=[1, 2, 3], generated_token_ids=[4, 5], generated_logprobs=[-0.1, -0.2]
    )


def _entry(capture_key: str, model_call_id: str) -> TokenEntry:
    return TokenEntry(
        capture_key=capture_key,
        model_call_id=model_call_id,
        prompt_token_ids=[1, 2, 3],
        generation_token_ids=[4, 5],
        generation_log_probs=[-0.1, -0.2],
    )


def test_staging_times_build_and_sink_and_counts_digest_work(collected):
    _complete(_RecordingSink())

    durations = _durations(collected)
    assert durations[("staging.build_record", "", "lt_4k", "")] == 1
    assert durations[("staging_sink.stage", "_RecordingSink", "lt_4k", "")] == 1
    tokens = {p.attributes[OPERATION]: p.value for p in collected()[recorders.OPERATION_TOKENS_INSTRUMENT]}
    assert tokens["staging.build_record"] == 5
    calls = {p.attributes[recorders.DIGEST_ATTRIBUTE]: p.value for p in collected()[recorders.DIGEST_CALLS_INSTRUMENT]}
    # Since #4213 the record validator reuses the builder's staging digest: one per record.
    assert calls["staging"] == 1
    assert calls["chain"] >= 1 and calls["cumulative"] >= 1


def test_a_failing_sink_is_timed_as_an_error_and_counted(collected):
    class _Broken:
        def stage(self, record, *, attachments=None):
            raise RuntimeError("store is down")

    _complete(_Broken())
    assert _durations(collected)[("staging_sink.stage", "_Broken", "lt_4k", "error")] == 1
    assert _outcomes(collected)[("capture_failed", "staging_sink_error")] == 1


def test_a_rejecting_sink_is_counted(collected):
    class _Rejecting:
        def stage(self, record, *, attachments=None):
            return StageResult(ok=False, error="full")

    _complete(_Rejecting())
    assert _outcomes(collected)[("capture_failed", "staging_sink_rejected")] == 1


@pytest.mark.asyncio
async def test_model_server_writes_are_timed_by_implementation(collected, tmp_path):
    store = TokenCaptureStore(tmp_path)
    context = CaptureContext(
        capture_key="r1", model_call_id="c1", token_sink=store, lineage_store=FileLineageStore(tmp_path)
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
    assert durations[("sink.put.in_thread", "TokenCaptureStore", "lt_4k", "")] == 1
    outcomes = _outcomes(collected)
    assert outcomes[(f"parent_{ParentResolutionStatus.UNRESOLVED.value}", "no_match")] == 1
    assert outcomes[("lock_acquired", "exclusive")] >= 1


def test_manifest_retire_and_delete_are_timed_at_the_route_and_in_the_thread(collected, tmp_path):
    """The manifest fetch a training framework waits on: route time minus in-thread time is thread-pool queueing."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from nemo_gym.models.token_capture.control_routes import CONTROL_ROUTE_PREFIX, install_rollout_control_routes

    app = FastAPI()
    install_rollout_control_routes(app, FileLineageStore(tmp_path), auth_token="t")
    headers = {"Authorization": "Bearer t"}
    with TestClient(app) as client:
        assert client.get(f"{CONTROL_ROUTE_PREFIX}/rollouts/r1/manifest", headers=headers).status_code == 200
        for action in ("retire", "delete"):
            response = client.post(
                f"{CONTROL_ROUTE_PREFIX}/rollouts/{action}", json={"capture_keys": ["r1"]}, headers=headers
            )
            assert response.status_code == 200

    durations = _durations(collected)
    for operation in ("ledger.manifest", "ledger.retire", "ledger.delete"):
        assert durations[(operation, "FileLineageStore", "", "")] == 1
        assert durations[(f"{operation}.in_thread", "FileLineageStore", "", "")] == 1


@pytest.mark.asyncio
async def test_a_failing_put_is_timed_as_an_error(collected):
    class _FailingSink:
        async def put(self, entry):
            raise OSError("disk full")

        async def mark_incomplete(self, capture_key, model_call_id=""):
            pass

    context = CaptureContext(capture_key="r1", model_call_id="c1", token_sink=_FailingSink(), lineage_store=None)
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
    with capture_metrics.timed("sink.put", implementation_name="TokenCaptureStore"):
        await awaitable


@pytest.mark.asyncio
async def test_a_failed_lookup_and_a_skipped_resolution_are_counted(collected, tmp_path):
    class _Broken:
        async def resolve(self, capture_key, request_items):
            raise RuntimeError("ledger unreadable")

    context = CaptureContext(capture_key="r1", model_call_id="c1", token_sink=None, lineage_store=_Broken())
    token = set_token_sink(context)
    try:
        await resolve_parent(SEEDED_HISTORY)
    finally:
        reset_token_sink(token)

    # A call committed without resolving its parent first.
    context = CaptureContext(
        capture_key="r2", model_call_id="c1", token_sink=TokenCaptureStore(tmp_path), lineage_store=None
    )
    token = set_token_sink(context)
    try:
        await commit_entry(_entry("r2", "c1"))
    finally:
        reset_token_sink(token)

    outcomes = _outcomes(collected)
    assert outcomes[(f"parent_{ParentResolutionStatus.UNRESOLVED.value}", "lookup_error")] == 1
    assert outcomes[(f"parent_{ParentResolutionStatus.UNRESOLVED.value}", "not_attempted")] == 1


@pytest.mark.asyncio
async def test_unknown_resolver_reasons_are_counted_as_other(collected):
    class _Resolver:
        async def resolve(self, capture_key, request_items):
            return LineageResolution(ParentResolutionStatus.UNRESOLVED, reason=f"custom-{capture_key}")

    for index in range(3):
        context = CaptureContext(
            capture_key=f"r{index}", model_call_id="c1", token_sink=None, lineage_store=_Resolver()
        )
        token = set_token_sink(context)
        try:
            await resolve_parent(SEEDED_HISTORY)
        finally:
            reset_token_sink(token)
    assert _outcomes(collected)[(f"parent_{ParentResolutionStatus.UNRESOLVED.value}", "other")] == 3


@pytest.mark.asyncio
async def test_poisoned_calls_are_counted_by_reason(collected):
    await record_ledger_failure(InMemoryLineageStore(), "r1", "c1", "unresolved_parent")
    assert _outcomes(collected)[("poisoned", "unresolved_parent")] == 1
    assert _durations(collected)[("ledger.record_failure", "InMemoryLineageStore", "", "")] == 1


def test_a_lock_wait_below_the_threshold_is_not_recorded(collected):
    with capture_metrics.timed("file_lock.wait", implementation_name="TokenCaptureStore", threshold_ms=10_000):
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
    with capture_metrics.timed("staging.build_record", tokens=200_000):
        pass
    (point,) = collected()[recorders.OPERATION_DURATION_INSTRUMENT]
    assert point.explicit_bounds[-1] == 300_000
    assert point.attributes[SIZE] == "ge_128k"


def test_attribute_sets_are_capped_per_instrument(collected, monkeypatch):
    monkeypatch.setattr(recorders, "MAX_ATTRIBUTE_SETS", 2)
    for index in range(5):
        with capture_metrics.timed(f"op-{index}"):
            pass
    operations = {p.attributes[OPERATION] for p in collected()[recorders.OPERATION_DURATION_INSTRUMENT]}
    assert recorders.OVERFLOW_LABEL in operations
    assert len(operations) <= 3


def test_outcome_keys_are_capped_for_new_outcomes_as_well_as_new_reasons(collected, monkeypatch):
    monkeypatch.setattr(capture_metrics, "MAX_OUTCOME_KEYS", 2)
    for index in range(5):
        capture_metrics.count(f"outcome-{index}", f"reason-{index}")
    outcomes = _outcomes(collected)
    assert len(outcomes) <= 3
    assert outcomes[(capture_metrics.OTHER_REASON, capture_metrics.OTHER_REASON)] == 3


def test_concurrent_counts_are_not_lost(collected, monkeypatch):
    """Capture counts from many worker threads; a lost read-modify-write would undercount."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as possible to expose a race
    try:
        threads = [
            threading.Thread(
                target=lambda: [capture_metrics.count("lock_acquired", "exclusive") for _ in range(20_000)]
            )
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
    assert capture_metrics.implementation(cls()) == "ShortLivedSink"
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
    capture_metrics.track_cache_owner(broken)
    capture_metrics.track_cache_owner(healthy)
    sizes = {p.attributes[recorders.CACHE_ATTRIBUTE]: p.value for p in collected()[recorders.CACHE_SIZE_INSTRUMENT]}
    assert sizes["healthy_cache"] == 7


def test_a_slow_operation_logs_once_per_interval(collected, monkeypatch, caplog):
    monkeypatch.setattr(capture_metrics, "SLOW_OPERATION_MS", 0.0)
    with caplog.at_level(logging.WARNING, logger=capture_metrics.__name__):
        for _ in range(3):
            with capture_metrics.timed("sink.put", implementation_name="TokenCaptureStore", tokens=10):
                pass
    assert sum("Slow token-capture operation sink.put" in r.getMessage() for r in caplog.records) == 1


def test_staging_records_through_a_framework_meter_provider(framework_meter_provider):
    """External staging in a framework's inference worker is the primary training path.

    The worker set up OpenTelemetry itself and never called Gym's ``init_telemetry``, and its staging metrics still record.
    """
    _complete(_RecordingSink())
    exported = framework_meter_provider()

    operations = {p.attributes[OPERATION] for p in exported[recorders.OPERATION_DURATION_INSTRUMENT]}
    assert {"staging.build_record", "staging_sink.stage"} <= operations
    digests = {p.attributes[recorders.DIGEST_ATTRIBUTE]: p.value for p in exported[recorders.DIGEST_CALLS_INSTRUMENT]}
    assert digests["staging"] == 1


def test_a_provider_installed_after_gym_is_imported_is_picked_up(monkeypatch, no_meter_provider):
    """A worker imports Gym, which counts a digest at import time, before it sets telemetry up.

    Until the OpenTelemetry metrics SDK is loaded, no provider can exist, so the off answer is not cached.
    """
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", False)
    capture_metrics._reset_for_testing()
    sdk_metrics = sys.modules.pop("opentelemetry.sdk.metrics")
    try:
        assert capture_metrics.active() is False
        assert capture_metrics._PROVIDER_CHECKED_AT == float("-inf"), "nothing to recheck before the SDK loads"
    finally:
        sys.modules["opentelemetry.sdk.metrics"] = sdk_metrics

    provider = MeterProvider()
    monkeypatch.setattr(metrics, "get_meter_provider", lambda: provider)
    assert capture_metrics.active() is True
    provider.shutdown()


def test_with_the_sdk_loaded_an_off_answer_is_rechecked_after_an_interval(monkeypatch, no_meter_provider):
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", False)
    capture_metrics._reset_for_testing()
    assert capture_metrics.active() is False

    provider = MeterProvider()
    monkeypatch.setattr(metrics, "get_meter_provider", lambda: provider)
    assert capture_metrics.active() is False, "the off answer is kept for up to the recheck interval"
    monkeypatch.setattr(capture_metrics, "_PROVIDER_CHECKED_AT", float("-inf"))
    assert capture_metrics.active() is True
    provider.shutdown()


@pytest.mark.parametrize("traces_only", [False, True])
def test_nothing_is_recorded_without_a_metrics_exporter(monkeypatch, no_meter_provider, traces_only):
    """A traces-only process exports spans but has no metrics exporter, so capture sites must stay idle."""

    class _TracesOnly:
        is_exporting = True

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _TracesOnly() if traces_only else None)
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", False)
    capture_metrics._reset_for_testing()
    assert capture_metrics.timed("sink.put") is capture_metrics._NO_TIMER
    with capture_metrics.timed("sink.put") as timer:
        timer.tokens = 5
    capture_metrics.count("poisoned", "x")
    capture_metrics.digest("staging", 10)
    assert capture_metrics._Totals().outcomes() == [] and capture_metrics._Totals().digests() == []


def test_importing_capture_does_not_import_the_telemetry_stack():
    script = (
        "import sys; import nemo_gym.models.token_capture.staging.capture; "
        "import nemo_gym.models.token_capture.sink; "
        "print('nemo_gym.telemetry.setup' in sys.modules)"
    )
    # Without coverage's subprocess hooks, which can import modules and print of their own.
    env = {key: value for key, value in os.environ.items() if not key.startswith("COV_CORE")}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    assert result.stdout.strip().splitlines()[-1] == "False"


#: The same bound Gym's span-gate overhead test uses against an empty ``with nullcontext()``.
MAX_RATIO_VS_NULLCONTEXT = 4.0


def test_disabled_site_costs_about_an_empty_with_block(monkeypatch, no_meter_provider):
    """Ratio between two timings in the same process, never an absolute time."""
    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", False)

    def empty():
        with nullcontext():
            pass

    def site():
        with capture_metrics.timed("sink.put"):
            pass

    empty_ns = min(timeit.repeat(empty, number=50_000, repeat=5)) / 50_000 * 1e9
    site_ns = min(timeit.repeat(site, number=50_000, repeat=5)) / 50_000 * 1e9
    assert site_ns < empty_ns * MAX_RATIO_VS_NULLCONTEXT


# Each capture method end to end: every metric family must report real values on both.
# Token capture is Gym's own path: the model server writes token entries to its store and resolves parents
# from them, and the consumer freezes and retires the rollout. External staging is the framework path: the
# inference worker builds and stages records, and the model server writes ledger rows for them.


def _counts(collected, instrument: str, attribute: str) -> dict[str, int]:
    return {p.attributes[attribute]: p.value for p in collected().get(instrument, [])}


def _token_response(text: str, prompt: list[int], generated: list[int]) -> dict:
    return {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": text,
                "prompt_token_ids": prompt,
                "generation_token_ids": generated,
                "generation_log_probs": [-0.1] * len(generated),
            }
        ]
    }


async def _token_capture_call(store, lineage, capture_key: str, model_call_id: str, request: list, response) -> None:
    context = CaptureContext(
        capture_key=capture_key, model_call_id=model_call_id, token_sink=store, lineage_store=lineage
    )
    token = set_token_sink(context)
    try:
        await register_call_intent()
        await resolve_parent(request)
        await capture_tokens(response, request_messages=request)
    finally:
        reset_token_sink(token)


@pytest.mark.asyncio
async def test_token_capture_reports_operations_outcomes_digests_and_caches(collected, tmp_path):
    store = TokenCaptureStore(tmp_path)
    lineage = FileLineageStore(tmp_path)
    first = [{"role": "user", "content": "hi"}]
    second = [*first, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]
    await _token_capture_call(store, lineage, "r1", "c1", first, _token_response("hello", [1, 2, 3], [4, 5]))
    await _token_capture_call(store, lineage, "r1", "c2", second, _token_response("done", [1, 2, 3, 4, 5, 6, 7], [8]))
    # A call whose response carries no token ids leaves its rollout incomplete.
    await _token_capture_call(store, lineage, "r2", "c1", first, {"output": []})

    built = await trajectories_from_source("r1", store)
    # A call still in flight when the rollout froze cannot change the frozen verdict.
    late = CaptureContext(capture_key="r1", model_call_id="c3", token_sink=store, lineage_store=None)
    token = set_token_sink(late)
    try:
        await commit_entry(_entry("r1", "c3"))
    finally:
        reset_token_sink(token)
    # A consumer holding a snapshot identity the store no longer has must not remove the current records.
    stale = {**built, "_capture_snapshot": {**built["_capture_snapshot"], "snapshot_id": "older-freeze"}}
    assert await retire_rollout_token_capture("r1", store, stale) is False
    assert await retire_rollout_token_capture("r1", store, built) is True

    durations = _durations(collected)
    assert durations[("sink.begin_call", "TokenCaptureStore", "", "")] == 3
    # A request with no assistant turn is a root without a lookup; only the continuation asks the store.
    assert durations[("lineage.resolve", "FileLineageStore", "", "")] == 1
    assert durations[("sink.put", "TokenCaptureStore", "lt_4k", "")] == 2
    assert durations[("sink.put.in_thread", "TokenCaptureStore", "lt_4k", "")] == 2
    assert durations[("source.freeze", "TokenCaptureStore", "", "")] == 1
    assert durations[("source.drop", "TokenCaptureStore", "", "")] == 2
    # The late write reached the store and was refused there.
    assert durations[("sink.put", "TokenCaptureStore", "lt_4k", "error")] == 1
    tokens = _counts(collected, recorders.OPERATION_TOKENS_INSTRUMENT, OPERATION)
    assert tokens["sink.put"] == 5 + 8 + 5
    assert tokens["sink.put.in_thread"] == 5 + 8 + 5

    outcomes = _outcomes(collected)
    assert outcomes[(f"parent_{ParentResolutionStatus.ROOT.value}", "")] == 2
    assert outcomes[(f"parent_{ParentResolutionStatus.RESOLVED.value}", "")] == 1
    assert outcomes[("late_write_dropped", "frozen")] == 1
    assert outcomes[("marked_incomplete", "")] == 1
    assert outcomes[("retire_skipped", "stale_snapshot")] == 1
    assert outcomes[("lock_acquired", "exclusive")] >= 4
    assert not any(outcome == "poisoned" for outcome, _ in outcomes), "token capture writes no ledger rows"

    # Token capture identifies calls by fingerprint; it computes no staging digests.
    assert collected().get(recorders.DIGEST_CALLS_INSTRUMENT, []) == []
    assert collected().get(recorders.DIGEST_BYTES_INSTRUMENT, []) == []

    sizes = _counts(collected, recorders.CACHE_SIZE_INSTRUMENT, recorders.CACHE_ATTRIBUTE)
    # Only c2's lookup read lineage: one rollout, materialized through its parent c1's five tokens.
    assert sizes["lineage_rollouts"] == 1
    assert sizes["lineage_materialized_tokens"] == 5
    assert sizes["ledger_rollouts"] == 0


@pytest.mark.asyncio
async def test_token_capture_times_the_freeze_of_a_rollout_that_already_carries_token_ids(collected, tmp_path):
    """The harness returned token ids itself; the capture is frozen only so it can be retired after handoff."""
    store = TokenCaptureStore(tmp_path)
    store.append(_entry("r1", "c1"))
    result = {
        "_ng_rollout_id": "r1",
        "response": {"output": [{"type": "message", "generation_token_ids": [4, 5]}]},
    }
    built = await finalize_rollout_token_capture(result, store)
    assert built["_capture_snapshot"]["snapshot_id"]

    assert _durations(collected)[("source.freeze", "TokenCaptureStore", "", "")] == 1


@contextmanager
def _held_lock(path):
    with open(path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_token_capture_times_a_contended_store_lock(collected, tmp_path):
    store = TokenCaptureStore(tmp_path)
    held, release = threading.Event(), threading.Event()

    def holder():
        # A raw lock, so the holder's own acquisition is not a sample.
        with _held_lock(store.lock_path_for("r1")):
            held.set()
            release.wait()

    thread = threading.Thread(target=holder)
    thread.start()
    held.wait()
    threading.Timer(0.05, release.set).start()
    store.append(_entry("r1", "c1"))
    thread.join()

    (point,) = [
        p for p in collected()[recorders.OPERATION_DURATION_INSTRUMENT] if p.attributes[OPERATION] == "file_lock.wait"
    ]
    assert point.attributes[IMPLEMENTATION] == "TokenCaptureStore"
    assert point.count == 1
    # Milliseconds: the writer waited for the 50 ms hold.
    assert 40 <= point.sum < 5_000


def _staged(capture: RolloutTokenCapture, model_call_id: str, generated_logprobs: list[float]):
    call = capture.begin_call(CaptureAdmission(capture_key="r1", model_call_id=model_call_id, mode="text"))
    return capture.complete_call(
        call, prompt_token_ids=[1, 2, 3], generated_token_ids=[4, 5], generated_logprobs=generated_logprobs
    )


async def _finalize_on_model_server(ledger, model_call_id: str, coords) -> CaptureContext:
    context = CaptureContext(
        capture_key="r1",
        model_call_id=model_call_id,
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        request_items=[{"role": "user", "content": "go"}],
        capture_admission=CaptureAdmission(capture_key="r1", model_call_id=model_call_id, mode="text"),
    )
    payload = {
        "id": f"response-{model_call_id}",
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
        "ng_commit_coords": coords.model_dump(mode="json"),
    }
    handler = VLLMWorkerCaptureHandler()
    token = set_token_sink(context)
    try:
        handler.prepare_response(payload)
        await handler.finalize_response(payload)
    finally:
        reset_token_sink(token)
    return context


@pytest.mark.asyncio
async def test_external_staging_reports_operations_outcomes_digests_and_caches(collected, tmp_path):
    capture = RolloutTokenCapture(sink=_RecordingSink(), weight_version_fn=lambda: 7)
    ledger = FileLineageStore(tmp_path)
    committed = await _finalize_on_model_server(ledger, "c1", _staged(capture, "c1", [-0.1, -0.2]))
    # Too few logprobs: the worker cannot build the record and acknowledges a failed capture.
    poisoned = await _finalize_on_model_server(ledger, "c2", _staged(capture, "c2", [-0.1]))
    assert committed.committed and not poisoned.committed

    durations = _durations(collected)
    assert durations[("staging.build_record", "", "lt_4k", "")] == 1
    assert durations[("staging.build_record", "", "", "error")] == 1
    # Ordinary staging is not a checkpoint cut, so it records no cut.* operation.
    assert not any(operation.startswith("cut.") for operation, *_ in durations)
    assert durations[("staging_sink.stage", "_RecordingSink", "lt_4k", "")] == 1
    assert durations[("ledger.record", "FileLineageStore", "lt_4k", "")] == 1
    assert durations[("ledger.record.in_thread", "FileLineageStore", "lt_4k", "")] == 1
    assert durations[("ledger.record_failure", "FileLineageStore", "", "")] == 1
    tokens = _counts(collected, recorders.OPERATION_TOKENS_INSTRUMENT, OPERATION)
    assert tokens["staging.build_record"] == 5
    assert tokens["ledger.record"] == 5

    outcomes = _outcomes(collected)
    assert outcomes[("capture_failed", "build")] == 1
    assert outcomes[("poisoned", WORKER_CAPTURE_FAILED_REASON)] == 1
    assert ("marked_incomplete", "") not in outcomes, "external staging has no local token sink to mark"

    # One staged record of five tokens: one cumulative and one chain hash over its tokens and one staging digest.
    # The extras digest is computed twice, by the builder and again by the record's validator.
    # The failed build stopped before hashing anything.
    calls = _counts(collected, recorders.DIGEST_CALLS_INSTRUMENT, recorders.DIGEST_ATTRIBUTE)
    assert calls == {"cumulative": 1, "chain": 1, "extras": 2, "staging": 1}
    nbytes = _counts(collected, recorders.DIGEST_BYTES_INSTRUMENT, recorders.DIGEST_ATTRIBUTE)
    assert nbytes["cumulative"] == len(encode_token_ids([1, 2, 3, 4, 5]))
    assert nbytes["chain"] == 1 + len(encode_token_ids([1, 2, 3, 4, 5]))

    sizes = _counts(collected, recorders.CACHE_SIZE_INSTRUMENT, recorders.CACHE_ATTRIBUTE)
    # The ledger caches r1's rows as it appends them; no lookup materialized token lineage.
    assert sizes["ledger_rollouts"] == 1
    assert sizes["lineage_rollouts"] == 0


@pytest.mark.asyncio
async def test_external_staging_times_framework_admission_against_the_ledger(collected, tmp_path):
    """Framework-owned parent selection reads the ledger manifest and records a call intent before dispatch."""
    ledger = FileLineageStore(tmp_path)
    context = CaptureContext(
        capture_key="r1",
        model_call_id="c1",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        parent_resolution_mode="source_history",
    )
    token = set_token_sink(context)
    try:
        await resolve_parent([{"role": "user", "content": "go"}])
        await register_call_intent()
    finally:
        reset_token_sink(token)
    assert context.intent_registered

    durations = _durations(collected)
    assert durations[("ledger.manifest", "FileLineageStore", "", "")] == 1
    assert durations[("ledger.manifest.in_thread", "FileLineageStore", "", "")] == 1
    assert durations[("ledger.begin_call", "FileLineageStore", "", "")] == 1


def test_external_staging_times_a_contended_ledger_lock(collected, tmp_path):
    ledger = FileLineageStore(tmp_path)
    held, release = threading.Event(), threading.Event()

    def holder():
        with _held_lock(ledger._store.lock_path_for("r1")):
            held.set()
            release.wait()

    asyncio.run(ledger.record_failure("r1", "c0", WORKER_CAPTURE_FAILED_REASON))
    capture_metrics._reset_for_testing()
    recorders._reset_for_testing()
    recorders.instruments()
    thread = threading.Thread(target=holder)
    thread.start()
    held.wait()
    threading.Timer(0.05, release.set).start()
    ledger._manifest("r1")
    thread.join()

    waits = [
        p for p in collected()[recorders.OPERATION_DURATION_INSTRUMENT] if p.attributes[OPERATION] == "file_lock.wait"
    ]
    # The ledger shares the token store's lock file; the wait is labelled with the ledger that waited.
    assert [(p.attributes[IMPLEMENTATION], p.count) for p in waits] == [("FileLineageStore", 1)]
    assert 40 <= waits[0].sum < 5_000


def test_stores_built_during_an_export_do_not_blank_the_cache_metric(collected):
    """Stores are built on capture threads while the export thread reads every cache owner."""

    class _Owner:
        def cache_sizes(self):
            return [("steady_cache", 1)]

    steady = _Owner()
    capture_metrics.track_cache_owner(steady)
    stop = threading.Event()
    kept = []

    def build_stores():
        while not stop.is_set():
            owner = type("_Churn", (), {"cache_sizes": lambda self: []})()
            capture_metrics.track_cache_owner(owner)
            kept.append(owner)
            if len(kept) > 200:
                kept.clear()

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    thread = threading.Thread(target=build_stores)
    thread.start()
    try:
        missing = 0
        for _ in range(300):
            sizes = _counts(collected, recorders.CACHE_SIZE_INSTRUMENT, recorders.CACHE_ATTRIBUTE)
            missing += sizes.get("steady_cache") != 1
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(previous)
    assert missing == 0


# Review follow-ups: each pins a count or label a reviewer found wrong or untested.


@pytest.mark.asyncio
async def test_token_capture_counts_a_parent_matched_on_its_tool_call_turn(collected):
    """A harness rewrote the tool arguments it echoes back; Gym still finds the parent by the turn's tool calls."""
    context_items = [{"role": "user", "content": "Inspect the repository."}]

    def turn(arguments: str) -> list[dict]:
        return [
            {"role": "assistant", "content": "I will inspect it."},
            {"type": "function_call", "call_id": "call_0", "name": "Bash", "arguments": arguments},
        ]

    store = InMemoryLineageStore()
    entry = TokenEntry(
        capture_key="r1",
        model_call_id="c1",
        prompt_token_ids=[11, 12],
        generation_token_ids=[101],
        generation_log_probs=[-0.1],
        output_items=turn('{"command":"cd /repo && ls"}'),
        response_id="response-c1",
    )
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    stamp_continuation(entry, context_items)
    await store.put(entry)

    request = [
        *context_items,
        *turn('{"command":"ls"}'),
        {"type": "function_call_output", "call_id": "call_0", "output": "files"},
        {"role": "user", "content": "continue"},
    ]
    context = CaptureContext(capture_key="r1", model_call_id="c2", token_sink=None, lineage_store=store)
    token = set_token_sink(context)
    try:
        await resolve_parent(request)
    finally:
        reset_token_sink(token)

    assert context.parent_resolution.reason == "tool_call_turn"
    assert _outcomes(collected) == {(f"parent_{ParentResolutionStatus.RESOLVED.value}", "tool_call_turn"): 1}


@pytest.mark.asyncio
async def test_external_staging_counts_late_ledger_rows_and_only_written_failures(collected, tmp_path):
    """A call that finishes after its rollout was retired leaves staged tokens that no manifest names."""
    ledger = FileLineageStore(tmp_path)
    await record_ledger_failure(ledger, "r1", "c1", WORKER_CAPTURE_FAILED_REASON)
    # A replay of the same failure row writes nothing new.
    await record_ledger_failure(ledger, "r1", "c1", WORKER_CAPTURE_FAILED_REASON)
    await ledger.retire(["r1"])
    await record_ledger_failure(ledger, "r1", "c2", WORKER_CAPTURE_FAILED_REASON)
    capture = RolloutTokenCapture(sink=_RecordingSink(), weight_version_fn=lambda: 7)
    late = await _finalize_on_model_server(ledger, "c3", _staged(capture, "c3", [-0.1, -0.2]))
    assert late.committed

    outcomes = _outcomes(collected)
    assert outcomes[("poisoned", WORKER_CAPTURE_FAILED_REASON)] == 1
    assert outcomes[("late_write_dropped", "retired")] == 2
    durations = _durations(collected)
    # Every attempt is timed at the call site and in the ledger's thread, written or not.
    assert durations[("ledger.record_failure", "FileLineageStore", "", "")] == 3
    assert durations[("ledger.record_failure.in_thread", "FileLineageStore", "", "")] == 3


@pytest.mark.asyncio
async def test_token_capture_says_why_a_drop_was_refused(collected, tmp_path):
    store = TokenCaptureStore(tmp_path)
    request = [{"role": "user", "content": "hi"}]
    await _token_capture_call(store, None, "r1", "c1", request, _token_response("hello", [1, 2, 3], [4, 5]))
    built = await trajectories_from_source("r1", store)
    assert await retire_rollout_token_capture("r1", store, built) is True
    await store.delete(["r1"])
    # The rollout was deleted after the drop, so it is no longer frozen.
    assert await retire_rollout_token_capture("r1", store, built) is False

    class _PlainSource:
        """A framework source without the optional ``drop_with_reason`` extension."""

        async def drop(self, rollout_id, *, snapshot_id, version):
            return False

    assert await retire_rollout_token_capture("r1", _PlainSource(), built) is False

    outcomes = _outcomes(collected)
    assert outcomes[("retire_skipped", "not_frozen")] == 1
    assert outcomes[("retire_skipped", "")] == 1
    assert ("retire_skipped", "stale_snapshot") not in outcomes


@pytest.mark.asyncio
async def test_token_capture_does_not_count_a_mark_that_failed(collected):
    class _BrokenSink:
        async def put(self, entry):
            raise OSError("disk full")

        async def mark_incomplete(self, capture_key, model_call_id=""):
            raise OSError("disk still full")

    context = CaptureContext(capture_key="r1", model_call_id="c1", token_sink=_BrokenSink(), lineage_store=None)
    token = set_token_sink(context)
    try:
        await commit_entry(_entry("r1", "c1"))
    finally:
        reset_token_sink(token)

    outcomes = _outcomes(collected)
    assert ("marked_incomplete", "") not in outcomes
    # The failure is still counted: the rollout may be incomplete without the mark.
    assert outcomes[("capture_failed", "write")] == 1


def test_token_capture_counts_a_rollout_marked_for_an_unclassified_request(collected, tmp_path):
    from nemo_gym.models.model_call_capture import _CaptureMiddleware

    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        pass

    store = TokenCaptureStore(tmp_path)
    path = "/ng-rollout/r1/training-token-capture/v1/completions"
    middleware = _CaptureMiddleware(
        app, store=None, model_server_name="srv", token_store=store, token_capture_enabled=True
    )
    asyncio.run(middleware({"type": "http", "path": path, "raw_path": path.encode(), "headers": []}, receive, send))

    assert store.is_incomplete("r1")
    assert _outcomes(collected)[("marked_incomplete", "unclassified_request")] == 1


@pytest.mark.asyncio
async def test_external_staging_times_a_ledger_row_published_on_delivery(collected, tmp_path):
    """The ledger row of a call whose commit waits for delivery is timed when it is published."""
    capture = RolloutTokenCapture(sink=_RecordingSink(), weight_version_fn=lambda: 7)
    ledger = FileLineageStore(tmp_path)
    context = CaptureContext(
        capture_key="r1",
        model_call_id="c1",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        publish_on_delivery=True,
        request_items=[{"role": "user", "content": "go"}],
        capture_admission=CaptureAdmission(capture_key="r1", model_call_id="c1", mode="text"),
    )
    payload = {
        "id": "response-c1",
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
        "ng_commit_coords": _staged(capture, "c1", [-0.1, -0.2]).model_dump(mode="json"),
    }
    handler = VLLMWorkerCaptureHandler()
    token = set_token_sink(context)
    try:
        handler.prepare_response(payload)
        await handler.finalize_response(payload)
    finally:
        reset_token_sink(token)
    assert context.deferred_commit is not None
    assert ("ledger.record", "FileLineageStore", "lt_4k", "") not in _durations(collected)

    await publish_deferred_commit(context)
    assert _durations(collected)[("ledger.record", "FileLineageStore", "lt_4k", "")] == 1


def test_external_staging_times_checkpoint_cut_chunks(collected):
    capture = RolloutTokenCapture(sink=_RecordingSink(), weight_version_fn=lambda: 7)
    call = capture.begin_call(CaptureAdmission(capture_key="r1", model_call_id="c1", mode="text"))
    capture.build_prefix_record(call, prompt_token_ids=[1, 2, 3], generated_token_ids=[4], generated_logprobs=[-0.1])
    capture.build_generation_chunk_record(call, generated_token_ids=[5, 6], generated_logprobs=[-0.1, -0.2])

    durations = _durations(collected)
    assert durations[("cut.build_prefix_record", "", "lt_4k", "")] == 1
    assert durations[("cut.build_chunk_record", "", "lt_4k", "")] == 1
    assert not any(operation == "staging.build_record" for operation, *_ in durations)


@pytest.mark.asyncio
async def test_external_staging_times_the_checkpoint_ledger_retire(collected, tmp_path):
    from nemo_gym.core.episode_types import EpisodeId
    from nemo_gym.models._checkpoint import retire_ledgers

    await retire_ledgers(FileLineageStore(tmp_path), [EpisodeId(rollout_id="r1", attempt=1)])
    durations = _durations(collected)
    assert durations[("ledger.retire", "FileLineageStore", "", "")] == 1
    assert durations[("ledger.retire.in_thread", "FileLineageStore", "", "")] == 1


@pytest.mark.parametrize(("tokens", "expected"), [(4_095, "lt_4k"), (4_096, "lt_32k"), (131_072, "ge_128k")])
def test_size_classes_split_at_their_exact_bounds(collected, tokens, expected):
    with capture_metrics.timed("sink.put", implementation_name="TokenCaptureStore", tokens=tokens):
        pass
    (point,) = collected()[recorders.OPERATION_DURATION_INSTRUMENT]
    assert point.attributes[SIZE] == expected


def test_token_totals_are_split_by_implementation(collected):
    for name, tokens in (("TokenCaptureStore", 10), ("FrameworkSink", 30)):
        with capture_metrics.timed("sink.put", implementation_name=name, tokens=tokens):
            pass
    totals = {
        (p.attributes[OPERATION], p.attributes[IMPLEMENTATION]): p.value
        for p in collected()[recorders.OPERATION_TOKENS_INSTRUMENT]
    }
    assert totals == {("sink.put", "TokenCaptureStore"): 10, ("sink.put", "FrameworkSink"): 30}


def test_an_operation_under_the_slow_threshold_does_not_warn(collected, caplog):
    assert capture_metrics.SLOW_OPERATION_MS == 5_000.0
    with caplog.at_level(logging.WARNING, logger=capture_metrics.__name__):
        with capture_metrics.timed("sink.put", implementation_name="TokenCaptureStore", tokens=10):
            pass
    assert not caplog.records


def test_lineage_cache_sizes_survive_concurrent_rollout_growth(collected):
    """The export thread reads the lineage index while the event loop adds rollouts to it."""
    index = LineageIndex(max_rollouts=100_000, max_tokens=10**12)
    stop = threading.Event()

    def add_rollouts():
        count = 0
        while not stop.is_set():
            index.for_rollout(f"r{count}")
            count += 1

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    thread = threading.Thread(target=add_rollouts)
    thread.start()
    errors = 0
    try:
        for _ in range(5_000):
            try:
                index.cache_sizes()
            except RuntimeError:
                errors += 1
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(previous)
    assert errors == 0


@pytest.mark.asyncio
async def test_token_capture_counts_a_write_after_retirement_as_retired(collected, tmp_path):
    store = TokenCaptureStore(tmp_path)
    await store.retire(["r1"])
    context = CaptureContext(capture_key="r1", model_call_id="c1", token_sink=store, lineage_store=None)
    token = set_token_sink(context)
    try:
        await commit_entry(_entry("r1", "c1"))
    finally:
        reset_token_sink(token)

    outcomes = _outcomes(collected)
    assert outcomes[("late_write_dropped", "retired")] == 1
    assert ("late_write_dropped", "frozen") not in outcomes


@pytest.mark.asyncio
async def test_token_capture_labels_a_contended_lineage_read_with_the_lineage_store(collected, tmp_path):
    """A parent lookup reads the token store's records under a shared lock on the same lock file."""
    store = TokenCaptureStore(tmp_path)
    lineage = FileLineageStore(tmp_path)
    store.append(_entry("r1", "c1"))
    held, release = threading.Event(), threading.Event()

    def holder():
        with _held_lock(store.lock_path_for("r1")):
            held.set()
            release.wait()

    thread = threading.Thread(target=holder)
    thread.start()
    held.wait()
    threading.Timer(0.05, release.set).start()
    request = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "more"},
    ]
    await lineage.resolve("r1", request)
    thread.join()

    waits = [
        p for p in collected()[recorders.OPERATION_DURATION_INSTRUMENT] if p.attributes[OPERATION] == "file_lock.wait"
    ]
    assert [(p.attributes[IMPLEMENTATION], p.count) for p in waits] == [("FileLineageStore", 1)]
    assert 40 <= waits[0].sum < 5_000
