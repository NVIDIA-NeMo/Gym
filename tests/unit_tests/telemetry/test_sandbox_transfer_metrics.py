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
"""Sandbox file-transfer duration histogram and byte counter, against a real in-memory reader."""

import asyncio
import os

import pytest

from nemo_gym.sandbox.api import AsyncSandbox, SandboxSpec
from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry.span_groups import GymSpanGroup
from tests.unit_tests.sandbox.test_sandbox import FakeSandboxProvider
from tests.unit_tests.telemetry.test_call_sites import recorded_spans  # noqa: F401 - fixture
from tests.unit_tests.telemetry.test_sandbox_active import (  # noqa: F401 - fixtures
    collected_metrics,
    sandbox_group_on,
    set_sandbox_gates,
)


pytest.importorskip("opentelemetry.sdk.metrics")

PROVIDER = gym_metrics.SANDBOX_PROVIDER_ATTRIBUTE
DIRECTION = gym_metrics.SANDBOX_TRANSFER_DIRECTION_ATTRIBUTE
OUTCOME = gym_metrics.SANDBOX_TRANSFER_OUTCOME_ATTRIBUTE
DURATION = gym_metrics.SANDBOX_TRANSFER_DURATION_INSTRUMENT
BYTES = gym_metrics.SANDBOX_TRANSFER_BYTES_INSTRUMENT


class _SlowUploads(FakeSandboxProvider):
    """Uploads take a measurable time, so a seconds-for-milliseconds slip would show in the sum."""

    async def upload_file(self, handle, source_path, target_path):
        await asyncio.sleep(0.03)
        await super().upload_file(handle, source_path, target_path)


def _by_attributes(points):
    return {tuple(sorted(point.attributes.items())): point for point in points}


@pytest.fixture
def metrics_exporting(monkeypatch):
    """The `sandbox` span group off and a metrics exporter on: the default preset with metrics enabled."""
    set_sandbox_gates(monkeypatch, group=False, exporter=True)


@pytest.fixture
def local_stats(monkeypatch):
    """Count ``os.stat`` calls per path. ``Path.stat`` and ``Path.is_file`` go through it too."""
    real_stat, counts = os.stat, {}

    def counting_stat(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)):
            counts[os.fspath(path)] = counts.get(os.fspath(path), 0) + 1
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting_stat)
    return lambda path: counts.get(os.fspath(path), 0)


@pytest.mark.asyncio
async def test_seed_files_uploads_and_downloads_record_duration_and_bytes(
    collected_metrics,  # noqa: F811
    metrics_exporting,
    tmp_path,
):
    sandbox = AsyncSandbox(_SlowUploads(), SandboxSpec(image="img", files={"/seed.txt": "seed"}))
    await sandbox.start()
    local = tmp_path / "upload.bin"
    local.write_bytes(b"x" * 1000)
    await sandbox.upload(local, "/remote/upload.bin")
    await sandbox.download("/remote/out.txt", tmp_path / "out.txt")

    durations = _by_attributes(collected_metrics()[DURATION])
    upload = durations[tuple(sorted({PROVIDER: "fake", DIRECTION: "upload", OUTCOME: "ok"}.items()))]
    download = durations[tuple(sorted({PROVIDER: "fake", DIRECTION: "download", OUTCOME: "ok"}.items()))]
    assert upload.count == 2  # the seed file and the explicit upload
    assert 60 <= upload.sum < 10_000  # two 30 ms uploads, recorded in milliseconds
    assert download.count == 1
    assert list(upload.explicit_bounds) == list(gym_metrics.SANDBOX_TRANSFER_DURATION_BOUNDARIES_MS)

    sizes = {point.attributes[DIRECTION]: point.value for point in collected_metrics()[BYTES]}
    assert sizes == {"upload": len("seed") + 1000, "download": len(b"downloaded")}


@pytest.mark.asyncio
async def test_failed_upload_records_an_error_duration_and_no_bytes(
    collected_metrics,  # noqa: F811
    sandbox_group_on,  # noqa: F811
    tmp_path,
):
    class _Refusing(FakeSandboxProvider):
        async def upload_file(self, handle, source_path, target_path):
            raise RuntimeError("disk full")

    sandbox = await AsyncSandbox(_Refusing(), SandboxSpec(image="img")).start()
    local = tmp_path / "upload.bin"
    local.write_bytes(b"payload")
    with pytest.raises(RuntimeError, match="disk full"):
        await sandbox.upload(local, "/remote/upload.bin")

    (duration,) = collected_metrics()[DURATION]
    assert duration.attributes == {PROVIDER: "fake", DIRECTION: "upload", OUTCOME: "error"}
    assert BYTES not in collected_metrics()


@pytest.mark.asyncio
async def test_cancelled_download_records_a_cancelled_duration(
    collected_metrics,  # noqa: F811
    sandbox_group_on,  # noqa: F811
    tmp_path,
):
    class _Hanging(FakeSandboxProvider):
        async def download_file(self, handle, source_path, target_path):
            await asyncio.Event().wait()

    sandbox = await AsyncSandbox(_Hanging(), SandboxSpec(image="img")).start()
    task = asyncio.create_task(sandbox.download("/remote/out.txt", tmp_path / "out.txt"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (duration,) = collected_metrics()[DURATION]
    assert duration.attributes == {PROVIDER: "fake", DIRECTION: "download", OUTCOME: "cancelled"}
    assert BYTES not in collected_metrics()


@pytest.mark.asyncio
async def test_nothing_is_recorded_without_the_span_group_or_a_metrics_exporter(
    collected_metrics,  # noqa: F811
    local_stats,
    monkeypatch,
    tmp_path,
):
    set_sandbox_gates(monkeypatch, group=False, exporter=False)
    provider = FakeSandboxProvider()
    sandbox = await AsyncSandbox(provider, SandboxSpec(image="img", files={"/seed.txt": "seed"})).start()
    local = tmp_path / "upload.bin"
    local.write_bytes(b"payload")
    await sandbox.upload(local, "/remote/upload.bin")
    await sandbox.download("/remote/out.txt", tmp_path / "out.txt")

    assert len(provider.upload_calls) == 2
    assert (tmp_path / "out.txt").read_bytes() == b"downloaded"
    assert DURATION not in collected_metrics() and BYTES not in collected_metrics()
    assert local_stats(local) == 0


@pytest.mark.asyncio
async def test_a_missing_local_file_never_fails_a_transfer(
    collected_metrics,  # noqa: F811
    metrics_exporting,
    tmp_path,
):
    class _WritesElsewhere(FakeSandboxProvider):
        async def download_file(self, handle, source_path, target_path):
            pass

    sandbox = await AsyncSandbox(_WritesElsewhere(), SandboxSpec(image="img")).start()
    await sandbox.download("/remote/out.txt", tmp_path / "never-written.txt")

    (duration,) = collected_metrics()[DURATION]
    assert duration.attributes == {PROVIDER: "fake", DIRECTION: "download", OUTCOME: "ok"}
    assert BYTES not in collected_metrics()


@pytest.mark.asyncio
async def test_a_stat_failure_never_fails_a_transfer(
    collected_metrics,  # noqa: F811
    metrics_exporting,
    monkeypatch,
    tmp_path,
):
    """A permission or I/O error on the size read is telemetry's problem, not the transfer's."""
    local = tmp_path / "upload.bin"
    local.write_bytes(b"payload")
    sandbox = await AsyncSandbox(FakeSandboxProvider(), SandboxSpec(image="img")).start()
    real_stat = os.stat

    def refusing_stat(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and os.fspath(path) == os.fspath(local):
            raise PermissionError(13, "denied", os.fspath(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", refusing_stat)
    await sandbox.upload(local, "/remote/upload.bin")

    (duration,) = collected_metrics()[DURATION]
    assert duration.attributes[OUTCOME] == "ok"
    assert BYTES not in collected_metrics()


@pytest.mark.asyncio
async def test_a_local_path_that_is_not_a_regular_file_records_no_bytes(
    collected_metrics,  # noqa: F811
    metrics_exporting,
    tmp_path,
):
    """A directory's ``st_size`` is filesystem bookkeeping, not bytes moved."""

    class _AcceptsAnything(FakeSandboxProvider):
        async def upload_file(self, handle, source_path, target_path):
            pass

    sandbox = await AsyncSandbox(_AcceptsAnything(), SandboxSpec(image="img")).start()
    await sandbox.upload(tmp_path, "/remote/dir")

    (duration,) = collected_metrics()[DURATION]
    assert duration.attributes[OUTCOME] == "ok"
    assert BYTES not in collected_metrics()


def _non_recording_spans(monkeypatch, *, group_spans: bool) -> None:
    """Turn the lens `sandbox` group on with a tracer whose spans never record, or leave the group off."""
    import nemo.lens.helpers as lens_helpers
    from nemo.lens.state import set_enabled_span_groups
    from opentelemetry import trace

    monkeypatch.setattr(lens_helpers, "get_tracer", lambda *a, **k: trace.NoOpTracer())
    set_enabled_span_groups(GymSpanGroup.resolve("all") if group_spans else frozenset())


@pytest.mark.asyncio
@pytest.mark.parametrize("group_spans", [True, False], ids=["non-recording span", "no span"])
async def test_no_stat_without_a_recording_span_or_a_metrics_exporter(
    collected_metrics,  # noqa: F811
    local_stats,
    monkeypatch,
    tmp_path,
    group_spans,
):
    """A stat on a shared filesystem is a metadata round trip, so a size nothing reads is never taken."""
    pytest.importorskip("nemo.lens")
    set_sandbox_gates(monkeypatch, group=True, exporter=False)
    _non_recording_spans(monkeypatch, group_spans=group_spans)
    sandbox = await AsyncSandbox(FakeSandboxProvider(), SandboxSpec(image="img")).start()
    upload, download = tmp_path / "upload.bin", tmp_path / "out.txt"
    upload.write_bytes(b"payload")
    await sandbox.upload(upload, "/remote/upload.bin")
    await sandbox.download("/remote/out.txt", download)

    assert download.read_bytes() == b"downloaded"
    assert local_stats(upload) == 0 and local_stats(download) == 0
    assert BYTES not in collected_metrics()


@pytest.mark.asyncio
async def test_a_metrics_exporter_alone_records_durations_and_bytes_with_one_stat_each(
    collected_metrics,  # noqa: F811
    metrics_exporting,
    local_stats,
    tmp_path,
):
    """The `sandbox` span group is off in the default preset; the transfer metrics follow the exporter alone."""
    sandbox = await AsyncSandbox(FakeSandboxProvider(), SandboxSpec(image="img")).start()
    upload, download = tmp_path / "upload.bin", tmp_path / "out.txt"
    upload.write_bytes(b"x" * 1000)
    await sandbox.upload(upload, "/remote/upload.bin")
    await sandbox.download("/remote/out.txt", download)

    durations = {
        (point.attributes[DIRECTION], point.attributes[OUTCOME]): point.count
        for point in collected_metrics()[DURATION]
    }
    assert durations == {("upload", "ok"): 1, ("download", "ok"): 1}
    sizes = {point.attributes[DIRECTION]: point.value for point in collected_metrics()[BYTES]}
    assert sizes == {"upload": 1000, "download": len(b"downloaded")}
    assert local_stats(upload) == 1 and local_stats(download) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exporter", [False, True], ids=["traces only", "traces and metrics"])
async def test_a_recording_span_takes_one_stat_per_transfer(
    recorded_spans,  # noqa: F811
    collected_metrics,  # noqa: F811
    local_stats,
    monkeypatch,
    tmp_path,
    exporter,
):
    set_sandbox_gates(monkeypatch, group=True, exporter=exporter)
    sandbox = await AsyncSandbox(FakeSandboxProvider(), SandboxSpec(image="img")).start()
    upload, download = tmp_path / "upload.bin", tmp_path / "out.txt"
    upload.write_bytes(b"x" * 64)
    await sandbox.upload(upload, "/remote/upload.bin")
    await sandbox.download("/remote/out.txt", download)

    spans = {span.name: span for span in recorded_spans()}
    assert spans["gym.sandbox.upload"].attributes["nemo.gym.sandbox.transfer.size_bytes"] == 64
    assert spans["gym.sandbox.download"].attributes["nemo.gym.sandbox.transfer.size_bytes"] == len(b"downloaded")
    assert local_stats(upload) == 1 and local_stats(download) == 1


@pytest.mark.asyncio
async def test_a_failed_transfer_takes_no_stat(
    collected_metrics,  # noqa: F811
    local_stats,
    monkeypatch,
    tmp_path,
):
    class _Refusing(FakeSandboxProvider):
        async def upload_file(self, handle, source_path, target_path):
            raise RuntimeError("disk full")

    set_sandbox_gates(monkeypatch, group=True, exporter=True)
    sandbox = await AsyncSandbox(_Refusing(), SandboxSpec(image="img")).start()
    local = tmp_path / "upload.bin"
    local.write_bytes(b"payload")
    with pytest.raises(RuntimeError, match="disk full"):
        await sandbox.upload(local, "/remote/upload.bin")

    (duration,) = collected_metrics()[DURATION]
    assert duration.attributes[OUTCOME] == "error"
    assert local_stats(local) == 0


def test_transfer_instruments_declare_their_units(monkeypatch):
    """Backends scale and label by the declared unit, so bytes must say bytes and durations milliseconds."""
    created = {}

    class _Instrument:
        def record(self, *args, **kwargs):
            pass

        def add(self, *args, **kwargs):
            pass

    class _Meter:
        def create_histogram(self, name, **kwargs):
            created[name] = kwargs["unit"]
            return _Instrument()

        def create_counter(self, name, **kwargs):
            created[name] = kwargs["unit"]
            return _Instrument()

    monkeypatch.setattr(gym_metrics, "_meter", lambda: _Meter())
    gym_metrics._reset_for_testing()
    gym_metrics.record_sandbox_transfer(1.0, provider="fake", direction="upload", outcome="ok", size_bytes=10)
    gym_metrics._reset_for_testing()

    assert created == {DURATION: "ms", BYTES: "By"}
