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

import pytest

from nemo_gym.sandbox import api as sandbox_api
from nemo_gym.sandbox.api import AsyncSandbox, SandboxSpec
from nemo_gym.telemetry import gym_metrics
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics, sandbox_group_on  # noqa: F401 - fixtures
from tests.unit_tests.test_sandbox import FakeSandboxProvider


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


@pytest.mark.asyncio
async def test_seed_files_uploads_and_downloads_record_duration_and_bytes(
    collected_metrics,  # noqa: F811
    sandbox_group_on,  # noqa: F811
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
async def test_sandbox_group_off_records_no_transfers(collected_metrics, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(sandbox_api, "is_span_group_enabled", lambda group: False)
    provider = FakeSandboxProvider()
    sandbox = await AsyncSandbox(provider, SandboxSpec(image="img", files={"/seed.txt": "seed"})).start()
    local = tmp_path / "upload.bin"
    local.write_bytes(b"payload")
    await sandbox.upload(local, "/remote/upload.bin")
    assert len(provider.upload_calls) == 2
    assert DURATION not in collected_metrics() and BYTES not in collected_metrics()


@pytest.mark.asyncio
async def test_unreadable_local_file_never_fails_a_transfer(
    collected_metrics,  # noqa: F811
    sandbox_group_on,  # noqa: F811
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
