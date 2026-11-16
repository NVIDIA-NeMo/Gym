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

"""``gym.sandbox.active``: the Gym-owned up-down counter behind "sandboxes running now".

Asserted against a real in-memory OTel metric reader. The counter is created on the lens
meter directly (it is not one of lens's ``gym.*`` instruments), so these run wherever the
OTel SDK is importable, with or without nemo-lens.
"""

import pytest

from nemo_gym.sandbox import api as sandbox_api
from nemo_gym.sandbox.api import AsyncSandbox, SandboxSpec
from nemo_gym.telemetry import _fallbacks as telemetry_fallbacks
from nemo_gym.telemetry import gym_metrics as telemetry_metrics
from nemo_gym.telemetry import setup as telemetry_setup
from tests.unit_tests.sandbox.test_sandbox import FakeSandboxProvider
from tests.unit_tests.telemetry.in_memory_meter import collected_metrics  # noqa: F401 - fixture, re-exported


pytest.importorskip("opentelemetry.sdk.metrics")

PROVIDER_ATTR = telemetry_metrics.SANDBOX_PROVIDER_ATTRIBUTE
ACTIVE = telemetry_metrics.SANDBOX_ACTIVE_INSTRUMENT


def _active_by_provider(points) -> dict[str, int]:
    return {p.attributes[PROVIDER_ATTR]: p.value for p in points}


def set_sandbox_gates(monkeypatch, *, group: bool, exporter: bool) -> None:
    """Set the `sandbox` span group and the metrics exporter where the sandbox sites read them."""
    monkeypatch.setattr(telemetry_fallbacks, "is_span_group_enabled", lambda _: group)
    monkeypatch.setattr(sandbox_api, "is_span_group_enabled", lambda _: group)
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", exporter)


@pytest.fixture
def sandbox_group_on(monkeypatch):
    set_sandbox_gates(monkeypatch, group=True, exporter=False)


def test_recording_without_telemetry_is_a_no_op(monkeypatch):
    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    telemetry_metrics.record_sandbox_active(1, provider="fake")


def test_recording_errors_never_reach_the_caller(monkeypatch):
    class _Broken:
        is_exporting = True

        @property
        def meter(self):
            raise RuntimeError("meter is gone")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Broken())
    telemetry_metrics.record_sandbox_active(1, provider="fake")


def test_counter_is_non_monotonic_and_dimensioned_by_provider(collected_metrics):
    telemetry_metrics.record_sandbox_active(1, provider="opensandbox")
    telemetry_metrics.record_sandbox_active(1, provider="opensandbox")
    telemetry_metrics.record_sandbox_active(1, provider="local")
    telemetry_metrics.record_sandbox_active(-1, provider="opensandbox")

    points = collected_metrics()[ACTIVE]
    assert _active_by_provider(points) == {"opensandbox": 1, "local": 1}
    # A second reader collection must still see the same instrument, not a duplicate one.
    telemetry_metrics.record_sandbox_active(-1, provider="local")
    assert _active_by_provider(collected_metrics()[ACTIVE])["local"] == 0


def test_instrument_is_created_once_per_meter(collected_metrics):
    handle = telemetry_setup.get_telemetry()
    first = telemetry_metrics._get_or_create(handle.meter, ACTIVE, lambda: object())
    assert telemetry_metrics._get_or_create(handle.meter, ACTIVE, lambda: object()) is first


@pytest.mark.asyncio
async def test_sandbox_start_and_stop_move_the_counter(collected_metrics, sandbox_group_on):
    provider = FakeSandboxProvider()
    sandbox = AsyncSandbox(provider, SandboxSpec(image="img"))

    await sandbox.start()
    assert _active_by_provider(collected_metrics()[ACTIVE]) == {"fake": 1}

    await sandbox.stop()
    assert _active_by_provider(collected_metrics()[ACTIVE]) == {"fake": 0}

    # A second stop (idempotent path) must not subtract again.
    await sandbox.stop()
    assert _active_by_provider(collected_metrics()[ACTIVE]) == {"fake": 0}


@pytest.mark.asyncio
async def test_failed_start_does_not_count(collected_metrics, sandbox_group_on):
    class _Refusing(FakeSandboxProvider):
        async def create(self, spec):
            raise RuntimeError("no capacity")

    sandbox = AsyncSandbox(_Refusing(), SandboxSpec(image="img"))
    with pytest.raises(RuntimeError):
        await sandbox.start()
    assert ACTIVE not in collected_metrics()


@pytest.mark.asyncio
async def test_failed_initial_upload_releases_the_count(collected_metrics, sandbox_group_on):
    class _FailingUpload(FakeSandboxProvider):
        async def upload_file(self, handle, source_path, target_path):
            raise RuntimeError("upload failed")

    sandbox = AsyncSandbox(_FailingUpload(), SandboxSpec(image="img", files={"/tmp/a": "x"}))
    with pytest.raises(RuntimeError):
        await sandbox.start()
    assert _active_by_provider(collected_metrics()[ACTIVE]) == {"fake": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("owns_provider", [False, True])
async def test_failed_close_keeps_the_sandbox_counted_until_a_retry_releases_it(
    collected_metrics, sandbox_group_on, owns_provider
):
    """A close the provider rejected has not released anything; stop() stays retryable and the
    count drops on the attempt that succeeds, once."""

    class _FlakyClose(FakeSandboxProvider):
        fail = True

        async def close(self, handle):
            if self.fail:
                self.fail = False
                raise RuntimeError("close failed")
            await super().close(handle)

    sandbox = AsyncSandbox(_FlakyClose(), SandboxSpec(image="img"), owns_provider=owns_provider)
    await sandbox.start()
    with pytest.raises(RuntimeError):
        await sandbox.stop()
    assert _active_by_provider(collected_metrics()[ACTIVE]) == {"fake": 1}

    await sandbox.stop()
    assert _active_by_provider(collected_metrics()[ACTIVE]) == {"fake": 0}

    await sandbox.stop()
    assert _active_by_provider(collected_metrics()[ACTIVE]) == {"fake": 0}


@pytest.mark.asyncio
async def test_sandbox_records_nothing_without_the_group_or_a_metrics_exporter(collected_metrics, monkeypatch):
    set_sandbox_gates(monkeypatch, group=False, exporter=False)
    sandbox = AsyncSandbox(FakeSandboxProvider(), SandboxSpec(image="img"))
    await sandbox.start()
    await sandbox.stop()
    assert ACTIVE not in collected_metrics()


@pytest.mark.asyncio
async def test_connected_sandbox_is_not_subtracted_on_stop(collected_metrics, sandbox_group_on):
    """A sandbox rebuilt via connect() was counted by the process that started it."""
    from nemo_gym.sandbox.providers.base import ConnectableProvider, SandboxHandle

    class _Connectable(FakeSandboxProvider):
        async def connect(self, descriptor):
            return SandboxHandle(sandbox_id=descriptor["sandbox_id"], provider_name=self.name, raw={})

        async def serialize_handle(self, handle, *, scope=None):
            return {"sandbox_id": handle.sandbox_id}

    assert isinstance(_Connectable(), ConnectableProvider)
    sandbox = await AsyncSandbox.connect({"sandbox_id": "elsewhere"}, provider=_Connectable())
    await sandbox.stop()
    assert ACTIVE not in collected_metrics()
