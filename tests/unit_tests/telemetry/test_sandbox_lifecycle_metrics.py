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

"""Sandbox lifecycle histograms and the create-retry counter, against a real in-memory reader.

Runs wherever the OTel SDK is importable; the span-side assertions (sandbox id attributes,
the stop span) live in test_call_sites.py because they need nemo-lens.
"""

import pytest

from nemo_gym.sandbox import api as sandbox_api
from nemo_gym.sandbox.api import AsyncSandbox, SandboxSpec
from nemo_gym.telemetry import gym_metrics
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics, sandbox_group_on  # noqa: F401 - fixtures
from tests.unit_tests.test_sandbox import FakeSandboxProvider


pytest.importorskip("opentelemetry.sdk.metrics")

PROVIDER = gym_metrics.SANDBOX_PROVIDER_ATTRIBUTE
STARTUP = gym_metrics.SANDBOX_STARTUP_INSTRUMENT
EXEC = gym_metrics.SANDBOX_EXEC_INSTRUMENT


def _points(collected, name):
    return collected().get(name, [])


@pytest.mark.asyncio
async def test_start_and_exec_record_one_duration_sample_each(collected_metrics, sandbox_group_on):  # noqa: F811
    sandbox = AsyncSandbox(FakeSandboxProvider(), SandboxSpec(image="img"))
    await sandbox.start()
    await sandbox.exec("echo one")
    await sandbox.exec("echo two")

    (startup,) = _points(collected_metrics, STARTUP)
    assert startup.attributes == {PROVIDER: "fake"}
    assert startup.count == 1 and startup.sum >= 0

    (execs,) = _points(collected_metrics, EXEC)
    assert execs.attributes == {PROVIDER: "fake"}
    assert execs.count == 2


@pytest.mark.asyncio
async def test_duration_histograms_use_explicit_boundaries_up_to_thirty_minutes(collected_metrics, sandbox_group_on):  # noqa: F811
    sandbox = AsyncSandbox(FakeSandboxProvider(), SandboxSpec(image="img"))
    await sandbox.start()
    (startup,) = _points(collected_metrics, STARTUP)
    assert list(startup.explicit_bounds) == list(gym_metrics.SANDBOX_DURATION_BOUNDARIES_MS)
    assert startup.explicit_bounds[-1] == 1_800_000


@pytest.mark.asyncio
async def test_failed_create_records_no_startup_sample(collected_metrics, sandbox_group_on):  # noqa: F811
    class _Refusing(FakeSandboxProvider):
        async def create(self, spec):
            raise RuntimeError("no capacity")

    with pytest.raises(RuntimeError):
        await AsyncSandbox(_Refusing(), SandboxSpec(image="img")).start()
    assert STARTUP not in collected_metrics()


@pytest.mark.asyncio
async def test_sandbox_group_off_records_no_durations(collected_metrics, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sandbox_api, "is_span_group_enabled", lambda group: False)
    sandbox = AsyncSandbox(FakeSandboxProvider(), SandboxSpec(image="img"))
    await sandbox.start()
    await sandbox.exec("echo hi")
    assert STARTUP not in collected_metrics() and EXEC not in collected_metrics()
