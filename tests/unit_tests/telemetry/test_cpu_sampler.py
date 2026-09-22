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
"""``nemo_gym.telemetry.cpu`` — the shared, inline-sampled CPU reader.

No thread, no lifecycle: every call site samples synchronously at a span boundary, so
these tests exercise `sample_cpu_percent` and `host_cpu_count_logical`/`host_cpu_count_physical` directly rather than any
start/stop machinery (there isn't any -- see the module docstring for why).
"""

import psutil
import pytest

from nemo_gym.telemetry import cpu


@pytest.fixture(autouse=True)
def _reset():
    cpu._reset_for_testing()
    yield
    cpu._reset_for_testing()


def test_first_call_primes_and_returns_none():
    assert cpu.sample_cpu_percent(0.0) is None


def test_second_call_returns_a_value():
    cpu.sample_cpu_percent(0.0)
    value = cpu.sample_cpu_percent(0.0)
    assert value is None or isinstance(value, float)


def test_psutil_process_is_instantiated_exactly_once(monkeypatch):
    calls = []

    class _FakeProcess:
        def __init__(self):
            calls.append(1)

        def cpu_percent(self, interval=None):
            return 12.5

    monkeypatch.setattr(psutil, "Process", _FakeProcess)

    cpu.sample_cpu_percent(0.0)
    cpu.sample_cpu_percent(0.0)
    cpu.sample_cpu_percent(0.0)

    assert len(calls) == 1


def test_rate_limiting_reuses_the_cached_value_inside_the_window(monkeypatch):
    readings = iter([0.0, 1.0, 2.0, 3.0])  # first value consumed by the priming call
    read_count = [0]

    class _FakeProcess:
        def cpu_percent(self, interval=None):
            read_count[0] += 1
            return next(readings)

    monkeypatch.setattr(psutil, "Process", lambda: _FakeProcess())

    assert cpu.sample_cpu_percent(min_resample_interval_s=1000.0) is None  # priming call
    first = cpu.sample_cpu_percent(min_resample_interval_s=1000.0)
    second = cpu.sample_cpu_percent(min_resample_interval_s=1000.0)

    assert first == second == 1.0
    # One real read for the priming call (discarded) and one for the first real sample;
    # the second call reused the cache instead of reading again.
    assert read_count[0] == 2


def test_resampling_after_the_window_elapses_reads_again(monkeypatch):
    readings = iter([0.0, 1.0, 2.0])  # first value consumed by the priming call

    class _FakeProcess:
        def cpu_percent(self, interval=None):
            return next(readings)

    monkeypatch.setattr(psutil, "Process", lambda: _FakeProcess())

    cpu.sample_cpu_percent(min_resample_interval_s=0.0)  # priming call
    first = cpu.sample_cpu_percent(min_resample_interval_s=0.0)
    second = cpu.sample_cpu_percent(min_resample_interval_s=0.0)

    assert first == 1.0
    assert second == 2.0


def test_psutil_error_on_priming_is_swallowed(monkeypatch):
    class _BrokenProcess:
        def __init__(self):
            raise psutil.Error("boom")

    monkeypatch.setattr(psutil, "Process", _BrokenProcess)
    assert cpu.sample_cpu_percent(0.0) is None


def test_psutil_error_during_priming_is_swallowed_and_retried_next_call(monkeypatch):
    """The priming read itself can raise (Process() succeeds, cpu_percent() doesn't) --
    that must not poison the cached handle. The next call retries a real read rather
    than getting stuck returning None forever."""
    calls = [0]

    class _FlakyProcess:
        def cpu_percent(self, interval=None):
            calls[0] += 1
            if calls[0] == 1:
                raise psutil.Error("boom")
            return 42.0

    monkeypatch.setattr(psutil, "Process", lambda: _FlakyProcess())

    assert cpu.sample_cpu_percent(0.0) is None  # priming call's own read raised
    assert cpu.sample_cpu_percent(0.0) == 42.0  # next call succeeds


def test_host_cpu_count_logical_returns_a_positive_int():
    count = cpu.host_cpu_count_logical()
    assert count is None or count >= 1


def test_host_cpu_count_physical_returns_a_positive_int_or_none():
    """Physical count can legitimately be None on platforms psutil can't determine it
    for -- a distinct, expected outcome, not a read failure."""
    count = cpu.host_cpu_count_physical()
    assert count is None or count >= 1


def test_host_cpu_count_logical_and_physical_are_independent(monkeypatch):
    calls = []

    def fake_cpu_count(logical=True):
        calls.append(logical)
        return 24 if logical else 12

    monkeypatch.setattr(psutil, "cpu_count", fake_cpu_count)

    assert cpu.host_cpu_count_logical() == 24
    assert cpu.host_cpu_count_physical() == 12
    assert calls == [True, False]


def test_host_cpu_count_logical_swallows_psutil_errors(monkeypatch):
    def boom(logical=True):
        raise psutil.Error("boom")

    monkeypatch.setattr(psutil, "cpu_count", boom)
    assert cpu.host_cpu_count_logical() is None


def test_host_cpu_count_physical_swallows_psutil_errors(monkeypatch):
    def boom(logical=True):
        raise psutil.Error("boom")

    monkeypatch.setattr(psutil, "cpu_count", boom)
    assert cpu.host_cpu_count_physical() is None
