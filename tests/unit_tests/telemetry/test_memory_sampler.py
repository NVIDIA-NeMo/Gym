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
"""``nemo_gym.telemetry.memory`` — the shared, inline-sampled host memory reader.

Same design as `cpu.py` (inline at span boundaries, cached, rate-limited) but with no
priming call: `psutil.virtual_memory()` has no delta semantics, so the first call already
returns a real reading, unlike `psutil.Process().cpu_percent()`.
"""

import psutil
import pytest

from nemo_gym.telemetry import memory


@pytest.fixture(autouse=True)
def _reset():
    memory._reset_for_testing()
    yield
    memory._reset_for_testing()


def _fake_virtual_memory(used_bytes, total_bytes):
    class _Result:
        used = used_bytes
        total = total_bytes

    return _Result()


def test_first_call_returns_a_real_reading_no_priming_needed(monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _fake_virtual_memory(1024**3, 8 * 1024**3))
    result = memory.sample_host_memory_mib(0.0)
    assert result == (1024.0, 8192.0)


def test_rate_limiting_reuses_the_cached_value_inside_the_window(monkeypatch):
    readings = iter([_fake_virtual_memory(1 * 1024**3, 8 * 1024**3), _fake_virtual_memory(2 * 1024**3, 8 * 1024**3)])
    read_count = [0]

    def fake_virtual_memory():
        read_count[0] += 1
        return next(readings)

    monkeypatch.setattr(psutil, "virtual_memory", fake_virtual_memory)

    first = memory.sample_host_memory_mib(min_resample_interval_s=1000.0)
    second = memory.sample_host_memory_mib(min_resample_interval_s=1000.0)

    assert first == second == (1024.0, 8192.0)
    assert read_count[0] == 1


def test_resampling_after_the_window_elapses_reads_again(monkeypatch):
    readings = iter([_fake_virtual_memory(1 * 1024**3, 8 * 1024**3), _fake_virtual_memory(2 * 1024**3, 8 * 1024**3)])
    monkeypatch.setattr(psutil, "virtual_memory", lambda: next(readings))

    first = memory.sample_host_memory_mib(min_resample_interval_s=0.0)
    second = memory.sample_host_memory_mib(min_resample_interval_s=0.0)

    assert first == (1024.0, 8192.0)
    assert second == (2048.0, 8192.0)


def test_psutil_error_returns_none_when_nothing_cached_yet(monkeypatch):
    def boom():
        raise psutil.Error("boom")

    monkeypatch.setattr(psutil, "virtual_memory", boom)
    assert memory.sample_host_memory_mib(0.0) is None


def test_psutil_error_returns_the_last_good_value_when_available(monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _fake_virtual_memory(1 * 1024**3, 8 * 1024**3))
    first = memory.sample_host_memory_mib(0.0)
    assert first == (1024.0, 8192.0)

    def boom():
        raise psutil.Error("boom")

    monkeypatch.setattr(psutil, "virtual_memory", boom)
    second = memory.sample_host_memory_mib(0.0)
    assert second == (1024.0, 8192.0)


def test_real_read_does_not_raise():
    """Smoke test against the real psutil implementation."""
    result = memory.sample_host_memory_mib(0.0)
    assert result is None or (result[0] >= 0 and result[1] > 0)
