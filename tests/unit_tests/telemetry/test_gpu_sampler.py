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
"""``nemo_gym.telemetry.gpu`` — the background, thread-based GPU sampler.

Unlike `cpu.py`'s inline sampler, this one has a real lifecycle (start/stop a thread) and
shells out to `nvidia-smi`, so `subprocess.run` is mocked throughout -- no real GPU or
`nvidia-smi` binary is required to run these tests.
"""

import subprocess
import time

import pytest

from nemo_gym.telemetry import gpu


@pytest.fixture(autouse=True)
def _reset():
    gpu._reset_for_testing()
    yield
    gpu._reset_for_testing()


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parse_well_formed_multi_row_output():
    text = "0, GPU-aaaa, 10, 1024, 8192\n1, GPU-bbbb, 20, 2048, 8192\n"
    rows = gpu._parse_nvidia_smi_output(text)
    assert rows == [
        (0, "GPU-aaaa", 10.0, 1024.0, 8192.0),
        (1, "GPU-bbbb", 20.0, 2048.0, 8192.0),
    ]


def test_parse_tolerates_trailing_blank_lines():
    text = "0, GPU-aaaa, 10, 1024, 8192\n\n\n"
    rows = gpu._parse_nvidia_smi_output(text)
    assert len(rows) == 1


def test_parse_skips_a_malformed_row_without_failing_the_batch():
    text = "0, GPU-aaaa, 10, 1024, 8192\nnot,a,valid,row\n1, GPU-bbbb, 20, 2048, 8192\n"
    rows = gpu._parse_nvidia_smi_output(text)
    assert [row[0] for row in rows] == [0, 1]


def test_parse_skips_a_row_with_the_wrong_field_count():
    text = "0, GPU-aaaa, 10, 1024, 8192\n0, GPU-aaaa, 10\n"
    rows = gpu._parse_nvidia_smi_output(text)
    assert len(rows) == 1


# --------------------------------------------------------------------------- #
# CUDA_VISIBLE_DEVICES filtering -- the property most likely to silently regress
# --------------------------------------------------------------------------- #


def test_selector_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gpu._visible_gpu_selector() is None


def test_selector_is_none_when_empty_string(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu._visible_gpu_selector() is None


def test_selector_filters_by_index(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    assert gpu._visible_gpu_selector() == {"0", "2"}


def test_selector_filters_by_uuid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa,GPU-bbbb")
    assert gpu._visible_gpu_selector() == {"GPU-aaaa", "GPU-bbbb"}


def test_selector_falls_back_to_none_on_malformed_value(monkeypatch, caplog):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,GPU-aaaa")  # mixed forms
    import logging

    with caplog.at_level(logging.DEBUG, logger="nemo_gym.telemetry.gpu"):
        assert gpu._visible_gpu_selector() is None


def test_sample_once_only_records_gpus_in_the_selector(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    class _Result:
        returncode = 0
        stdout = "0, GPU-aaaa, 10, 1024, 8192\n1, GPU-bbbb, 20, 2048, 8192\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())

    recorded = []
    monkeypatch.setattr(
        "nemo_gym.telemetry.gym_metrics.record_process_gpu_utilization",
        lambda value, *, index, uuid: recorded.append((index, uuid, value)),
    )
    monkeypatch.setattr("nemo_gym.telemetry.gym_metrics.record_process_gpu_memory_used_mib", lambda *a, **k: None)
    monkeypatch.setattr("nemo_gym.telemetry.gym_metrics.record_process_gpu_memory_total_mib", lambda *a, **k: None)

    gpu._sample_once()

    assert recorded == [(0, "GPU-aaaa", 10.0)]


# --------------------------------------------------------------------------- #
# Degrade-to-no-op
# --------------------------------------------------------------------------- #


def test_missing_nvidia_smi_sets_the_sticky_flag_and_stops_spawning(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise FileNotFoundError("no nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)

    gpu._sample_once()
    gpu._sample_once()
    gpu._sample_once()

    assert len(calls) == 1
    assert gpu._NVIDIA_SMI_UNAVAILABLE is True


def test_nonzero_return_code_is_not_sticky(monkeypatch):
    calls = []

    class _Result:
        returncode = 1
        stdout = ""
        stderr = "driver error"

    def fake_run(*a, **k):
        calls.append(1)
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    gpu._sample_once()
    gpu._sample_once()

    assert len(calls) == 2  # retried, not sticky
    assert gpu._NVIDIA_SMI_UNAVAILABLE is False


def test_timeout_is_not_sticky(monkeypatch):
    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    gpu._sample_once()  # must not raise

    assert gpu._NVIDIA_SMI_UNAVAILABLE is False


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_start_then_stop_leaves_no_thread_alive(monkeypatch):
    monkeypatch.setattr(gpu, "_sample_once", lambda: None)

    gpu.start_gpu_sampler(0.01)
    time.sleep(0.05)
    gpu.stop_gpu_sampler()

    assert gpu._THREAD is None


def test_start_is_idempotent(monkeypatch):
    monkeypatch.setattr(gpu, "_sample_once", lambda: None)

    gpu.start_gpu_sampler(1.0)
    first_thread = gpu._THREAD
    gpu.start_gpu_sampler(1.0)

    assert gpu._THREAD is first_thread
    gpu.stop_gpu_sampler()


def test_stop_without_start_is_a_safe_no_op():
    gpu.stop_gpu_sampler()  # must not raise


def test_sample_once_is_actually_invoked_by_the_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu, "_sample_once", lambda: calls.append(1))

    gpu.start_gpu_sampler(0.01)
    time.sleep(0.1)
    gpu.stop_gpu_sampler()

    assert len(calls) >= 1
