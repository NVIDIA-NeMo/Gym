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
"""Process CPU sampling, called inline from span boundaries.

A gauge data point can only carry a span's trace/span id as an OTel exemplar if it is
recorded *while that span is the active context* -- exemplar capture reads the current
context at record time, and OTel context does not cross OS threads without explicit
propagation. That rules out a decoupled background-thread sampler: it would have no span
to link to. So this module has no thread and no start/stop lifecycle. Callers (
``nemo_gym.telemetry.endpoints.traced_endpoint``, ``AsyncSandbox.start``/``exec``, the
``gym.job`` span) call :func:`sample_cpu_percent` themselves, inside their own span's
``with managed_span(...) as span:`` block, right before recording the span attribute and
the ``gym.process.cpu.percent`` gauge.

``psutil.Process.cpu_percent(interval=None)`` only returns a meaningful value relative to
the *last* call on that same handle, so exactly one cached handle is shared by every
caller in this process -- two independent callers would otherwise steal each other's
measurement window. Concurrent/rapid span closes are rate-limited to one real psutil read
per ``min_resample_interval_s``; calls inside that window reuse the last cached value.
"""

import logging
import threading
import time
from typing import Optional

import psutil


logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PROCESS: Optional[psutil.Process] = None
_LAST_VALUE: Optional[float] = None
_LAST_SAMPLE_TIME: float = 0.0


def sample_cpu_percent(min_resample_interval_s: float) -> Optional[float]:
    """This process's CPU utilization (0-100 per logical core; can exceed 100 on a
    multi-threaded workload -- see ``nemo.gym.host.cpu_count`` for normalizing against
    node capacity).

    Resamples at most once per ``min_resample_interval_s``; a call inside that window
    since the last real sample reuses the cached value rather than issuing a fresh
    psutil read, keeping this cheap when many spans close in quick succession.

    Returns ``None`` on the very first call in this process (the priming call has
    nothing meaningful to report yet) and on any ``psutil.Error``.
    """
    global _PROCESS, _LAST_VALUE, _LAST_SAMPLE_TIME

    now = time.monotonic()
    with _LOCK:
        if _PROCESS is None:
            try:
                _PROCESS = psutil.Process()
                _PROCESS.cpu_percent(interval=None)  # prime -- first call is meaningless
            except psutil.Error:
                logger.debug("cpu sampler: failed to open psutil.Process()", exc_info=True)
                return None
            # Deliberately leave `_LAST_SAMPLE_TIME` at its initial 0.0 rather than
            # `now`: the first real sample after priming must not have to wait out a
            # full `min_resample_interval_s` window before it can happen at all.
            return None

        if now - _LAST_SAMPLE_TIME < min_resample_interval_s:
            return _LAST_VALUE

        try:
            _LAST_VALUE = _PROCESS.cpu_percent(interval=None)
        except psutil.Error:
            logger.debug("cpu sampler: psutil read failed", exc_info=True)
            return _LAST_VALUE
        _LAST_SAMPLE_TIME = now
        return _LAST_VALUE


def host_cpu_count_logical() -> Optional[int]:
    """Static logical CPU count for this node (includes SMT/hyperthreads), read once
    (not sampled). This is the count to normalize `gym.process.cpu.percent` against --
    `psutil`'s process CPU percentage is also computed with one logical core = 100%."""
    try:
        return psutil.cpu_count(logical=True)
    except psutil.Error:
        logger.debug("cpu sampler: failed to read logical cpu_count", exc_info=True)
        return None


def host_cpu_count_physical() -> Optional[int]:
    """Static physical CPU core count for this node (excludes SMT/hyperthreads), read
    once (not sampled). Not the right denominator for `gym.process.cpu.percent` -- use
    `host_cpu_count_logical` for that -- but useful on its own for raw compute-capacity
    reasoning. Can return `None` if `psutil` can't determine it on this platform, which
    is a real, distinct outcome from a read failure."""
    try:
        return psutil.cpu_count(logical=False)
    except psutil.Error:
        logger.debug("cpu sampler: failed to read physical cpu_count", exc_info=True)
        return None


def _reset_for_testing() -> None:
    """Drop cached sampler state. Test-only."""
    global _PROCESS, _LAST_VALUE, _LAST_SAMPLE_TIME
    with _LOCK:
        _PROCESS = None
        _LAST_VALUE = None
        _LAST_SAMPLE_TIME = 0.0
