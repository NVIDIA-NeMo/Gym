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
"""Host memory sampling, called inline from span boundaries -- the same design as
`nemo_gym.telemetry.cpu`, not `nemo_gym.telemetry.gpu`.

`psutil.virtual_memory()` is a single, cheap syscall-backed read with no delta/priming
semantics (unlike `psutil.Process().cpu_percent()`), so there is even less reason here
than for CPU to avoid sampling inline: a reading taken inside a span's active context can
carry an OTel exemplar linking it back to that exact span, the same way CPU readings do.

This is host-wide, not process-scoped -- `virtual_memory()` reports the whole node's
memory, not this process's share of it. That is deliberate: unlike CPU (where "how busy
is this process" is the useful question) or GPU (owned by a specific, often different,
process), memory *pressure* is a node-level condition every process on that node shares,
so there is no meaningful "just this process's" reading to prefer instead.
"""

import logging
import threading
import time
from typing import Optional

import psutil


logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_LAST_USED_MIB: Optional[float] = None
_LAST_TOTAL_MIB: Optional[float] = None
_LAST_SAMPLE_TIME: float = 0.0

_BYTES_PER_MIB = 1024.0 * 1024.0


def sample_host_memory_mib(min_resample_interval_s: float) -> Optional[tuple]:
    """Return `(used_mib, total_mib)` for this node, or `None` on a `psutil.Error`.

    Resamples at most once per `min_resample_interval_s`; a call inside that window
    since the last real sample reuses the cached value, same rate-limiting shape as
    `nemo_gym.telemetry.cpu.sample_cpu_percent` (kept as a separate cache here rather
    than sharing one, since a caller may want CPU and memory readings taken at
    independent cadences).
    """
    global _LAST_USED_MIB, _LAST_TOTAL_MIB, _LAST_SAMPLE_TIME

    now = time.monotonic()
    with _LOCK:
        if now - _LAST_SAMPLE_TIME < min_resample_interval_s and _LAST_USED_MIB is not None:
            return (_LAST_USED_MIB, _LAST_TOTAL_MIB)

        try:
            virtual_memory = psutil.virtual_memory()
        except psutil.Error:
            logger.debug("memory sampler: psutil read failed", exc_info=True)
            if _LAST_USED_MIB is not None:
                return (_LAST_USED_MIB, _LAST_TOTAL_MIB)
            return None

        _LAST_USED_MIB = virtual_memory.used / _BYTES_PER_MIB
        _LAST_TOTAL_MIB = virtual_memory.total / _BYTES_PER_MIB
        _LAST_SAMPLE_TIME = now
        return (_LAST_USED_MIB, _LAST_TOTAL_MIB)


def _reset_for_testing() -> None:
    """Drop cached sampler state. Test-only."""
    global _LAST_USED_MIB, _LAST_TOTAL_MIB, _LAST_SAMPLE_TIME
    with _LOCK:
        _LAST_USED_MIB = None
        _LAST_TOTAL_MIB = None
        _LAST_SAMPLE_TIME = 0.0
