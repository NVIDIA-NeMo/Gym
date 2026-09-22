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


_PROCESS_LOCK = threading.Lock()
_PROCESS: Optional[psutil.Process] = None
_LAST_PROCESS_TREE_RSS_MIB: Optional[float] = None
_LAST_PROCESS_SAMPLE_TIME: float = 0.0


def sample_process_tree_memory_mib(min_resample_interval_s: float) -> Optional[float]:
    """This process's RSS plus every live child process's RSS, recursively summed --
    the process-scoped counterpart to :func:`sample_host_memory_mib`'s host-wide
    reading, and the memory half of "the actual job workload"
    (:func:`nemo_gym.telemetry.cpu.sample_process_tree_cpu_percent` is the CPU half).

    Unlike CPU, RSS needs no delta/priming: `psutil.Process.memory_info().rss` is a
    direct read, so a child contributes its full reading from the very first sample that
    sees it (no "0 on first sight" caveat like the CPU tree sampler has). Still
    rate-limited the same way, so a burst of span closes does not turn into a burst of
    `/proc` reads across a potentially large child-process tree.

    Returns ``None`` on any ``psutil.Error`` reading the root process's own children list,
    including on the very first call.
    """
    global _PROCESS, _LAST_PROCESS_TREE_RSS_MIB, _LAST_PROCESS_SAMPLE_TIME

    now = time.monotonic()
    with _PROCESS_LOCK:
        if now - _LAST_PROCESS_SAMPLE_TIME < min_resample_interval_s and _LAST_PROCESS_TREE_RSS_MIB is not None:
            return _LAST_PROCESS_TREE_RSS_MIB

        if _PROCESS is None:
            try:
                _PROCESS = psutil.Process()
            except psutil.Error:
                logger.debug("memory sampler: failed to open psutil.Process()", exc_info=True)
                return None

        try:
            total_bytes = _PROCESS.memory_info().rss
            for child in _PROCESS.children(recursive=True):
                try:
                    total_bytes += child.memory_info().rss
                except psutil.Error:
                    continue  # exited between children() and this read -- skip, not fatal
        except psutil.Error:
            logger.debug("memory sampler: process-tree RSS read failed", exc_info=True)
            return _LAST_PROCESS_TREE_RSS_MIB

        _LAST_PROCESS_TREE_RSS_MIB = total_bytes / _BYTES_PER_MIB
        _LAST_PROCESS_SAMPLE_TIME = now
        return _LAST_PROCESS_TREE_RSS_MIB


def _reset_for_testing() -> None:
    """Drop cached sampler state. Test-only."""
    global _LAST_USED_MIB, _LAST_TOTAL_MIB, _LAST_SAMPLE_TIME
    global _PROCESS, _LAST_PROCESS_TREE_RSS_MIB, _LAST_PROCESS_SAMPLE_TIME
    with _LOCK:
        _LAST_USED_MIB = None
        _LAST_TOTAL_MIB = None
        _LAST_SAMPLE_TIME = 0.0
    with _PROCESS_LOCK:
        _PROCESS = None
        _LAST_PROCESS_TREE_RSS_MIB = None
        _LAST_PROCESS_SAMPLE_TIME = 0.0
