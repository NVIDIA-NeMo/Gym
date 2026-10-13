# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sandbox utility helpers."""

import tempfile
from pathlib import Path

from nemo_gym.sandbox.api import AsyncSandbox


# Parallelism caps for CPU-limited sandboxes, each set by cpu_cap_env() to the
# floored CPU limit (min 1). Tools size worker pools by host core count, not
# the cgroup limit — Python's multiprocessing.cpu_count(), BLAS/OpenMP thread
# pools, Go's runtime, cargo, Node — so on wide hosts with small limits they
# overcommit, CFS-throttle, and OOM. GNU nproc honors OMP_NUM_THREADS, which
# also tames `make -j$(nproc)` builds; PYTHON_CPU_COUNT caps os/multiprocessing
# .cpu_count() itself on Python 3.13+. MAKEFLAGS is deliberately absent:
# injecting `-j` would flip plain `make` runs from serial to parallel, and
# explicit `-j` flags override MAKEFLAGS anyway. Injection is intentionally
# per-call-site (no sandbox-layer default); callers merge explicit env over it.
CPU_CAP_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "PYTHON_CPU_COUNT",
    "NUMBA_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
    "POLARS_MAX_THREADS",
    "PYTEST_XDIST_AUTO_NUM_WORKERS",
    "DJANGO_TEST_PROCESSES",
    "GOMAXPROCS",
    "CARGO_BUILD_JOBS",
    "RAYON_NUM_THREADS",
    "RUST_TEST_THREADS",
    "UV_THREADPOOL_SIZE",
    "CMAKE_BUILD_PARALLEL_LEVEL",
)


async def upload_text(sandbox: AsyncSandbox, *, path: str, text: str) -> None:
    """Upload UTF-8 text through file transfer without shell interpolation."""
    with tempfile.TemporaryDirectory(prefix="sandbox-upload-") as directory:
        source = Path(directory) / "payload"
        source.write_text(text, encoding="utf-8")
        await sandbox.upload(source, path)


async def read_text(sandbox: AsyncSandbox, *, path: str) -> str:
    """Download UTF-8 text, replacing invalid bytes in partial process output."""
    with tempfile.TemporaryDirectory(prefix="sandbox-download-") as directory:
        destination = Path(directory) / "payload"
        await sandbox.download(path, destination)
        return destination.read_text(encoding="utf-8", errors="replace")


def cpu_cap_env(cpu: float | int | None) -> dict[str, str]:
    """CPU_CAP_ENV_VARS derived from a sandbox CPU limit; empty when unset."""
    if cpu is None:
        return {}
    cores = str(max(1, int(cpu)))
    return {name: cores for name in CPU_CAP_ENV_VARS}


def rewrite_image(image: str | None, rewrites: list[dict[str, str]]) -> str | None:
    """Apply ordered image-prefix rewrites used by sandbox configs."""
    if image is None:
        return None
    for rewrite in rewrites:
        from_prefix = rewrite["from"]
        to_prefix = rewrite["to"]
        if image.startswith(from_prefix):
            return to_prefix + image[len(from_prefix) :]
    return image
