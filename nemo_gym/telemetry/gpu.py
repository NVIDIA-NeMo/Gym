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
"""Process GPU sampling, on an independent background thread.

Unlike `nemo_gym.telemetry.cpu` (sampled inline at span boundaries so a reading can
carry an OTel exemplar linking it to the exact span that was running), GPU readings here
carry **no** exemplar and are correlated to a process only through that process's
resource attributes (`host.name`, `service.name`, `service.instance.id`). That is a
deliberate scope decision, not an oversight: GPU compute for a `local_vllm_model` server
happens in a *separate* Ray actor process from the one that owns Gym's spans, which has
no span context to link to in the first place -- doing this properly would need OTel
context propagated across the Ray actor boundary, which does not exist today.

Because there is no span-context constraint, there is also no reason to avoid a
background thread the way `cpu.py` does -- a daemon thread polling `nvidia-smi` on its
own cadence is simpler than threading this through every span-boundary call site, and it
decouples GPU sampling from request volume (every `nvidia-smi` call forks+execs a
subprocess, an order of magnitude more expensive than `cpu.py`'s cached-handle reads, so
it must run far less often -- see `TelemetryConfig.gpu_sample_interval_s`, default 10s).

`nvidia-smi` enumerates every physical GPU on the node regardless of
`CUDA_VISIBLE_DEVICES` -- unlike `psutil.Process()`, which is inherently process-scoped.
So every sample is filtered against this process's own `CUDA_VISIBLE_DEVICES`
(`_visible_gpu_selector`), mirroring how nemo-lens's own `_detect_gpu_count` resource
detector already reads that env var before falling back to `nvidia-smi`. Known caveat:
`responses_api_models/local_vllm_model/local_vllm_model_actor.py` deliberately clears
`CUDA_VISIBLE_DEVICES` and disables Ray's automatic per-actor GPU masking
(`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`), so in that actor this filter will
almost always fall back to "report everything `nvidia-smi` sees" -- harmless on a node
running exactly one actor, over-reporting (attributing other actors'/tenants' GPUs to
this one) on a node sharing GPUs across more than one actor/process. This is a real,
inherited limitation of that actor's existing GPU-visibility design, not something this
module can fix on its own.
"""

import logging
import subprocess
import threading
from typing import Optional


logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
_STOP_EVENT: Optional[threading.Event] = None

#: Sticky once set: `nvidia-smi` is absent on this host (no GPU, or a CPU-only image).
#: Set on the first `FileNotFoundError` so later ticks short-circuit before forking a
#: doomed subprocess again, rather than repeating a failure every interval forever. NOT
#: set on a transient failure (non-zero exit, timeout) -- those are worth retrying.
_NVIDIA_SMI_UNAVAILABLE = False

_NVIDIA_SMI_QUERY_FIELDS = "index,uuid,utilization.gpu,memory.used,memory.total"
_SUBPROCESS_TIMEOUT_S = 5.0


def start_gpu_sampler(interval_s: float) -> None:
    """Start the background sampler thread. Idempotent -- a second call while one is
    already running is a no-op."""
    global _THREAD, _STOP_EVENT
    with _LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return
        _STOP_EVENT = threading.Event()
        stop_event = _STOP_EVENT
        _THREAD = threading.Thread(
            target=_run,
            args=(stop_event, interval_s),
            name="nemo-gym-gpu-sampler",
            daemon=True,
        )
        _THREAD.start()


def stop_gpu_sampler(timeout_s: float = 2.0) -> None:
    """Stop the background sampler thread, if running. Idempotent, never raises."""
    global _THREAD, _STOP_EVENT
    with _LOCK:
        thread, stop_event = _THREAD, _STOP_EVENT
        _THREAD, _STOP_EVENT = None, None
    if stop_event is not None:
        stop_event.set()
    if thread is not None:
        try:
            thread.join(timeout=timeout_s)
        except Exception:
            logger.debug("gpu sampler: failed to join sampler thread", exc_info=True)


def _run(stop_event: threading.Event, interval_s: float) -> None:
    while True:
        try:
            _sample_once()
        except Exception:
            logger.debug("gpu sampler: unhandled error during sampling", exc_info=True)
        if stop_event.wait(interval_s):
            return


def _visible_gpu_selector() -> Optional[set]:
    """Parse `CUDA_VISIBLE_DEVICES` into a set of index strings or UUID strings this
    process is allowed to see. Returns `None` (meaning "report everything `nvidia-smi`
    sees") when the env var is unset, empty, or does not cleanly parse as either an
    all-index or all-uuid list -- the same fallback shape nemo-lens's own
    `_detect_gpu_count` uses.
    """
    import os

    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None

    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return None

    if all(token.isdigit() for token in tokens):
        return set(tokens)
    if all(token.startswith("GPU-") or token.startswith("MIG-") for token in tokens):
        return set(tokens)

    logger.debug("gpu sampler: could not parse CUDA_VISIBLE_DEVICES=%r; reporting all GPUs", raw)
    return None


def _parse_nvidia_smi_output(text: str) -> list:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` output into a
    list of `(index: int, uuid: str, utilization_pct: float, memory_used_mib: float,
    memory_total_mib: float)` tuples. A malformed row is skipped (logged at debug), not
    fatal to the rest of the batch.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            logger.debug("gpu sampler: unexpected nvidia-smi row shape: %r", line)
            continue
        try:
            index = int(fields[0])
            uuid = fields[1]
            utilization_pct = float(fields[2])
            memory_used_mib = float(fields[3])
            memory_total_mib = float(fields[4])
        except ValueError:
            logger.debug("gpu sampler: could not parse nvidia-smi row: %r", line)
            continue
        rows.append((index, uuid, utilization_pct, memory_used_mib, memory_total_mib))
    return rows


def _sample_once() -> None:
    global _NVIDIA_SMI_UNAVAILABLE

    if _NVIDIA_SMI_UNAVAILABLE:
        return

    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_NVIDIA_SMI_QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except FileNotFoundError:
        logger.debug("gpu sampler: nvidia-smi not found; disabling GPU sampling for this process")
        _NVIDIA_SMI_UNAVAILABLE = True
        return
    except subprocess.TimeoutExpired:
        logger.debug("gpu sampler: nvidia-smi timed out", exc_info=True)
        return
    except Exception:
        logger.debug("gpu sampler: failed to run nvidia-smi", exc_info=True)
        return

    if result.returncode != 0:
        logger.debug("gpu sampler: nvidia-smi exited %s: %s", result.returncode, result.stderr)
        return

    selector = _visible_gpu_selector()
    from nemo_gym.telemetry.gym_metrics import (
        record_process_gpu_memory_total_mib,
        record_process_gpu_memory_used_mib,
        record_process_gpu_utilization,
    )

    for index, uuid, utilization_pct, memory_used_mib, memory_total_mib in _parse_nvidia_smi_output(result.stdout):
        if selector is not None and str(index) not in selector and uuid not in selector:
            continue
        record_process_gpu_utilization(utilization_pct, index=index, uuid=uuid)
        record_process_gpu_memory_used_mib(memory_used_mib, index=index, uuid=uuid)
        record_process_gpu_memory_total_mib(memory_total_mib, index=index, uuid=uuid)


def _reset_for_testing() -> None:
    """Stop the sampler and clear sticky state. Test-only."""
    global _NVIDIA_SMI_UNAVAILABLE
    stop_gpu_sampler()
    _NVIDIA_SMI_UNAVAILABLE = False
