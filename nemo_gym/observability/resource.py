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

"""Best-effort in-sandbox resource sampling via cgroup counters."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from nemo_gym.observability.recorder import (
    SandboxEventRecorder,
    suppress_observability_events,
)


_RESOURCE_SAMPLE_COMMAND = r"""python3 - <<'PY'
import hashlib
import json
import os


def read_int(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def cgroup_memory():
    for path in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        value = read_int(path)
        if value is not None:
            return value
    return None


def cgroup_cpu_seconds():
    for path in (
        "/sys/fs/cgroup/cpu.stat",
        "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage",
    ):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            continue
        if "usage_usec" in text:
            for line in text.splitlines():
                key, _, value = line.partition(" ")
                if key == "usage_usec":
                    return int(value) / 1_000_000.0
        stripped = text.strip()
        if stripped.isdigit():
            return int(stripped) / 1_000_000_000.0
    total_ticks = 0
    ticks_per_second = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK"))
    try:
        pid_names = os.listdir("/proc")
    except Exception:
        return None
    for pid in pid_names:
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
                fields = f.read().split()
            total_ticks += int(fields[13]) + int(fields[14])
        except Exception:
            continue
    return total_ticks / ticks_per_second if ticks_per_second else None


def process_snapshot():
    max_processes = int(os.environ.get("NEMO_RL_PROCESS_TRACE_MAX_PROCESSES", "128"))
    include_cmdline = os.environ.get("NEMO_RL_PROCESS_TRACE_INCLUDE_CMDLINE") == "1"
    try:
        ticks_per_second = os.sysconf("SC_CLK_TCK")
    except Exception:
        ticks_per_second = 100
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except Exception:
        page_size = 4096

    rows = []
    try:
        pid_names = os.listdir("/proc")
    except Exception:
        return rows
    for pid_name in pid_names:
        if not pid_name.isdigit():
            continue
        pid = int(pid_name)
        try:
            stat_text = read_text(f"/proc/{pid_name}/stat")
            if not stat_text:
                continue
            before_comm, after_comm = stat_text.rsplit(")", 1)
            comm = before_comm.split("(", 1)[1]
            fields = after_comm.strip().split()
            state = fields[0]
            ppid = int(fields[1])
            user_ticks = int(fields[11])
            system_ticks = int(fields[12])
            start_time_ticks = int(fields[19])
        except Exception:
            continue

        rss_bytes = None
        try:
            statm_fields = read_text(f"/proc/{pid_name}/statm").split()
            rss_bytes = int(statm_fields[1]) * page_size
        except Exception:
            pass

        cpu_time_s = None
        if ticks_per_second:
            cpu_time_s = (user_ticks + system_ticks) / ticks_per_second

        command_text = comm
        try:
            with open(f"/proc/{pid_name}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").strip()
            if cmdline:
                command_text = cmdline.decode("utf-8", errors="replace")
        except Exception:
            pass

        row = {
            "pid": pid,
            "ppid": ppid,
            "state": state,
            "comm": comm[:120],
            "start_time_ticks": start_time_ticks,
            "cpu_time_s": cpu_time_s,
            "rss_bytes": rss_bytes,
            "cmdline_hash": hashlib.blake2s(
                command_text.encode("utf-8", errors="replace"),
                digest_size=16,
            ).hexdigest()[:12],
        }
        if include_cmdline:
            row["cmdline"] = command_text[:512]
        else:
            row["cmdline_redacted"] = True
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["rss_bytes"] if isinstance(row.get("rss_bytes"), int) else -1,
            row["cpu_time_s"] if isinstance(row.get("cpu_time_s"), (int, float)) else -1,
        ),
        reverse=True,
    )
    return rows[:max_processes] if max_processes > 0 else rows


process_count = 0
try:
    process_count = sum(1 for name in os.listdir("/proc") if name.isdigit())
except Exception:
    process_count = None

payload = {
    "memory_usage_bytes": cgroup_memory(),
    "cpu_usage_seconds_total": cgroup_cpu_seconds(),
    "process_count": process_count,
}
if os.environ.get("NEMO_RL_PROCESS_TRACE") == "1":
    payload["processes"] = process_snapshot()
print(json.dumps(payload, separators=(",", ":")))
PY"""


def _resource_sample_command(process_trace: dict[str, Any]) -> str:
    enabled = bool(process_trace.get("enabled"))
    max_processes = int(process_trace.get("max_processes_per_sample") or 0)
    include_cmdline = bool(process_trace.get("include_cmdline"))
    return (
        f"NEMO_RL_PROCESS_TRACE={int(enabled)} "
        f"NEMO_RL_PROCESS_TRACE_MAX_PROCESSES={max_processes} "
        f"NEMO_RL_PROCESS_TRACE_INCLUDE_CMDLINE={int(include_cmdline)} "
        f"{_RESOURCE_SAMPLE_COMMAND}"
    )


class SandboxResourceSampler:
    """Periodically sample resource usage from inside one sandbox.

    Runs a Python heredoc via ``provider.exec`` to read cgroup memory/CPU counters
    without needing any external tools installed in the sandbox image.
    """

    def __init__(
        self,
        *,
        provider: Any,
        handle: Any,
        recorder: SandboxEventRecorder,
        interval_s: float,
        process_trace: dict[str, Any] | None = None,
        attributes: dict[str, Any],
        otel_span: Any = None,
    ) -> None:
        self._provider = provider
        self._handle = handle
        self._recorder = recorder
        self._interval_s = interval_s
        self._process_trace = process_trace or {"enabled": False}
        self._attributes = attributes
        self._otel_span = otel_span  # Optional OTel span; resource samples are added as span events
        self._task: asyncio.Task[None] | None = None
        self._last_cpu_usage_s: float | None = None
        self._last_sample_monotonic_s: float | None = None

    def start(self) -> None:
        """Start the background sampler task."""
        if self._interval_s <= 0 or self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the background sampler task."""
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _run(self) -> None:
        while True:
            await self._sample_once()
            await asyncio.sleep(self._interval_s)

    async def _sample_once(self) -> None:
        try:
            with suppress_observability_events():
                result = await self._provider.exec(
                    self._handle,
                    _resource_sample_command(self._process_trace),
                    timeout_s=20,
                    user="root",
                )
            if result.return_code != 0 or not result.stdout:
                self._recorder.record_event(
                    "error",
                    "sandbox.resource_sample_error",
                    attributes={
                        **self._attributes,
                        "return_code": result.return_code,
                    },
                )
                return
            sample = json.loads(result.stdout)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            self._recorder.record_event(
                "error",
                "sandbox.resource_sample_error",
                attributes={**self._attributes, "error_type": type(e).__name__},
            )
            return

        now = asyncio.get_running_loop().time()
        cpu_usage_s = sample.get("cpu_usage_seconds_total")
        cpu_utilization = None
        if (
            isinstance(cpu_usage_s, (int, float))
            and self._last_cpu_usage_s is not None
            and self._last_sample_monotonic_s is not None
        ):
            elapsed_s = max(now - self._last_sample_monotonic_s, 1e-6)
            cpu_utilization = max(0.0, (float(cpu_usage_s) - self._last_cpu_usage_s) / elapsed_s)
        if isinstance(cpu_usage_s, (int, float)):
            self._last_cpu_usage_s = float(cpu_usage_s)
            self._last_sample_monotonic_s = now
        sample["cpu_utilization"] = cpu_utilization
        self._recorder.record_resource_sample(
            sandbox_id=getattr(self._handle, "sandbox_id", None),
            sample=sample,
            attributes=self._attributes,
        )
        # Attach resource data as a timestamped event on the sandbox OTel span so
        # CPU/memory appear inline with tool call spans in Honeycomb / Tempo / Jaeger.
        if self._otel_span is not None:
            try:
                if getattr(self._otel_span, "is_recording", lambda: False)():
                    event_attrs = {
                        k: v
                        for k, v in {
                            "memory_usage_bytes": sample.get("memory_usage_bytes"),
                            "cpu_utilization": sample.get("cpu_utilization"),
                            "process_count": sample.get("process_count"),
                        }.items()
                        if v is not None
                    }
                    self._otel_span.add_event("resource.sample", event_attrs)
            except Exception:
                pass
