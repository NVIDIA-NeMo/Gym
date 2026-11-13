# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readers of a sandbox's measured CPU and memory for providers that implement ``SupportsSandboxUsage``.

``cgroup_v2_sample`` reads a cgroup the sandbox owns or shares.
``in_sandbox_cgroup_command`` and ``parse_in_sandbox_cgroup`` read the container's cgroup from inside it, for
runtimes whose host is not this one. Every reader returns None rather than a guess when it cannot measure.
"""

import time
from collections.abc import Mapping
from pathlib import Path

from nemo_gym.sandbox.providers.base import SandboxUsageSample, SandboxUsageScope


CGROUP_FILES = ("cpu.stat", "memory.current", "memory.peak", "memory.stat")


def _int_or_none(text: str | None) -> int | None:
    try:
        return int(text.strip()) if text is not None else None
    except ValueError:
        return None


def _keyed(text: str | None) -> dict[str, int]:
    """Parse ``key value`` lines (cpu.stat, memory.stat)."""
    values = {}
    for line in (text or "").splitlines():
        key, _, value = line.partition(" ")
        parsed = _int_or_none(value)
        if parsed is not None:
            values[key] = parsed
    return values


def parse_cgroup_v2(
    texts: Mapping[str, str | None], *, scope: SandboxUsageScope, sampled_at: float
) -> SandboxUsageSample | None:
    """A sample from the contents of a cgroup v2 directory's files, or None when it has no CPU or memory counter."""
    cpu = _keyed(texts.get("cpu.stat")).get("usage_usec")
    current = _int_or_none(texts.get("memory.current"))
    peak = _int_or_none(texts.get("memory.peak"))
    if cpu is None and current is None and peak is None:
        return None
    stat = _keyed(texts.get("memory.stat"))
    return SandboxUsageSample(
        source="cgroup_v2",
        scope=scope,
        sampled_at=sampled_at,
        cpu_usage_usec_total=cpu,
        cpu_kind="counter" if cpu is not None else None,
        memory_current_bytes=current,
        memory_peak_bytes=peak,
        peak_kind="counter" if peak is not None else None,
        memory_anon_bytes=stat.get("anon"),
        memory_file_bytes=stat.get("file"),
        includes_page_cache=True if current is not None or peak is not None else None,
    )


def cgroup_v2_sample(directory: Path, *, scope: SandboxUsageScope) -> SandboxUsageSample | None:
    """Read a cgroup v2 directory on this host."""
    texts = {}
    for name in CGROUP_FILES:
        try:
            texts[name] = (directory / name).read_text()
        except OSError:
            texts[name] = None
    return parse_cgroup_v2(texts, scope=scope, sampled_at=time.time())


def cgroup_path(pid: int | str, *, proc_root: Path = Path("/proc")) -> str | None:
    """The unified (v2) cgroup path of a process as this process's cgroup namespace sees it."""
    try:
        lines = (proc_root / str(pid) / "cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("0::"):
            return line[3:]
    return None


# A command printing the container's cgroup files from inside it, each after a ``==<name>`` line. Run where the
# sandbox has its own cgroup namespace (Docker and Kubernetes on cgroup v2), whose root is the container's cgroup.
IN_SANDBOX_CGROUP_COMMAND = (
    "for f in " + " ".join(CGROUP_FILES) + "; do printf '==%s\\n' \"$f\"; cat /sys/fs/cgroup/$f 2>/dev/null; done; "
    "printf '==cgroup\\n'; cat /proc/self/cgroup"
)


def parse_in_sandbox_cgroup(stdout: str, *, sampled_at: float) -> SandboxUsageSample | None:
    """A container-scope sample from ``IN_SANDBOX_CGROUP_COMMAND`` output.

    Only a reader at its cgroup namespace's root (``0::/``) sees the container's own cgroup. Without that
    namespace it sees the host hierarchy root, whose totals are not the sandbox's, so it returns None.
    """
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in stdout.splitlines():
        if line.startswith("=="):
            current = sections.setdefault(line[2:], [])
        elif current is not None:
            current.append(line)
    if "0::/" not in sections.get("cgroup", []):
        return None
    texts = {name: "\n".join(sections[name]) if sections.get(name) else None for name in CGROUP_FILES}
    return parse_cgroup_v2(texts, scope="container", sampled_at=sampled_at)
