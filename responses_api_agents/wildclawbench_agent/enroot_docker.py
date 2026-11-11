# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delegate Docker compatibility to the original shim while owning detached warmup groups."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

from responses_api_agents.wildclawbench_agent.data import digest, source_pin


def process_start(pid: int) -> str | None:
    """Reject a reused PID before signalling a recorded detached group."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if int(fields[2]) == pid and int(fields[3]) == pid else None
    except FileNotFoundError:
        return None


def stop_owned_groups(path: Path) -> None:
    if path.is_file():
        for record in json.loads(path.read_text()):
            pid = record["pid"]
            if record["start"] is not None and process_start(pid) == record["start"]:
                try:
                    os.killpg(pid, signal.SIGTERM)
                    time.sleep(0.2)
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


class TrackedSubprocess:
    """Record only the native shim's detached Popen, leaving its other operations unchanged."""

    def __init__(self, records: Path) -> None:
        self.records = records

    def __getattr__(self, name: str) -> object:
        return getattr(subprocess, name)

    def Popen(self, *args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(*args, **kwargs)
        try:
            records = json.loads(self.records.read_text()) if self.records.is_file() else []
            records.append(dict(pid=process.pid, start=process_start(process.pid)))
            self.records.write_text(json.dumps(records))
        except Exception:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            raise
        return process


def remove(shim: ModuleType, name: str) -> int:
    # state_path also validates the container identifier before any process or filesystem operation.
    state = shim.state_path(name)
    records = state.with_suffix(".processes.json")
    stop_owned_groups(records)
    rootfs = Path(os.environ["ENROOT_DATA_PATH"]) / name
    result = shim.remove_container([name]) if state.exists() or rootfs.exists() else 0
    if result == 0:
        records.unlink(missing_ok=True)
    return result


def main() -> int:
    source = Path(os.environ["WILDCLAW_ROOT"]) / "cluster/enroot_docker_shim.py"
    if digest(source) != source_pin()["files"]["cluster/enroot_docker_shim.py"]:
        raise ValueError("Original Enroot shim differs from the pinned source")
    spec = importlib.util.spec_from_file_location("wildclaw_original_docker_shim", source)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    operation, *arguments = sys.argv[1:]
    if operation == "rm":
        names = [name for name in arguments if name not in {"-f", "--force"}]
        return max((remove(shim, name) for name in names), default=0)
    if operation == "exec":
        index = 0
        while index < len(arguments) and arguments[index].startswith("-"):
            index += 2 if arguments[index] in {"-e", "--env", "-u", "--user", "-w", "--workdir"} else 1
        if index < len(arguments):
            state = shim.state_path(arguments[index])
            shim.subprocess = TrackedSubprocess(state.with_suffix(".processes.json"))
    return shim.main()


if __name__ == "__main__":
    raise SystemExit(main())
