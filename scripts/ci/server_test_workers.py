# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run four disjoint component-test workers inside one CI server shard."""

import argparse
import os
import shutil
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from time import monotonic, sleep
from types import FrameType


WORKERS = 4


def signal_groups(groups: set[int], signum: int) -> None:
    """Signal owned groups whose leader PIDs have not been reaped or released for reuse."""
    for group in groups:
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            pass


def live_groups(groups: set[int]) -> set[int]:
    """Find live members on Linux, excluding the unreaped leaders that reserve group IDs."""
    live = set()
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_bytes().rsplit(b")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        group = int(fields[2])
        if group in groups and fields[0] not in (b"Z", b"X"):
            live.add(group)
    return live


def stop_workers(processes: list[subprocess.Popen[bytes]]) -> None:
    """Bound group cleanup before reaping any leader, preserving ownership even after leader exit."""
    groups = {process.pid for process in processes}
    try:
        signal_groups(groups, signal.SIGTERM)
        deadline = monotonic() + 10
        while live_groups(groups) and monotonic() < deadline:
            sleep(0.05)
    finally:
        signal_groups(groups, signal.SIGKILL)
        deadline = monotonic() + 2
        try:
            while live_groups(groups):
                if monotonic() >= deadline:
                    raise TimeoutError("Server worker groups remained alive after SIGKILL")
                sleep(0.01)
        finally:
            for process in processes:
                process.wait(timeout=max(0, deadline - monotonic()))


def worker_exit_status(process: subprocess.Popen[bytes]) -> int | None:
    """Observe exit without reaping: the leader PID reserves its group until cleanup finishes."""
    result = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if result is None:
        return None
    return result.si_status if result.si_code == os.CLD_EXITED else -result.si_status


def run_workers(*, shard_index: int, num_shards: int, arguments: list[str]) -> int:
    """Preserve the original shard's selection and fail if any worker fails."""
    log_dir = Path("results/server-workers") / f"shard-{shard_index}"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    failed = False
    with ExitStack() as stack:
        try:
            pending = {}
            for worker in range(WORKERS):
                log_path = log_dir / f"worker-{worker}.log"
                command = [
                    "ng_test_all",
                    *arguments,
                    f"+num_shards={num_shards * WORKERS}",
                    f"+shard_index={shard_index + worker * num_shards}",
                ]
                print(f"Starting server worker {worker}: {' '.join(command)}; log: {log_path}", flush=True)
                log_output = stack.enter_context(log_path.open("wb"))
                process = subprocess.Popen(
                    command, stdout=log_output, stderr=subprocess.STDOUT, start_new_session=True
                )
                processes.append(process)
                pending[process] = (worker, log_path)
            while pending:
                for process in list(pending):
                    status = worker_exit_status(process)
                    if status is None:
                        continue
                    worker, log_path = pending.pop(process)
                    failed |= status != 0
                    print(f"Server worker {worker} exited {status}; log: {log_path}", flush=True)
                    with log_path.open(errors="replace") as log_contents:
                        shutil.copyfileobj(log_contents, sys.stdout)
                    sys.stdout.flush()
                if pending:
                    sleep(0.05)
        finally:
            stop_workers(processes)
    return int(failed)


def terminate(signum: int, frame: FrameType | None) -> None:
    """Unwind the launcher so cancellation cleans up every active worker."""
    raise SystemExit(128 + signum)


def main() -> int:
    """Validate shard selection before starting the four component-test workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_index", type=int)
    parser.add_argument("num_shards", type=int)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("Expected 0 <= shard_index < num_shards")
    if any(value.startswith(("+shard_index=", "+num_shards=")) for value in args.arguments):
        parser.error("Pass shard selection through the positional arguments")
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    return run_workers(shard_index=args.shard_index, num_shards=args.num_shards, arguments=args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
