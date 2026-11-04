# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run four disjoint component-test workers inside one CI server shard."""

import argparse
import os
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from time import monotonic
from types import FrameType


WORKERS = 4


def stop_workers(processes: list[subprocess.Popen[bytes]]) -> None:
    """Terminate active worker process groups and reap them within a bounded grace period."""
    active = [process for process in processes if process.poll() is None]
    for process in active:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = monotonic() + 10
    for process in active:
        try:
            process.wait(timeout=max(0, deadline - monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_workers(*, shard_index: int, num_shards: int, arguments: list[str]) -> int:
    """Preserve the original shard's selection and fail if any worker fails."""
    log_dir = Path("results/server-workers") / f"shard-{shard_index}"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    failed = False
    with ExitStack() as stack, ThreadPoolExecutor(max_workers=WORKERS) as executor:
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
                pending[executor.submit(process.wait)] = (worker, log_path)
            for future in as_completed(pending):
                worker, log_path = pending[future]
                status = future.result()
                failed |= status != 0
                print(f"Server worker {worker} exited {status}; log: {log_path}", flush=True)
                with log_path.open(errors="replace") as log_contents:
                    shutil.copyfileobj(log_contents, sys.stdout)
                sys.stdout.flush()
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
