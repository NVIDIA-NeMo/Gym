#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate a diagnostic window around the unchanged Gym workload.

Used by prepare_recipe.py. The worker sidecars and benchmark container must share
the SRT /logs mount. A failed collector records an error without failing Gym.
The workload's exit status is preserved. Profile requests are never retried.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from collect import save, utc


HERE = Path(__file__).resolve().parent


def stop(process: subprocess.Popen | None) -> None:
    """Stop a child process group, including a collector's nvidia-smi process."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def launch(arguments: list[str]) -> subprocess.Popen:
    return subprocess.Popen(arguments, start_new_session=True)


def runtime_identity() -> dict[str, object]:
    """Record installed package identities without importing GPU runtimes."""
    packages = {}
    for name in ("sglang", "torch", "flashinfer-python", "nixl"):
        try:
            distribution = importlib.metadata.distribution(name)
            direct = json.loads(distribution.read_text("direct_url.json") or "{}")
            packages[name] = {"version": distribution.version, "commit": direct.get("vcs_info", {}).get("commit_id")}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"host": socket.gethostname(), "utc": utc(), "python": sys.version, "packages": packages}


def worker(args: argparse.Namespace) -> int:
    """Sample every visible GPU during the controller's unprofiled metric window."""
    host = socket.gethostname()
    status_dir = args.root / "workers" / host
    status_dir.mkdir(parents=True, exist_ok=False)
    save(status_dir / "runtime.json", runtime_identity())
    process = None
    try:
        marker = args.root / "metrics-window.json"
        while not marker.exists():
            if (args.root / "done.json").exists():
                return 0
            time.sleep(1)
        window = json.loads(marker.read_text())
        remaining = window["end_epoch"] - time.time()
        if remaining <= 0:
            raise RuntimeError("Worker missed the metric window; check shared /logs and service startup")
        process = launch(
            [
                sys.executable,
                str(HERE / "collect.py"),
                "gpu",
                "--out",
                str(args.root / f"gpu-{host}"),
                "--seconds",
                str(remaining),
                "--interval",
                "1",
            ]
        )
        returncode = process.wait()
        save(status_dir / "result.json", {"returncode": returncode, "utc": utc()})
        return returncode
    except Exception as error:
        save(status_dir / "ERROR.json", {"utc": utc(), "error": str(error)})
        return 1
    finally:
        stop(process)


def wait_collector(process: subprocess.Popen, workload: subprocess.Popen) -> bool:
    """Do not let evidence collection outlive the workload it measures."""
    while process.poll() is None:
        if workload.poll() is not None:
            stop(process)
            return False
        time.sleep(0.2)
    if process.returncode:
        raise RuntimeError(f"Collector exited with status {process.returncode}; inspect diagnostics.log")
    return True


def finalize(root: Path, logs_root: Path) -> None:
    """Preserve visible worker logs and Gym artifacts, then analyze collected evidence."""
    logs = sorted(
        path
        for path in logs_root.rglob("*")
        if path.is_file()
        and path.suffix in (".log", ".out")
        and root not in path.parents
        and any(role in str(path.relative_to(logs_root)).lower() for role in ("prefill", "decode", "worker"))
    )
    if logs:
        arguments = [sys.executable, str(HERE / "collect.py"), "startup", "--out", str(root / "startup")]
        for path in logs:
            arguments.extend(["--log", str(path)])
        subprocess.run(arguments, check=False)
    artifacts = []
    for path in (logs_root / "gym").rglob("*"):
        if path.is_file() and (path.suffix in (".jsonl", ".csv") or path.name == "inference-metrics.yaml"):
            target = root / "gym-artifacts" / path.relative_to(logs_root / "gym")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            artifacts.append(str(target.relative_to(root)))
    save(
        root / "artifact-inventory.json",
        {
            "worker_logs": [str(path) for path in logs],
            "gym_artifacts": artifacts,
            "note": "Snapshots at benchmark exit. Preserve SRT's resolved recipe and full job logs after shutdown as well.",
        },
    )
    subprocess.run([sys.executable, str(HERE / "analyze.py"), str(root)], check=False)


def benchmark(args: argparse.Namespace) -> int:
    """Run the original benchmark, collecting once after the configured warm-up delay."""
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("benchmark mode requires a workload command after --")
    args.root.mkdir(parents=True, exist_ok=True)
    control = args.root / "controller"
    control.mkdir(exist_ok=False)
    save(control / "runtime.json", runtime_identity())
    recipe = os.environ.get("SGLANG_DIAG_SOURCE_RECIPE_JSON")
    if recipe:
        save(control / "source-recipe.json", json.loads(recipe))
    save(
        control / "workload.json",
        {
            "command": command,
            "start_utc": utc(),
            "delay_seconds": args.delay,
            "seconds": args.seconds,
            "profile_requested": args.profile,
            "rollouts": os.environ.get("SGLANG_DIAG_ROLLOUTS"),
        },
    )
    workload = launch(command)
    collector = None
    complete = False
    try:
        try:
            deadline = time.monotonic() + args.delay
            while workload.poll() is None and time.monotonic() < deadline:
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
            if workload.poll() is None:
                with (control / "diagnostics.log").open("w") as log:
                    common = ["--endpoints-json", str(args.endpoints)]
                    collector = subprocess.Popen(
                        [
                            sys.executable,
                            str(HERE / "collect.py"),
                            "metrics",
                            *common,
                            "--out",
                            str(args.root / "metrics"),
                            "--seconds",
                            str(args.seconds),
                            "--interval",
                            "5",
                            "--window-marker",
                            str(args.root / "metrics-window.json"),
                        ],
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                    complete = wait_collector(collector, workload)
                    if complete and args.profile and workload.poll() is None:
                        collector = subprocess.Popen(
                            [
                                sys.executable,
                                str(HERE / "collect.py"),
                                "profile",
                                *common,
                                "--out",
                                str(args.root / "profile-control"),
                                "--server-profile-dir",
                                str(args.root / "profiles"),
                                "--steps",
                                str(args.steps),
                            ],
                            stdout=log,
                            stderr=log,
                            start_new_session=True,
                        )
                        wait_collector(collector, workload)
        except Exception as error:
            save(control / "ERROR.json", {"utc": utc(), "error": str(error)})
            print(f"WARNING: diagnostic collection failed: {error}; Gym continues", file=sys.stderr)
        status = workload.wait()
        return status if status >= 0 else 128 - status
    finally:
        stop(collector)
        stop(workload)
        save(
            args.root / "done.json",
            {"utc": utc(), "workload_returncode": workload.returncode, "metric_window_complete": complete},
        )
        # Analyze after Gym finishes so auto-flushed traces have time to appear.
        # Missing captures/ranks remain explicit; no manual profiler stop is sent.
        try:
            finalize(args.root, args.root.parent)
        except Exception as error:
            save(control / "finalize-error.json", {"utc": utc(), "error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--root", type=Path, required=True)
    controller = sub.add_parser("benchmark")
    controller.add_argument("--root", type=Path, required=True)
    controller.add_argument("--endpoints", type=Path, required=True)
    controller.add_argument("--delay", type=float, default=300)
    controller.add_argument("--seconds", type=float, default=300)
    controller.add_argument("--steps", type=int, default=20)
    controller.add_argument(
        "--profile", action="store_true", help="Do not enable alongside an active nsys/CUPTI capture"
    )
    controller.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "benchmark" and (
        not math.isfinite(args.delay)
        or not math.isfinite(args.seconds)
        or args.delay < 0
        or args.seconds <= 0
        or args.steps <= 0
    ):
        parser.error("delay must be nonnegative; seconds and steps must be positive")

    def shutdown(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    raise SystemExit(worker(args) if args.mode == "worker" else benchmark(args))


if __name__ == "__main__":
    main()
