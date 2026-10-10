# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advance each benchmark from its real canary to full coverage independently."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.nooa_baselines.serving import validated_model


def run_logged(command: list[str], *, path: Path, env: dict[str, str], cwd: Path) -> None:
    """Keep each component's output without overwriting a previous attempt."""
    with path.open("x") as log:
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def wait_for_model(model_dir: Path, *, timeout_seconds: int) -> None:
    """Each benchmark advances after a preflight bound to its exact model manifest."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            validated_model(model_dir)
            return
        except FileNotFoundError:
            time.sleep(min(15, max(0, deadline - time.monotonic())))
    raise TimeoutError("Policy did not become ready within the controller preparation window")


def run_benchmark(
    benchmark: str,
    *,
    root: Path,
    gym: Path,
    attempt: Path,
    env: dict[str, str],
    model_dir: Path,
    model_wait_seconds: int,
    config: list[Path],
    concurrency: int | None = None,
    prepare_apptainer: bool = False,
) -> None:
    """A failed benchmark canary cannot dispatch its full phase or block its peers."""
    wait_for_model(model_dir, timeout_seconds=model_wait_seconds)
    for phase in ("canary", "full"):
        existing = root / benchmark / "canary/completion.json"
        if phase == "canary" and existing.is_file():
            if not json.loads(existing.read_text()).get("pipeline_passed"):
                raise RuntimeError("Existing canary failed; preserve its evidence and inspect before retrying")
            # The full launcher still checks its exact input/model/source/overlay hashes.
            continue
        command = [
            sys.executable,
            str(gym / "benchmarks/nooa_baselines/run_eval.py"),
            "--run-root",
            str(root),
            "--gym-source",
            str(gym),
            "--benchmark",
            benchmark,
            "--model-dir",
            str(model_dir),
            "--phase",
            phase,
        ]
        if prepare_apptainer and benchmark == "gdp":
            command.append("--prepare-apptainer")
        for overlay in config:
            command.extend(["--config", str(overlay)])
        if phase == "full" and concurrency is not None:
            command.extend(["--concurrency", str(concurrency)])
        run_logged(command, path=attempt / f"{benchmark}-{phase}.log", env=env, cwd=gym)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gym-source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--benchmarks", nargs="+", choices=("swe", "tb", "gdp"), default=["swe", "tb", "gdp"])
    parser.add_argument("--model-dir", type=Path, help="Shared preflighted endpoint directory; default RUN_ROOT/model")
    parser.add_argument("--prepare-apptainer", action="store_true", help="Prepare the GDP private Apptainer config")
    parser.add_argument("--model-wait-seconds", type=int, default=7200)
    for benchmark in ("swe", "tb", "gdp"):
        parser.add_argument(f"--{benchmark}-model-dir", type=Path, help="Optional separate policy replica")
        parser.add_argument(f"--{benchmark}-config", type=Path, action="append", default=[])
        parser.add_argument(f"--{benchmark}-concurrency", type=int)
    args = parser.parse_args()
    if args.model_wait_seconds <= 0 or len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error("Use a positive model wait and unique benchmarks")
    if any(
        getattr(args, name + "_concurrency") is not None and getattr(args, name + "_concurrency") < 1
        for name in args.benchmarks
    ):
        parser.error("Episode concurrency must be positive")
    root = args.run_root.resolve(strict=True)
    gym = args.gym_source.resolve(strict=True)
    attempt = root / ("controller-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    attempt.mkdir()
    env = os.environ.copy()
    for key in ("PYTHONHOME", "UV_CONSTRAINT", "PIP_CONSTRAINT", "UV_VENV_CLEAR"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", RAY_TMPDIR=env.get("RAY_TMPDIR", "/tmp"))
    (attempt / "launch.json").write_text(
        json.dumps(
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "benchmarks": args.benchmarks,
                "gym_source": str(gym),
            },
            indent=2,
        )
        + "\n"
    )
    outcome = {}
    with ThreadPoolExecutor(max_workers=len(args.benchmarks)) as pool:
        futures = {
            pool.submit(
                run_benchmark,
                benchmark,
                root=root,
                gym=gym,
                attempt=attempt,
                env=env,
                model_dir=(getattr(args, benchmark + "_model_dir") or args.model_dir or root / "model").resolve(),
                model_wait_seconds=args.model_wait_seconds,
                config=getattr(args, benchmark + "_config"),
                concurrency=getattr(args, benchmark + "_concurrency"),
                prepare_apptainer=args.prepare_apptainer,
            ): benchmark
            for benchmark in args.benchmarks
        }
        for future in as_completed(futures):
            benchmark = futures[future]
            try:
                future.result()
                outcome[benchmark] = {"completed": True}
            except Exception as error:
                outcome[benchmark] = {"completed": False, "error_type": type(error).__name__, "message": str(error)}
            receipt = {"updated_utc": datetime.now(timezone.utc).isoformat(), "benchmarks": outcome}
            (attempt / "progress.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({benchmark: outcome[benchmark]}), flush=True)
    raise SystemExit(0 if all(value["completed"] for value in outcome.values()) else 1)


if __name__ == "__main__":
    main()
