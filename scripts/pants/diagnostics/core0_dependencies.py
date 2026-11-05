# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate core0 dependency diagnostic; never evidence of test parity or CI speed."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pants", required=True)
    parser.add_argument("--uv", required=True)
    args = parser.parse_args()
    controller, source, output = (
        args.controller.resolve(),
        args.source.resolve(),
        args.output.resolve(),
    )
    assets = controller / "scripts/pants/native_core"
    backend = controller / "scripts/pants/diagnostics"
    sys.path[:0] = [str(assets), str(assets.parent)]
    import full_ci
    import run as native_run
    from batching import validate_plan
    from collect import targets
    from prepare import (
        requirement_pins,
        snapshot_distribution,
        validate_lock,
        write_build_configuration,
    )

    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise FileExistsError("Refusing to overwrite a diagnostic run")
    os.chdir(source)
    metadata = {
        "kind": "core0-dependency-diagnostic",
        "timing_comparison_valid": False,
        "tests_executed": False,
        "source_sha": args.source_sha,
        "shard": 0,
        "shards": 4,
        "trace_profile": True,
        "cache_save_policy": "never",
        "exit_code": 1,
        "limit": "Dependency traversal excludes test/PEX execution and may not reproduce the full engine interval.",
    }
    started = time.monotonic()
    try:
        actual = full_ci.git("rev-parse", "HEAD")
        if len(args.source_sha) != 40 or actual != args.source_sha:
            raise ValueError("Sample checkout must equal the requested immutable source SHA")
        if full_ci.git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Sample has tracked modifications")
        initial = full_ci.source_snapshot()
        metadata["source_digest"] = initial[1]
        metadata["controller_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=controller, text=True
        ).strip()
        native_run.verify_inputs(source, assets)
        metadata["sampled_pytest_workers"] = native_run.core_worker_count(source)
        env = {k: v for k, v in os.environ.items() if k in full_ci.ENVIRONMENT or k.startswith("JAVA_HOME_")}
        env.update(GITHUB_SHA=args.source_sha, GYM_PANTS_INPUT_DIGEST=initial[1])
        env = native_run.core_worker_environment(source, full_ci.pants_environment(env))

        def invoke(command: list[str], name: str, cwd: Path, *, timeout: int = 1200) -> None:
            (output / f"{name}-command.json").write_text(json.dumps(command, indent=2) + "\n")
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout,
                )

        with tempfile.TemporaryDirectory(prefix="gym-core0-diagnostic-", dir="/var/tmp") as temporary:
            checkout = Path(temporary) / "checkout"
            invoke(
                [
                    sys.executable,
                    str(assets / "prepare.py"),
                    "prepare",
                    "--source",
                    str(source),
                    "--destination",
                    str(checkout),
                    "--uv",
                    args.uv,
                ],
                "prepare",
                source,
            )
            native = checkout / "scripts/pants/native_core"
            for resolve in ("core", "sandbox"):
                exported = native / f"{resolve}.requirements.txt"
                if requirement_pins(exported.read_text()) != requirement_pins((assets / exported.name).read_text()):
                    raise ValueError(f"Fresh uv requirements differ for {resolve}")
                shutil.copy2(assets / f"{resolve}.lock", native / f"{resolve}.lock")
                validate_lock(exported, native / f"{resolve}.lock")
            manifest_path = output / "collection.json"
            invoke(
                [
                    sys.executable,
                    str(assets / "collect.py"),
                    "collect",
                    "--repo",
                    str(checkout),
                    "--output",
                    str(manifest_path),
                    "--resolve",
                    "core",
                    "--uv",
                    args.uv,
                ],
                "collection",
                checkout,
            )
            manifest = json.loads(manifest_path.read_text())
            snapshot_distribution(checkout)
            write_build_configuration(checkout)
            if manifest["source_sha"] != args.source_sha:
                raise ValueError("Collected source SHA differs")
            if (
                manifest["durations_sha256"]
                != hashlib.sha256((checkout / "scripts/ci/core_test_durations.json").read_bytes()).hexdigest()
            ):
                raise ValueError("Shard duration weights changed")
            selected = targets(manifest, resolve="core", shard=0)
            if not selected:
                raise ValueError("No selected core0 files")
            (output / "targets.json").write_text(json.dumps(selected, indent=2) + "\n")
            interpreter = subprocess.check_output(
                [
                    args.uv,
                    "python",
                    "find",
                    (checkout / ".python-version").read_text().strip(),
                ],
                text=True,
                env=env,
            ).strip()
            env["PATH"] = str(Path(interpreter).parent) + os.pathsep + env.get("PATH", "")
            base = [
                str(Path(args.pants).resolve()),
                "--no-pantsd",
                "--pants-config-files=pants.native.toml",
                "--process-execution-local-parallelism=1",
            ]
            with (output / "batch-plan.log").open("w") as log:
                planning = subprocess.run(
                    base + ["native-batch-plan", *selected],
                    cwd=checkout,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    check=True,
                    timeout=300,
                )
            plan = json.loads(planning.stdout)
            metadata["verified_batch_size"] = validate_plan(plan, selected)
            (output / "batch-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
            command = base + [
                '--backend-packages=+["bounded_profile"]',
                "--pythonpath=+" + json.dumps([str(backend)]),
                "--streaming-workunits-level=trace",
                "--bounded-profile-output=" + str(output / "workunits.json"),
                "dependencies",
                "--transitive",
                *selected,
            ]
            metadata["selected_target_count"] = len(selected)
            phase_start = time.monotonic()
            invoke(command, "dependencies", checkout, timeout=900)
            metadata["profiled_dependency_seconds"] = time.monotonic() - phase_start
            profile = json.loads((output / "workunits.json").read_text())
            if not profile["events"] or len(profile["rows"]) > 128:
                raise ValueError("Missing or unbounded diagnostic profile")
        if initial != full_ci.source_snapshot() or full_ci.git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Tracked source changed during diagnostic")
        metadata["exit_code"] = 0
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["total_diagnostic_seconds"] = time.monotonic() - started
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
