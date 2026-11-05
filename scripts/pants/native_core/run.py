# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the complete core shard or sandbox pass through native Pants Python targets."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from batching import validate_plan, verify_batch_result
from collect import targets
from prepare import BATCH_SIZE, requirement_pins, snapshot_distribution, validate_lock, write_build_configuration


def core_worker_count(source: Path) -> int | None:
    """Read sampled CI's explicit worker count, retaining historical serial runs."""
    script = (source / "scripts/ci/core_unit_tests.sh").read_text()
    assignment = re.search(r"^pytest_addopts=(.+)$", script, re.MULTILINE)
    if assignment is None:
        raise ValueError("Cannot read sampled core pytest options; inspect its worker policy")
    values = shlex.split(assignment[1])
    if len(values) != 1:
        raise ValueError("Unsupported sampled core pytest option assignment")
    options = shlex.split(values[0])
    if "-n" not in options:
        if any(option.startswith(("-n", "--numprocesses")) for option in options):
            raise ValueError("Unsupported sampled core worker option")
        return None
    index = options.index("-n") + 1
    if index == len(options):
        raise ValueError("Sampled core worker count is missing")
    count = options[index]
    if not count.isdecimal() or int(count) < 1 or "--dist=loadfile" not in options:
        raise ValueError("Sampled core workers require an explicit positive count and loadfile distribution")
    return int(count)


def core_worker_environment(source: Path, environment: dict[str, str]) -> dict[str, str]:
    """Apply sampled CI's thread limits before collection, setup and execution."""
    script = (source / "scripts/ci/core_unit_tests.sh").read_text()
    if 'source "${ci_dir}/test_worker_env.sh"' not in script:
        return dict(environment)
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    command = [
        "bash",
        "-e",
        "-u",
        "-o",
        "pipefail",
        "-c",
        'source "$1"; shift; exec "$@"',
        "pants-core-workers",
        str(source / "scripts/ci/test_worker_env.sh"),
        sys.executable,
        "-c",
        "import json, os, sys; print(json.dumps({name: os.environ[name] for name in sys.argv[1:]}))",
        *names,
    ]
    values = json.loads(subprocess.check_output(command, env=environment, text=True))
    return dict(environment, **values)


def verify_inputs(repo: Path, assets: Path) -> None:
    recorded = json.loads((assets / "lock-inputs.json").read_text())
    for name, expected in recorded["inputs"].items():
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Native dependency locks are stale for {name}; regenerate and validate both resolves")


def combine_junit(reports: Path, output: Path, *, collection: Path | None = None) -> None:
    """Preserve every testcase occurrence while matching the baseline artifact name."""
    combined = ElementTree.Element("testsuites")
    paths = sorted(reports.rglob("*.xml"))
    if not paths:
        raise ValueError("Native test execution produced no JUnit reports")
    for path in paths:
        root = ElementTree.parse(path).getroot()
        if root.tag == "testsuite":
            combined.append(root)
        elif root.tag == "testsuites":
            combined.extend(root)
        else:
            raise ValueError(f"Unexpected JUnit root in {path}: {root.tag}")
    if collection is not None:
        collected = ElementTree.parse(collection).getroot()
        collection_suite = ElementTree.Element("testsuite", name="collection", errors="0", failures="0")
        seen = Counter(
            tuple(case.get(key, "") for key in ("file", "classname", "name"))
            for case in combined.iter("testcase")
            if case.find("skipped") is not None
        )
        for case in collected.iter("testcase"):
            if case.find("skipped") is None:
                raise ValueError("Collect-only JUnit unexpectedly contains an executed or failed testcase")
            identity = tuple(case.get(key, "") for key in ("file", "classname", "name"))
            if seen[identity]:
                seen[identity] -= 1
            else:
                collection_suite.append(case)
        if len(collection_suite):
            collection_suite.set("tests", str(len(collection_suite)))
            collection_suite.set("skipped", str(len(collection_suite)))
            combined.append(collection_suite)
    output.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(combined).write(output, encoding="utf-8", xml_declaration=True)


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite a previous native run: {output}")
    source = args.source.resolve()
    assets = Path(__file__).resolve().parent
    workers = core_worker_count(source) if args.group == "core" else None
    environment = dict(os.environ)
    if args.group == "core":
        environment = core_worker_environment(source, environment)
    metadata = {
        "engine": "pants-native",
        "job": args.group,
        "shard": args.shard,
        "shards": args.shards,
        "started_at": started,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "exit_code": 1,
        "command_exit_code": None,
        "prototype": True,
        "pants_result_cache": "forced-native-tests",
        "batching": {
            "compatibility": "per-resolve",
            "target_files": BATCH_SIZE,
            "partition": "stable-hash",
            "local_parallelism": 1,
            "pytest_workers": workers or 1,
            "pytest_distribution": "loadfile" if workers is not None else "serial",
            "actual_pytest_batches": None,
        },
    }
    try:
        verify_inputs(source, assets)
        if args.group == "core" and (args.shards != 4 or not 0 <= args.shard < 4):
            raise ValueError("Core native execution preserves exactly four baseline shards")
        with tempfile.TemporaryDirectory(prefix="gym-pants-native-", dir="/var/tmp") as temporary:
            checkout = Path(temporary) / "checkout"
            with (output / "prepare.log").open("w") as log:
                subprocess.run(
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
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env=environment,
                )
            native = checkout / "scripts/pants/native_core"
            for resolve in ("core", "sandbox"):
                exported = native / f"{resolve}.requirements.txt"
                if requirement_pins(exported.read_text()) != requirement_pins(
                    (assets / f"{resolve}.requirements.txt").read_text()
                ):
                    raise ValueError(f"Fresh uv export differs from native {resolve} requirements")
                shutil.copy2(assets / f"{resolve}.lock", native / f"{resolve}.lock")
                validate_lock(exported, native / f"{resolve}.lock")
            manifest_path = output / "collection.json"
            with (output / "collection.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(assets / "collect.py"),
                        "collect",
                        "--repo",
                        str(checkout),
                        "--output",
                        str(manifest_path),
                        "--resolve",
                        args.group,
                        "--uv",
                        args.uv,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env=environment,
                )
            manifest = json.loads(manifest_path.read_text())
            snapshot_distribution(checkout)
            write_build_configuration(checkout)
            if manifest["source_sha"] != metadata["source_sha"]:
                raise ValueError("Source changed while preparing native execution")
            expected_durations = hashlib.sha256(
                (checkout / "scripts/ci/core_test_durations.json").read_bytes()
            ).hexdigest()
            if manifest["durations_sha256"] != expected_durations:
                raise ValueError("Shard duration weights changed after collection")
            selected = targets(manifest, resolve=args.group, shard=args.shard if args.group == "core" else None)
            if not selected:
                raise ValueError("Native collection selected no test files")
            (output / "targets.json").write_text(json.dumps(selected, indent=2) + "\n")
            marker = "not sandbox" if args.group == "core" else "sandbox"
            pytest_args = ["-m", marker, "--cov-fail-under=0", "-p", "scripts.pants.native_core.runtime"]
            if args.group == "core":
                pytest_args.append("--junit-prefix=nemo_gym.core")
            if workers is not None:
                pytest_args.extend(["-n", str(workers), "--dist=loadfile"])
            command = [
                str(Path(args.pants).resolve()),
                "--no-pantsd",
                "--pants-config-files=pants.native.toml",
                "--process-execution-local-parallelism=1",
                "test",
                *selected,
                "--stats-log",
                "--stats-format=jsonlines",
                "--stats-output-file=" + str(output / "pants-stats.jsonl"),
                "--test-use-coverage",
                "--test-force",
                "--test-report",
                "--test-report-dir=results/native-junit",
                "--coverage-py-output-dir=results/native-coverage",
                "--pytest-junit-family=" + ("xunit1" if args.group == "core" else "xunit2"),
                "--",
                *pytest_args,
            ]
            interpreter = subprocess.check_output(
                [args.uv, "python", "find", (checkout / ".python-version").read_text().strip()], text=True
            ).strip()
            environment["PATH"] = str(Path(interpreter).parent) + os.pathsep + environment.get("PATH", "")
            if args.group == "core":
                with (output / "batch-plan.log").open("w") as log:
                    planning = subprocess.run(
                        command[:4] + ["native-batch-plan", *selected],
                        cwd=checkout,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=log,
                        text=True,
                        check=True,
                    )
                plan = json.loads(planning.stdout)
                size = validate_plan(plan, selected)
                (output / "batch-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
                command[command.index("--") : command.index("--")] = [
                    f"--test-batch-size={size}",
                    "--test-experimental-report-test-result-info",
                ]
                metadata["batching"].update(
                    target_files=size,
                    partition="verified-single-forced-batch",
                    expected_pytest_batches=1,
                )
            (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            with (output / "pants.log").open("w") as log:
                result = subprocess.run(command, cwd=checkout, env=environment, stdout=log, stderr=subprocess.STDOUT)
            metadata["command_exit_code"] = result.returncode
            reports = checkout / "results/native-junit"
            if reports.exists():
                combine_junit(
                    reports, output / "junit" / f"{args.group}.xml", collection=manifest_path.with_suffix(".xml")
                )
            raw = checkout / "results/native-coverage/.coverage"
            if raw.exists():
                coverage_name = f".coverage.core-{args.shard}" if args.group == "core" else ".coverage.sandbox"
                collection_coverage = manifest_path.with_suffix(".coverage")
                if not collection_coverage.is_file():
                    raise ValueError("Global collection did not return raw coverage")
                coverage_environment = dict(environment, COVERAGE_FILE=str(output / coverage_name))
                subprocess.run(
                    [
                        str(checkout / ".venv/bin/python"),
                        "-m",
                        "coverage",
                        "combine",
                        "--keep",
                        str(collection_coverage),
                        str(raw),
                    ],
                    cwd=checkout,
                    env=coverage_environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command)
            if args.group == "core":
                metadata["batching"].update(verify_batch_result(checkout, reports, output, selected))
            if not raw.is_file() or not list((output / "junit").rglob("*.xml")):
                raise ValueError("Native execution did not return both raw coverage and JUnit reports")
            metadata["exit_code"] = 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        metadata["error"] = str(error)
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["elapsed_seconds"] = round(time.monotonic() - start, 3)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return int(metadata["exit_code"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--group", choices=["core", "sandbox"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pants", default=os.environ.get("PANTS_BIN", "pants"))
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    if not Path(args.pants).is_file():
        args.pants = shutil.which(args.pants) or args.pants
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
