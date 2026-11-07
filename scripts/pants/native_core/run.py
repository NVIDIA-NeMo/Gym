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

from batching import validate_plan
from collect import targets
from incremental import ENVIRONMENT as INCREMENTAL_ENVIRONMENT
from incremental import select_policy
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


def execute_batch(
    *,
    checkout: Path,
    output: Path,
    identifier: str,
    selected: list[str],
    cacheable: bool,
    args: argparse.Namespace,
    workers: int | None,
    environment: dict[str, str],
) -> dict:
    """Run one verified native batch, retaining authentic Pants result provenance."""
    directory = checkout / "results/incremental" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    prefix = [
        str(Path(args.pants).resolve()),
        "--no-pantsd",
        "--pants-config-files=pants.native.toml",
        "--process-execution-local-parallelism=1",
    ]
    batch_environment = dict(environment)
    store = environment.get("PANTS_LOCAL_STORE_DIR")
    if store:
        batch_environment["PANTS_LOCAL_STORE_DIR"] = str(Path(store) / ("reusable" if cacheable else "forced"))
    options = []
    if cacheable:
        # PEX builds its PATH from this subsystem option after pytest extra env.
        # Pin the actual search path rather than inherit the CI launcher's tools.
        options.append("--pex-executable-search-paths=['/usr/bin', '/bin']")
        options.append(
            "--test-extra-env-vars=" + json.dumps([f"{key}={value}" for key, value in INCREMENTAL_ENVIRONMENT.items()])
        )
    with (output / f"{identifier}-batch-plan.log").open("w") as log:
        planning = subprocess.run(
            prefix + options + ["native-batch-plan", *selected],
            cwd=checkout,
            env=batch_environment,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            check=True,
        )
    plan = json.loads(planning.stdout)
    size = validate_plan(plan, selected)
    (output / f"{identifier}-batch-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    pytest_args = [
        "-m",
        "not sandbox" if args.group == "core" else "sandbox",
        # Reset Pants-inferred source roots and match both sampled CI commands.
        "--cov-reset",
        "--cov=.",
        "--cov-fail-under=0",
        "-p",
        "scripts.pants.native_core.incremental_runtime" if cacheable else "scripts.pants.native_core.runtime",
    ]
    if args.group == "core":
        pytest_args.append("--junit-prefix=nemo_gym.core")
    if workers is not None:
        pytest_args.extend(["-n", str(workers), "--dist=loadfile"])
    command = (
        prefix
        + options
        + [
            "test",
            *selected,
            "--stats-log",
            "--stats-format=jsonlines",
            "--stats-output-file=" + str(output / f"{identifier}-pants-stats.jsonl"),
            "--test-use-coverage",
            "--test-report",
            "--test-experimental-report-test-result-info",
            f"--test-batch-size={size}",
            "--test-report-dir=" + str(directory.relative_to(checkout) / "junit"),
            "--coverage-py-output-dir=" + str(directory.relative_to(checkout) / "coverage"),
            "--pytest-junit-family=" + ("xunit1" if args.group == "core" else "xunit2"),
        ]
    )
    if not cacheable:
        command.append("--test-force")
    command += ["--", *pytest_args]
    previous = set(checkout.glob("test_result_info_report_runid*.json"))
    (output / f"{identifier}-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / f"{identifier}-pants.log").open("w") as log:
        result = subprocess.run(command, cwd=checkout, env=batch_environment, stdout=log, stderr=subprocess.STDOUT)
    reports = set(checkout.glob("test_result_info_report_runid*.json")) - previous
    if len(reports) != 1:
        raise ValueError("Expected one fresh native result-info report per invocation")
    shutil.copy2(reports.pop(), directory / "result-info.json")
    if result.returncode:
        # Keep failed evidence; failures must never be presented as successful reused output.
        shutil.copytree(directory, output / "failed-batches" / identifier)
        raise subprocess.CalledProcessError(result.returncode, command)
    return {
        "id": identifier,
        "mode": "cacheable" if cacheable else "forced",
        "directory": directory,
        "targets": selected,
    }


def run(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite a previous native run: {output}")
    source = args.source.resolve()
    assets = Path(__file__).resolve().parent
    mode = getattr(args, "execution_mode", "incremental")
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
        "pants_result_cache": "incremental-native-tests" if mode == "incremental" else "forced-native-tests",
        "execution_mode": mode,
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
        from lock_assets import lock_assets

        dependencies = lock_assets(source, assets)
        verify_inputs(source, dependencies)
        sampled_shards = 4 if (source / "scripts/ci/core_test_shards.py").is_file() else 1
        if args.group == "core" and (args.shards != sampled_shards or not 0 <= args.shard < sampled_shards):
            raise ValueError("Core native execution must preserve the sampled baseline shard count")
        metadata["dependency_bundle"] = str(dependencies.relative_to(assets))
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
                    (dependencies / f"{resolve}.requirements.txt").read_text()
                ):
                    raise ValueError(f"Fresh uv export differs from native {resolve} requirements")
                shutil.copy2(dependencies / f"{resolve}.lock", native / f"{resolve}.lock")
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
            if manifest["source_sha"] != metadata["source_sha"]:
                raise ValueError("Source changed while preparing native execution")
            duration_file = checkout / "scripts/ci/core_test_durations.json"
            expected_durations = (
                hashlib.sha256(duration_file.read_bytes()).hexdigest() if sampled_shards == 4 else None
            )
            if manifest["durations_sha256"] != expected_durations:
                raise ValueError("Shard duration weights changed after collection")
            selected = targets(manifest, resolve=args.group, shard=args.shard if args.group == "core" else None)
            if not selected:
                raise ValueError("Native collection selected no test files")
            (output / "targets.json").write_text(json.dumps(selected, indent=2) + "\n")
            incremental = False
            policy = None
            candidate = None
            decision = {"eligible": False, "forced_reasons": ["forced execution requested"]}
            if mode == "incremental" and args.group == "core":
                policy, decision = select_policy(checkout, assets)
                candidate = decision.get("candidate")
                if sys.platform != "linux":
                    decision = {"eligible": False, "forced_reasons": ["runtime verification supports Linux only"]}
                if decision["eligible"] and any(candidate in target for target in selected):
                    captured_at = time.monotonic()
                    with (output / "runtime-fingerprint.log").open("w") as log:
                        capture = subprocess.run(
                            [
                                str(checkout / ".venv/bin/python"),
                                str(assets / "runtime_fingerprint.py"),
                                str(native / "runtime-inputs.json"),
                            ],
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    metadata["runtime_fingerprint_seconds"] = round(time.monotonic() - captured_at, 3)
                    if capture.returncode:
                        decision = {
                            "eligible": False,
                            "forced_reasons": [
                                "runtime fingerprint could not be verified; see runtime-fingerprint.log"
                            ],
                        }
                    else:
                        shutil.copy2(policy, native / "incremental-policy.json")
                        shutil.copy2(native / "runtime-inputs.json", output / "runtime-inputs.json")
                        incremental = True
            metadata["incremental_eligibility"] = decision
            write_build_configuration(checkout, incremental=incremental)
            interpreter = subprocess.check_output(
                [args.uv, "python", "find", (checkout / ".python-version").read_text().strip()],
                cwd=checkout,
                text=True,
            ).strip()
            environment["PATH"] = str(Path(interpreter).parent) + os.pathsep + environment.get("PATH", "")
            reusable = [target for target in selected if incremental and candidate in target]
            forced = [target for target in selected if target not in reusable]
            batches = []
            for identifier, addresses, cacheable in (("migration", reusable, True), ("forced", forced, False)):
                if not addresses:
                    continue
                batch = execute_batch(
                    checkout=checkout,
                    output=output,
                    identifier=identifier,
                    selected=addresses,
                    cacheable=cacheable,
                    args=args,
                    workers=workers,
                    environment=environment,
                )
                batches.append(batch)
            sys.path.insert(0, str(assets.parent))
            from full_ci_aggregate import merge_execution_outputs

            accounting = merge_execution_outputs(
                checkout=checkout,
                batches=batches,
                collection_json=manifest_path,
                output=output,
                group=args.group,
                shard=args.shard,
                env=environment,
            )
            metadata["execution_accounting"] = {"path": "test-accounting.json", "counts": accounting["counts"]}
            metadata["command_exit_code"] = 0
            metadata["batching"].update(actual_pytest_batches=len(batches), partition="verified-explicit-groups")
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
    parser.add_argument("--execution-mode", choices=["incremental", "forced"], default="incremental")
    args = parser.parse_args()
    if not Path(args.pants).is_file():
        args.pants = shutil.which(args.pants) or args.pants
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
