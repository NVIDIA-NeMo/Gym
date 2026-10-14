# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the reference TE table from tested source and fresh full-suite probes.

From a checkout matching a full commit SHA, with Gym/dev and harness dependencies:

    python -m scripts.harness_conformance.table --commit <40-character-sha> \
        --output /absolute/path/to/new-evidence-directory

Runner, selected adapter, and checker tests must pass before any probes run.
The source tree must still match the commit after testing and probing. Evidence
failures are reportable; execution/checker errors prevent publication. The local
JSON is diagnostic output; CI renders the same table without persisting a report.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import nemo_gym
from nemo_gym.harness_capabilities import checker
from nemo_gym.harness_capabilities.cli import digest_file

from . import episode, runner
from .documentation import BEGIN, END, PAGE, atomic_write, replace_block
from .episode import HARNESSES
from .rendering import render_table
from .runner import run_suite
from .scenarios import SCENARIOS, SUITE


ROOT = Path(__file__).resolve().parents[2]


def verify_revision(root: Path, commit: str) -> None:
    """Require an existing full SHA and matching source, for Git or jj checkouts."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be a full lowercase 40-character commit SHA")

    def read(*command: str) -> str:
        return subprocess.check_output(command, cwd=root, text=True, stderr=subprocess.PIPE).strip()

    if (root / ".jj").exists():
        resolved = read("jj", "--no-pager", "log", "--no-graph", "-r", commit, "-T", "commit_id")
        changes = read("jj", "--no-pager", "diff", "--from", commit, "--to", "@", "--summary")
    else:
        resolved = read("git", "rev-parse", "--verify", f"{commit}^{{commit}}")
        changes = read("git", "diff", commit, "--name-only")
        changes += read("git", "ls-files", "--others", "--exclude-standard")
    if resolved != commit or changes:
        raise ValueError("source tree differs from --commit; save all source changes before regenerating")


def run_tests(root: Path, output: Path, *, name: str, path: str) -> dict:
    """Run a complete test directory; errors and empty/all-skipped suites fail closed."""
    report = output / f"{name}.xml"
    log = output / f"{name}.log"
    arguments = ["-m", "pytest", path, "--import-mode=importlib", "-o", "addopts=", "-q", "--junitxml", str(report)]
    env = {key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"}
    env["PYTHONPATH"] = str(root)
    env["NEMO_GYM_EXTRA_ROOTS"] = str(root)
    print(f"Testing {name}: {path}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(
            [sys.executable, *arguments], cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise ValueError(f"{name} tests failed (exit {result.returncode}); see {log}")
    cases = ET.parse(report).findall(".//testcase")
    skipped = sum(case.find("skipped") is not None for case in cases)
    if (
        not cases
        or skipped == len(cases)
        or any(case.find("failure") is not None or case.find("error") is not None for case in cases)
    ):
        raise ValueError(f"{name} tests did not establish a passing suite; see {report}")
    return {
        "path": path,
        "passed": len(cases) - skipped,
        "skipped": skipped,
        "log_sha256": digest_file(log),
        "junit_sha256": digest_file(report),
    }


def regenerate(*, root: Path, commit: str, output: Path, harnesses: list[str], timeout: float) -> None:
    """Gate fresh full-suite results and atomically replace only the generated page block."""
    if not harnesses or len(set(harnesses)) != len(harnesses) or set(harnesses) - HARNESSES.keys():
        raise ValueError("select at least one supported harness, without duplicates")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    verify_revision(root, commit)
    page = root / PAGE
    original = page.read_text()
    replace_block(original, BEGIN + "\n" + END)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    test_paths = {"runner": "scripts/harness_conformance/tests"}
    test_paths.update({harness: f"responses_api_agents/{harness}_agent/tests" for harness in harnesses})
    test_paths["checker"] = "tests/unit_tests/harness_capabilities"
    tests = {name: run_tests(root, output, name=name, path=path) for name, path in test_paths.items()}
    verify_revision(root, commit)
    probes = output / "probes"
    summary, code = run_suite(harnesses=harnesses, scenarios=SCENARIOS, output=probes, timeout=timeout)
    if code not in (0, 1) or summary["runner_status"] != "completed" or not summary["full_suite"]:
        raise ValueError(f"probe execution/checker error; reference table unchanged; see {probes}")
    runtimes = {}
    for harness in harnesses:
        identities = []
        for scenario in SCENARIOS:
            runtime = json.loads((probes / harness / scenario.name / "runtime.json").read_text())
            # Runtime locations are machine-local; versions and source hashes are portable.
            identities.append({key: value for key, value in runtime.items() if key not in ("source", "executable")})
        if not identities[0] or any(identity != identities[0] for identity in identities):
            raise ValueError(f"{harness} runtime changed during probing")
        runtimes[harness] = identities[0]
    report = {
        "schema_version": "trajectory-capabilities-table/v1",
        "gym_commit": commit,
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": SUITE,
        "tests": tests,
        "runtimes": runtimes,
        "python": sys.version,
        "harnesses": summary["harnesses"],
        "limits": summary["limits"],
        "suite_sha256": summary["suite_sha256"],
        "summary_sha256": digest_file(probes / "conformance_summary.json"),
    }
    serialized = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    block = render_table(report)
    updated = replace_block(original, block)
    verify_revision(root, commit)
    if page.read_text() != original:
        raise ValueError("reference page changed during regeneration")
    # Retain the local diagnostic report separately from the preview table.
    atomic_write(output / "generation_report.json", serialized)
    atomic_write(page, updated)
    print(f"Updated {page} for {commit}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="full source commit SHA; the checkout must match it")
    parser.add_argument(
        "--output", required=True, type=Path, help="new directory for test logs and full probe artifacts"
    )
    parser.add_argument(
        "--harness", action="append", choices=HARNESSES, help="defaults to all; others show Not measured"
    )
    parser.add_argument("--timeout", type=float, default=120, help="timeout in seconds per probe")
    args = parser.parse_args(argv)
    try:
        for module in (nemo_gym, checker, episode, runner):
            if not Path(module.__file__).resolve().is_relative_to(ROOT):
                raise ValueError(
                    f"{module.__name__} was imported from another checkout; use a clean Python environment"
                )
        regenerate(
            root=ROOT,
            commit=args.commit,
            output=args.output,
            harnesses=args.harness or list(HARNESSES),
            timeout=args.timeout,
        )
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, ET.ParseError) as exc:
        print(f"Regeneration failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
