# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch, witness, and inspect isolated episodes for the local P0 probe suite."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from nemo_gym import harness_capabilities
from nemo_gym.harness_capabilities.behavior import inspect_behavior
from nemo_gym.harness_capabilities.checker import NAMES, EvidenceScope, inspect_record
from nemo_gym.harness_capabilities.cli import digest_file, inspect_bundle, json_rows
from nemo_gym.harness_capabilities.health import inspect_health
from nemo_gym.harness_capabilities.results import gate_passes, render_matrices

from .episode import HARNESSES
from .scenarios import SCENARIOS, SUITE, Scenario, suite_manifest


ROOT = Path(__file__).resolve().parents[2]


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _fingerprint(request: dict, status: int, response: dict) -> str:
    # Gym's Chat SSE decoder omits the provider's optional creation clock.
    response = dict(response or {})
    if response.get("object") == "chat.completion":
        response.pop("created", None)
    return hashlib.sha256(_json([request, status, response]).encode()).hexdigest()


def run_process(command: list[str], *, directory: Path, timeout: float) -> dict:
    """Keep logs and reap the worker and its descendants, including new process groups."""
    with (directory / "episode.log").open("wb") as log:
        # Probe the checkout being reported, even when another editable Gym or
        # extra component root is present in the caller's environment.
        env = {**os.environ, "PYTHONPATH": str(ROOT), "NEMO_GYM_EXTRA_ROOTS": str(ROOT)}
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        tracked = {}
        timed_out = False
        try:
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                # CLIs may start their own sessions, so killing only our process group
                # would leave their children alive after a worker timeout.
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        tracked[(child.pid, child.create_time())] = child
                except psutil.NoSuchProcess:
                    pass
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    proc.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                proc.terminate()
            for child in reversed(list(tracked.values())):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(list(tracked.values()), timeout=2)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return {"returncode": proc.returncode, "timed_out": timed_out}


def inspect_episode(scenario: Scenario, directory: Path, execution: dict) -> dict:
    """Inspect artifacts, behavior and health independently of process exit status."""
    witness_path = directory / "witness.json"
    witness = json.loads(witness_path.read_text()) if witness_path.exists() else None
    bundle = directory / "rollouts.jsonl"
    health_report = directory / "health" / "quality_summary.json"
    records = list(json_rows(bundle)) if bundle.exists() else []
    record = records[0][1] if len(records) == 1 else None
    scope = EvidenceScope(tools=scenario.tool_steps > 0, verifier=not scenario.terminal_error, steps=scenario.steps)
    artifact = inspect_record(record, scope=scope)
    checks = artifact["checks"] + inspect_behavior(
        witness,
        record,
        http_errors=scenario.http_errors,
        terminal_error=scenario.terminal_error,
        tool_steps=scenario.tool_steps,
        expected_reward=scenario.expected_reward,
        fingerprint=_fingerprint,
    )
    health_inputs = [path for path in (bundle, directory / "rollouts_failures.jsonl") if path.is_file()]
    checks += inspect_health(
        health_inputs,
        output=health_report.parent,
        expectations=scenario.health_expectations,
        steps=scenario.steps,
    )
    summary, report = None, None
    if record is not None:
        destination, summary = inspect_bundle(bundle, output=directory / "evidence", scope=scope)
        report = str(destination.relative_to(directory) / "evidence_summary.json")
    evidence = {}
    for key in scenario.evidence:
        verdict = artifact["evidence"][key]["verdict"]
        evidence[key] = {"verdict": verdict, "artifact_verdict": verdict}
    behavioral = [c for c in checks if c["kind"] == "behavioral"]
    behavior_passed = all(c["status"] in ("pass", "not_applicable") for c in behavioral)
    return {
        "scenario": scenario.name,
        "exercised": record is not None,
        "verdict": "fulfilled" if gate_passes(checks) else "not_fulfilled",
        "issues": [reason for c in checks if c["status"] == "fail" for reason in c["reasons"]],
        "checks": checks,
        "behavioral_status": "pass" if behavior_passed else "fail",
        "execution": execution,
        "delivery": "rollout"
        if record is not None
        else "failure_record"
        if (directory / "rollouts_failures.jsonl").exists() and (directory / "rollouts_failures.jsonl").stat().st_size
        else "missing",
        "model_attempts": len((witness or {}).get("attempts", [])),
        "evidence": evidence,
        "artifact_report": report,
        "health_report": str(health_report.relative_to(directory)) if health_inputs else None,
        "hashes": {
            path.name: digest_file(path)
            for path in (
                bundle,
                directory / "rollouts_failures.jsonl",
                witness_path,
                directory / "runtime.json",
                directory / "launch.json",
                health_report,
            )
            if path.exists()
        },
    }


def run_suite(
    *, harnesses: list[str], scenarios: tuple[Scenario, ...], output: Path, timeout: float
) -> tuple[dict, int]:
    """Run each requested harness/scenario in a fresh process and publish the completed report."""
    if not harnesses or not scenarios:
        raise ValueError("at least one harness and one scenario are required")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = suite_manifest(scenarios)
    manifest.update(
        harnesses=harnesses,
        timeout=timeout,
        python=sys.version,
        runner_sources={p.name: digest_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        checker_sources={
            p.name: digest_file(p) for p in sorted(Path(harness_capabilities.__file__).parent.glob("*.py"))
        },
        health_sources={
            str(p.relative_to(ROOT)): digest_file(p)
            for p in [ROOT / "nemo_gym/rollout_health.py", *sorted((ROOT / "nemo_gym/health").glob("*.py"))]
        },
    )
    (output / "suite.json").write_text(_json(manifest))
    rows = {}
    execution_error = False
    for harness in harnesses:
        results = []
        for scenario in scenarios:
            directory = output / harness / scenario.name
            directory.mkdir(parents=True)
            print(f"{harness}: {scenario.name}", flush=True)
            command = [
                sys.executable,
                "-m",
                "scripts.harness_conformance.episode",
                "--harness",
                harness,
                "--scenario",
                scenario.name,
                "--directory",
                str(directory),
                "--timeout",
                str(timeout),
            ]
            execution = run_process(command, directory=directory, timeout=timeout + 30)
            execution_error |= execution["returncode"] != 0 or execution["timed_out"]
            try:
                result = inspect_episode(scenario, directory, execution)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
                execution_error = True
                result = {
                    "scenario": scenario.name,
                    "exercised": False,
                    "verdict": "not_fulfilled",
                    "execution": execution,
                    "checker_error": type(exc).__name__,
                    "issues": ["could not inspect this episode's artifacts"],
                    "artifact_report": None,
                    "evidence": {
                        key: {"verdict": "not_fulfilled", "artifact_verdict": "not_fulfilled"}
                        for key in scenario.evidence
                    },
                }
            (directory / "scenario_result.json").write_text(_json(result))
            results.append(result)
        counts = {}
        for key in NAMES:
            required = [r for r in results if key in r["evidence"]]
            counts[key] = {
                "required": len(required),
                "observed": sum(r["exercised"] for r in required),
                "passed": sum(r["evidence"][key]["verdict"] == "fulfilled" for r in required),
            }
        rows[harness] = {
            "scenarios": results,
            "evidence": counts,
            "verdict": "fulfilled" if all(r["verdict"] == "fulfilled" for r in results) else "not_fulfilled",
        }
    passed = all(row["verdict"] == "fulfilled" for row in rows.values())
    summary = {
        "schema_version": "harness-probe-report/v1",
        "suite": SUITE,
        "runner_status": "completed",
        "full_suite": scenarios == SCENARIOS,
        "verdict": "fulfilled" if passed else "not_fulfilled",
        "suite_sha256": digest_file(output / "suite.json"),
        "harnesses": rows,
        "limits": [
            "local harness runtime and controlled Chat/Responses model only; no remote sandbox qualification",
            "all current checks are P0, including ownership and call-to-step checks",
            "multimodal, compaction, parallelism and deployment health are outside this suite",
        ],
    }
    (output / "conformance_report.md").write_text(render_matrices(rows))
    temporary = output / ".conformance_summary.json.tmp"
    temporary.write_text(_json(summary))
    os.replace(temporary, output / "conformance_summary.json")
    return summary, 2 if execution_error else 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harness", action="append", choices=HARNESSES, help="repeat for a matrix; defaults to all four"
    )
    parser.add_argument(
        "--scenario", action="append", choices=[s.name for s in SCENARIOS], help="run a diagnostic subset"
    )
    parser.add_argument("--output", type=Path, help="new directory for rollouts, captures, witnesses and reports")
    parser.add_argument("--timeout", type=float, default=90, help="seconds per harness episode (default: 90)")
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args(argv)
    if args.list_scenarios:
        print(_json(suite_manifest(SCENARIOS)), end="")
        return 0
    if args.output is None:
        parser.error("--output is required")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    harnesses = list(dict.fromkeys(args.harness or HARNESSES))
    scenarios = tuple(s for s in SCENARIOS if not args.scenario or s.name in args.scenario)
    try:
        _, code = run_suite(harnesses=harnesses, scenarios=scenarios, output=args.output, timeout=args.timeout)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"runner_error: {exc}", file=sys.stderr)
        return 2
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(
            "The job summary links the uploaded report and probe logs (probes/conformance_report.md in the artifact)."
        )
    else:
        print(args.output.resolve() / "conformance_report.md")
    return code
