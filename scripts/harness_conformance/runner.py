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
from dataclasses import asdict
from pathlib import Path

import psutil

from nemo_gym.config import ConfigError
from nemo_gym.rollouts import harness_capabilities
from nemo_gym.rollouts.harness_capabilities import (
    NAMES,
    EvidenceScope,
    ToolTimingScope,
    digest_file,
    gate_passes,
    inspect_behavior,
    inspect_bundle,
    inspect_health,
    inspect_record,
    json_rows,
    render_matrices,
)

from .episode import HARNESSES
from .registry import P0_TOOL_TIMING_HARNESSES, tool_timing_inventory
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


def inspect_episode(
    scenario: Scenario, directory: Path, execution: dict, *, tool_timing: ToolTimingScope | None = None
) -> dict:
    """Inspect artifacts, behavior and health independently of process exit status."""
    witness_path = directory / "witness.json"
    witness = json.loads(witness_path.read_text()) if witness_path.exists() else None
    rollouts_path = directory / "rollouts.jsonl"
    failures_path = directory / "rollouts_failures.jsonl"
    health_report = directory / "health" / "quality_summary.json"

    bundle = rollouts_path
    records = list(json_rows(bundle)) if bundle.exists() else []
    if not records and scenario.terminal_error and failures_path.is_file():
        # A terminal model error fails the episode, so its evidence is retained in the failures file.
        bundle = failures_path
        records = list(json_rows(bundle))

    record = records[0][1] if len(records) == 1 else None
    if record is not None and bundle == rollouts_path:
        delivery = "rollout"
    elif failures_path.exists() and failures_path.stat().st_size:
        delivery = "failure_record"
    else:
        delivery = "missing"

    scope = EvidenceScope(
        tools=scenario.tool_steps > 0,
        steps=scenario.steps,
        verifier=not scenario.terminal_error,
        tool_timing=tool_timing if scenario.tool_steps > 0 else None,
    )
    artifact = inspect_record(record, scope=scope)
    checks = artifact["checks"] + inspect_behavior(
        witness,
        record,
        http_errors=scenario.http_errors,
        terminal_error=scenario.terminal_error,
        tool_steps=scenario.tool_steps,
        expected_reward=scenario.expected_reward,
        judge_verifier=scenario.judge_verifier,
        fingerprint=_fingerprint,
    )
    health_inputs = [path for path in [rollouts_path, failures_path] if path.is_file()]
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
        "tool_timing_scope": asdict(scope.tool_timing) if scope.tool_timing else None,
        "scenario": scenario.name,
        "exercised": record is not None,
        "verdict": "fulfilled" if gate_passes(checks) else "not_fulfilled",
        "issues": [reason for c in checks if c["status"] == "fail" for reason in c["reasons"]],
        "checks": checks,
        "behavioral_status": "pass" if behavior_passed else "fail",
        "execution": execution,
        "delivery": delivery,
        "model_attempts": len((witness or {}).get("attempts", [])),
        "evidence": evidence,
        "artifact_report": report,
        "health_report": str(health_report.relative_to(directory)) if health_inputs else None,
        "hashes": {
            path.name: digest_file(path)
            for path in (
                rollouts_path,
                failures_path,
                witness_path,
                directory / "runtime.json",
                directory / "launch.json",
                health_report,
            )
            if path.exists()
        },
    }


def run_suite(
    *,
    harnesses: list[str],
    scenarios: tuple[Scenario, ...],
    output: Path,
    timeout: float,
    tool_timing_scopes: dict[str, ToolTimingScope] | None = None,
) -> tuple[dict, int]:
    """Run each requested harness/scenario in a fresh process and publish the completed report."""
    if not harnesses or not scenarios:
        raise ValueError("at least one harness and one scenario are required")
    tool_timing_scopes = tool_timing_scopes or {}
    if set(tool_timing_scopes) - set(harnesses):
        raise ValueError("timing scopes must refer to selected runnable harnesses")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = suite_manifest(scenarios)
    manifest.update(
        harnesses=harnesses,
        tool_timing_targets=tool_timing_inventory(),
        tool_timing_scopes={name: asdict(scope) for name, scope in tool_timing_scopes.items()},
        timeout=timeout,
        python=sys.version,
        runner_sources={p.name: digest_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        checker_sources={
            p.name: digest_file(p) for p in sorted(Path(harness_capabilities.__file__).parent.glob("*.py"))
        },
        health_sources={
            str(p.relative_to(ROOT)): digest_file(p)
            for p in [
                ROOT / "nemo_gym/rollouts/rollout_health.py",
                *sorted((ROOT / "nemo_gym/rollouts/health").glob("*.py")),
            ]
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
                result = inspect_episode(scenario, directory, execution, tool_timing=tool_timing_scopes.get(harness))
            except (ConfigError, OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
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
        "tool_timing_targets": tool_timing_inventory(),
        "tool_timing_scopes": {name: asdict(scope) for name, scope in tool_timing_scopes.items()},
        "limits": [
            "local harness runtime and controlled Chat/Responses model only; no remote sandbox qualification",
            "multimodal, compaction, parallelism and deployment health are outside this suite",
        ],
    }
    (output / "conformance_report.md").write_text(render_matrices(rows))
    temporary = output / ".conformance_summary.json.tmp"
    temporary.write_text(_json(summary))
    os.replace(temporary, output / "conformance_summary.json")
    return summary, 2 if execution_error else 0 if passed else 1


def load_tool_timing_scopes(path: Path) -> dict[str, ToolTimingScope]:
    """Load explicit expectations; runtime artifacts retain the independently observed version."""
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or set(values) - set(P0_TOOL_TIMING_HARNESSES):
        raise ValueError("timing scopes must map registered timing targets to scope declarations")
    return {name: ToolTimingScope(**scope) for name, scope in values.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harness",
        action="append",
        choices=HARNESSES,
        help="repeat for a matrix; defaults to all registered harnesses",
    )
    parser.add_argument(
        "--scenario", action="append", choices=[s.name for s in SCENARIOS], help="run a diagnostic subset"
    )
    parser.add_argument("--output", type=Path, help="new directory for rollouts, captures, witnesses and reports")
    parser.add_argument("--timeout", type=float, default=90, help="seconds per harness episode (default: 90)")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--list-timing-targets", action="store_true")
    parser.add_argument(
        "--tool-timing-scopes",
        type=Path,
        help="JSON object mapping selected harness names to ToolTimingScope declarations",
    )
    args = parser.parse_args(argv)
    if args.list_timing_targets:
        print(_json(tool_timing_inventory()), end="")
        return 0
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
        scopes = load_tool_timing_scopes(args.tool_timing_scopes) if args.tool_timing_scopes else {}
        _, code = run_suite(
            harnesses=harnesses,
            scenarios=scenarios,
            output=args.output,
            timeout=args.timeout,
            tool_timing_scopes=scopes,
        )
    except (ConfigError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"runner_error: {exc}", file=sys.stderr)
        return 2
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(
            "The job summary links the uploaded report and probe logs (probes/conformance_report.md in the artifact)."
        )
    else:
        print(args.output.resolve() / "conformance_report.md")
    return code
