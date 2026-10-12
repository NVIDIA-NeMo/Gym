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
from collections import Counter
from pathlib import Path

import psutil

from nemo_gym.harness_capabilities.checker import NAMES
from nemo_gym.harness_capabilities.cli import inspect_bundle
from nemo_gym.harness_capabilities.reader import digest_file, hydrate_record, json_rows

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


def _tool_witness_issues(record: dict, witnessed: list[dict]) -> list[str]:
    """Join independently witnessed tools to retained executions and conversation items."""
    trajectory = record.get("ng_trajectory") or {}
    observations = (record.get("ng_agent_observations") or {}).get("records", [])
    # Match the inspector's supported trajectory/observation fallbacks.
    tools = trajectory.get("tool_calls") or [r for r in observations if r.get("kind") == "tool_call"]
    invocations = [r for r in observations if r.get("kind") == "agent_invocation"] or trajectory.get("invocations", [])
    expected_ids = Counter(t["id"] for t in witnessed)
    if any(count != 1 for count in expected_ids.values()) or expected_ids != Counter(t["tool_call_id"] for t in tools):
        return ["retained tool identities differ from the independent tool witness"]

    issues = []
    by_id = {t["tool_call_id"]: t for t in tools}
    for expected in witnessed:
        tool = by_id[expected["id"]]
        items = [
            item
            for invocation in invocations
            if invocation.get("invocation_id") == tool.get("invocation_id")
            for item in invocation.get("conversation", [])
            if item.get("call_id") == expected["id"]
        ]
        requests = [i for i in items if i.get("type") == "function_call"]
        results = [i for i in items if i.get("type") == "function_call_output"]
        if len(requests) != 1 or len(results) != 1:
            issues.append("retained tool request/result does not join uniquely to the witnessed execution")
            continue
        request = requests[0]
        try:
            arguments = json.loads(request["arguments"])
        except (KeyError, TypeError, ValueError):
            arguments = None
        name = request.get("name")
        if request.get("namespace"):
            name = f"{request['namespace']}__{name}"
        if (
            name != expected["name"]
            or arguments != expected["arguments"]
            or tool.get("tool_name") != request.get("name")
        ):
            issues.append("retained tool name or arguments differ from the independent tool witness")
        # Each probe command terminates with its prescribed code. A nonzero exit
        # must remain a failed execution even when its stdout was retained.
        status = "failed" if expected["exit_code"] else "completed"
        if tool.get("status") != status:
            issues.append("retained tool status differs from the independent tool witness")
        output = results[0].get("output")
        outputs = expected.get("outputs", [])
        if (
            not outputs
            or any(observed != output for observed in outputs)
            or (tool.get("output") is not None and tool["output"] != output)
        ):
            issues.append("retained tool output differs from the independent tool witness")
    return issues


def inspect_episode(scenario: Scenario, directory: Path, execution: dict) -> dict:
    """Require witnessed exercise and a real rollout before counting a TE as passing."""
    issues = []
    if execution["timed_out"]:
        issues.append("episode exceeded its timeout")
    if execution["returncode"] != 0:
        issues.append("episode process failed; see episode.log")
    witness_path = directory / "witness.json"
    witness = json.loads(witness_path.read_text()) if witness_path.exists() else {}
    attempts = witness.get("attempts", [])
    issues.extend(witness.get("violations", []))
    statuses = [attempt["status_code"] for attempt in attempts]
    if witness.get("seeded") != 1:
        issues.append("expected one fresh episode initialization")
    if not attempts:
        issues.append("the harness never reached the controlled model endpoint")
    if statuses[: len(scenario.http_errors)] != list(scenario.http_errors):
        issues.append("not all prescribed model failures were observed")
    if len(scenario.http_errors) > 1 and len(attempts) >= len(scenario.http_errors):
        if any(a["request"] != attempts[0]["request"] for a in attempts[1 : len(scenario.http_errors)]):
            issues.append("retry scenario did not repeat the same request body")
    if scenario.terminal_error:
        if not statuses or any(status != scenario.http_errors[-1] for status in statuses):
            issues.append("terminal model failure was not observed")
    elif not witness.get("finished"):
        issues.append("the harness did not finish the scripted model exchange")
    tools = witness.get("tool_calls", [])
    if len(tools) != scenario.tool_steps or any(
        not tool.get("executed") or not tool.get("result_seen") for tool in tools
    ):
        issues.append("prescribed tool executions and their returned results were not all witnessed")
    verifications = witness.get("verifications", [])
    if len(verifications) != 1 or verifications[0]["reward"] != scenario.expected_reward:
        issues.append("expected verifier outcome was not observed")
    elif not scenario.terminal_error and not verifications[0]["answer_seen"]:
        issues.append("the verifier did not receive the scripted final answer")
    bundle = directory / "rollouts.jsonl"
    records = list(json_rows(bundle)) if bundle.exists() else []
    summary = None
    report = None
    if len(records) != 1:
        issues.append("expected exactly one collected rollout")
    else:
        record = hydrate_record(records[0][1])
        calls = record.get("ng_model_call_capture", {}).get("calls", [])
        expected = Counter(_fingerprint(a["request"], a["status_code"], a["response"]) for a in attempts)
        observed = Counter(_fingerprint(c.get("request"), c.get("status_code"), c.get("response")) for c in calls)
        if expected != observed:
            issues.append("retained model attempts differ from the independent endpoint witness")
        issues.extend(_tool_witness_issues(record, tools))
        if record.get("reward") != scenario.expected_reward:
            issues.append("rollout reward differs from the verifier witness")
        destination, summary = inspect_bundle(bundle, output=directory / "evidence", capture_dir=directory / "capture")
        report = str(destination.relative_to(directory) / "evidence_summary.json")
    exercised = not issues
    evidence = {}
    for key in scenario.evidence:
        verdict = summary["evidence"][key]["verdict"] if summary else "not_fulfilled"
        evidence[key] = {
            "verdict": "fulfilled" if exercised and verdict == "fulfilled" else "not_fulfilled",
            "artifact_verdict": verdict,
        }
    # The two join contracts remain alternatives, just as in the P0 inspector.
    mandatory = [key for key in scenario.evidence if key not in ("TE-8", "TE-9")]
    passed = exercised and all(evidence[key]["verdict"] == "fulfilled" for key in mandatory)
    passed = passed and any(evidence[key]["verdict"] == "fulfilled" for key in ("TE-8", "TE-9"))
    return {
        "scenario": scenario.name,
        "exercised": exercised,
        "verdict": "fulfilled" if passed else "not_fulfilled",
        "issues": issues,
        "execution": execution,
        "model_attempts": len(attempts),
        "evidence": evidence,
        "artifact_report": report,
        "hashes": {
            path.name: digest_file(path)
            for path in (bundle, witness_path, directory / "runtime.json", directory / "launch.json")
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
            "TE-10, P1, multimodal, compaction, parallelism and deployment health are outside this suite",
        ],
    }
    table = [
        "# Live harness P0 probes",
        "",
        "Each cell is passing / exercised / required scenarios.",
        "",
        "| Harness | " + " | ".join(NAMES) + " | Gate |",
        "|---|" + "---|" * (len(NAMES) + 1),
    ]
    for harness, row in rows.items():
        cells = [f"{c['passed']}/{c['observed']}/{c['required']}" for c in row["evidence"].values()]
        table.append("| " + " | ".join([harness, *cells, row["verdict"]]) + " |")
    table += [
        "",
        "TE-8 or TE-9 is sufficient per scenario. A selected subset does not qualify the full suite.",
        "See each scenario_result.json for execution gaps and the unmodified artifact-checker report.",
        "",
    ]
    (output / "conformance_report.md").write_text("\n".join(table))
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
    print(args.output.resolve() / "conformance_report.md")
    return code
