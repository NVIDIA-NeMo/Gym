#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare and report matched native/Pants executions of Gym's routed CPU CI."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from full_ci_aggregate import verify_accounting
from full_ci_history import NATIVE_JOBS, ci_contract
from full_ci_junit import compare_reports, summarize_reports


CPU = "nemo-ci-aws-cpu-4-16"
EXCLUSIONS = [
    "PR administration: preflight authorization, copyright, secrets and guardwords checks",
    "Fern configuration/preview and optional harness-conformance workflow",
    "Nightly-only container build, GPU/provider E2E and wheel-install smoke",
]


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def source_patterns(text: str, group: str) -> list[str]:
    match = re.search(rf"^          {group}:\n((?:            - .*\n)+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Unrecognized source routing contract: missing {group}")
    return [line.strip()[2:].strip("'\"") for line in match[1].splitlines()]


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def routing(source: Path, parent: str, force_full: bool) -> dict:
    action = (source / ".github/actions/classify-changes/action.yml").read_text()
    doc_patterns = source_patterns(action, "doc")
    server_patterns = source_patterns(action, "server")
    # Keep the native shell decision itself authoritative, rather than duplicating its logic.
    decision = action.split("      run: |\n")[-1]
    decision = "\n".join(line[8:] for line in decision.splitlines())
    changed = git("diff", "--name-only", "--no-renames", parent, "HEAD", cwd=source).splitlines()
    servers = [path for path in changed if matches(path, server_patterns)]
    other = [path for path in changed if not matches(path, doc_patterns + server_patterns)]
    output = source / git("rev-parse", "--git-path", "pants-routing-output", cwd=source)
    output.unlink(missing_ok=True)
    env = dict(
        os.environ,
        FORCE_RUN_ALL=str(force_full).lower(),
        OTHER_COUNT=str(len(other)),
        SERVER_CHANGED=str(bool(servers)).lower(),
        DOC_CHANGED=str(any(matches(p, doc_patterns) for p in changed)).lower(),
        GITHUB_OUTPUT=str(output.resolve()),
    )
    subprocess.run(["bash", "-e", "-u", "-o", "pipefail", "-c", decision], env=env, check=True, cwd=source)
    result = dict(line.split("=", 1) for line in output.read_text().splitlines())
    output.unlink()
    workflow = (source / ".github/workflows/unit-tests.yml").read_text()
    session_block = workflow.split("id: session-routing-changes", 1)[1].split("files: |", 1)[1].split("\n\n", 1)[0]
    session_patterns = [line.strip() for line in session_block.splitlines() if line.strip()]
    result.update(
        changed_files=changed,
        changed_server_files=" ".join(servers),
        session_routing=force_full or any(matches(p, session_patterns) for p in changed),
    )
    contract = ci_contract(source)
    legacy = contract["layout"] == "legacy-combined"

    def item(job: str, shard: int = 0, count: int = 1) -> dict:
        return {"job": job, "shard": shard, "shards": count, **contract["jobs"][job]}

    items = [item("lint")]
    run_core = result["run_full"] == "true" or result["run_servers"] == "true"
    if run_core:
        if legacy:
            if result["run_servers"] == "true":
                raise ValueError("Historical combined changed-server routing requires full-scope replay")
            items.append(item("legacy-core"))
        else:
            items += [item("core", i, contract["core_shards"]) for i in range(contract["core_shards"])]
            items.append(item("sandbox"))
    if result["run_full"] == "true":
        items += [item("servers", i, contract["server_shards"]) for i in range(contract["server_shards"])]
    elif result["run_servers"] == "true":
        items.append(item("changed-servers"))
    if result["session_routing"]:
        items.append(item("session-routing"))
    result["ci_contract"] = contract
    result["separate_coverage"] = run_core and not legacy
    result["matrix"] = {"include": [dict(item, engine=engine) for engine in ("baseline", "pants") for item in items]}
    result["source_contracts"] = {
        p: hashlib.sha256((source / p).read_bytes()).hexdigest()
        for p in (
            ".github/actions/classify-changes/action.yml",
            ".github/workflows/unit-tests.yml",
            ".github/workflows/code-linting.yml",
        )
    }
    return result


def prepare(args: argparse.Namespace) -> None:
    result = routing(args.source, args.parent, args.force_full)
    result.update(
        source_sha=git("rev-parse", "HEAD", cwd=args.source),
        parent_sha=args.parent,
        experiment_sha=args.experiment_sha,
        force_full=args.force_full,
        execution_mode=args.execution_mode,
        cache_namespace=args.cache_namespace,
        replay_token=args.replay_token,
        exclusions=EXCLUSIONS,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for group, core in (("core_matrix", True), ("other_matrix", False)):
            selected = [
                {key: value for key, value in item.items() if key != "engine"}
                for item in result["matrix"]["include"]
                if item["engine"] == "baseline" and (item["job"] in NATIVE_JOBS) == core
            ]
            print(group + "=" + json.dumps({"include": selected}), file=stream)
        print("coverage=" + str(result["run_full"] == "true" or result["run_servers"] == "true").lower(), file=stream)
        print("separate_coverage=" + str(result["separate_coverage"]).lower(), file=stream)
        print("uv_version=" + result["ci_contract"]["uv_version"], file=stream)
        coverage = result["ci_contract"]["coverage"] or {"runner": "ubuntu-latest", "timeout": 360}
        print("coverage_runner=" + coverage["runner"], file=stream)
        print("coverage_timeout=" + str(coverage["timeout"]), file=stream)
        print("changed_server_files=" + result["changed_server_files"], file=stream)


def seconds(start: str, end: str) -> float:
    return (
        datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))
    ).total_seconds()


def timing(jobs: list[dict], engine: str, created_at: str | None = None) -> dict:
    # Reusable-workflow jobs are named "<arm> / <child>" by GitHub. Exclude
    # legitimately skipped routing branches; expected-job counts still fail if
    # a required shard was skipped.
    selected = [job for job in jobs if job["name"].startswith(engine + " / ") and job.get("conclusion") != "skipped"]
    completed = [job for job in selected if job.get("started_at") and job.get("completed_at")]
    return {
        "job_count": len(selected),
        "job_seconds": sum(seconds(j["started_at"], j["completed_at"]) for j in completed),
        "elapsed_seconds": seconds(min(j["started_at"] for j in completed), max(j["completed_at"] for j in completed))
        if completed
        else None,
        "feedback_seconds": seconds(created_at, max(j["completed_at"] for j in completed))
        if completed and created_at
        else None,
        "successful": bool(selected) and all(j.get("conclusion") == "success" for j in selected),
        "jobs": [
            {
                "name": job["name"],
                "conclusion": job.get("conclusion"),
                "url": job.get("html_url"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "job_seconds": seconds(job["started_at"], job["completed_at"])
                if job.get("started_at") and job.get("completed_at")
                else None,
                "feedback_seconds": seconds(created_at, job["completed_at"])
                if created_at and job.get("completed_at")
                else None,
            }
            for job in selected
        ],
    }


def timing_scorecard(result: dict, jobs: list[dict]) -> dict:
    """Compare complete occupied job times only after whole-workload validation."""
    route = result["routing"]
    scopes = {
        "native_unit_tests": {"core", "sandbox", "legacy-core"},
        "server_tests": {"servers", "changed-servers"},
        "full_ci": None,
    }
    scorecard = {
        "source_sha": route["source_sha"],
        "cache_warmth": "unknown",
        "replay_phase": "unknown",
        "valid": result["valid"],
        "scope_note": "Native unit tests include core and sandbox jobs; legacy jobs also include their embedded coverage gate. Separate coverage, lint and session-routing jobs belong to full CI.",
        "timing_note": "Complete GitHub job times include setup, cache restore/save and cleanup. Execution span includes gaps between selected jobs; runner-minutes sum assigned job durations. Cache restore/save conditions are not assessed by this reporter, so speed ratios are withheld.",
        "scopes": {},
    }
    for scope, kinds in scopes.items():
        selected = [
            job
            for job in jobs
            if job.get("conclusion") != "skipped"
            and len(job["name"].split(" / ")) >= 2
            and (kinds is None or job["name"].split(" / ")[1] in kinds)
        ]
        arms = {}
        for engine in ("baseline", "pants"):
            required = [job for job in selected if job["name"].startswith(engine + " / ")]
            assigned = [job for job in required if type(job.get("runner_id")) is int and job["runner_id"] > 0]
            value = timing(assigned, engine)
            accounting = [
                proof
                for key, proof in result.get("test_accounting", {}).items()
                if key.startswith(engine + "/") and (kinds is None or key.split("/", 1)[1].rsplit("-", 1)[0] in kinds)
            ]
            value["runner_minutes"] = value["job_seconds"] / 60
            value["test_counts"] = {
                key: sum(proof["counts"][key] for proof in accounting)
                for key in ("forced", "executed", "cached", "collection_skips", "total")
            }
            value["accounting_valid"] = bool(accounting) and all(proof["valid"] for proof in accounting)
            value["timing_complete"] = (
                bool(value["jobs"])
                and all(job["job_seconds"] is not None and job["job_seconds"] >= 0 for job in value["jobs"])
                and len(assigned) == len(required)
            )
            arms[engine] = value
        valid = result["valid"] and all(
            value["successful"] and value["accounting_valid"] and value["timing_complete"] for value in arms.values()
        )
        pairs = {}
        for engine, value in arms.items():
            for job in value["jobs"]:
                pairs.setdefault(job["name"].removeprefix(engine + " / "), {})[engine] = job["job_seconds"]
        scorecard["scopes"][scope] = {
            "valid": valid,
            "arms": arms,
            "paired_job_seconds": pairs,
            "baseline_over_pants": {"elapsed_seconds": None, "runner_minutes": None},
            "ratio_unavailable_reason": "Comparison failed validation" if not valid else "Cache controls not assessed",
        }
    return scorecard


def coverage_summary(path: Path) -> dict:
    """Audit portable raw coverage without installing a second coverage version.

    Coverage 7 stores measured file names and line bitmaps (or executed branch
    arcs) in SQLite. A database containing only empty/omitted files is not proof
    that the repository code was measured, even when the file itself exists.
    """

    def repository_path(value: str) -> str | None:
        value = value.replace("\\", "/")
        if value.startswith("nemo_gym/"):
            return value
        if "/nemo_gym/" in value:
            return "nemo_gym/" + value.split("/nemo_gym/", 1)[1]
        return None

    result = {"valid": False, "errors": [], "measured_files": [], "executed_files": [], "executed_line_count": 0}
    if not path.is_file():
        result["errors"].append("Missing raw coverage database")
        return result
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
            files = {number: repository_path(name) for number, name in database.execute("SELECT id, path FROM file")}
            result["measured_files"] = sorted({name for name in files.values() if name})
            tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            executed = set()
            line_count = 0
            if "line_bits" in tables:
                for file_id, bitmap in database.execute("SELECT file_id, numbits FROM line_bits"):
                    if files.get(file_id) and any(bitmap):
                        executed.add(files[file_id])
                        line_count += sum(byte.bit_count() for byte in bitmap)
            if "arc" in tables:
                lines = set()
                for file_id, start, end in database.execute("SELECT file_id, fromno, tono FROM arc"):
                    if files.get(file_id):
                        lines.update((file_id, number) for number in (start, end) if number > 0)
                executed.update(files[file_id] for file_id, _ in lines)
                line_count += len(lines)
            result["executed_files"] = sorted(executed)
            result["executed_line_count"] = line_count
            if not executed:
                result["errors"].append("Raw coverage contains no executed nemo_gym source lines")
    except (sqlite3.Error, OSError, TypeError) as error:
        result["errors"].append(f"Unreadable coverage data: {error}")
    result["valid"] = not result["errors"]
    return result


def execution_description(execution: dict) -> str:
    """Describe recorded execution, including historical workspace-only runs."""
    pants = [value for name, value in execution.items() if name.startswith("full-ci-pants-")]
    if not pants:
        return "Pants execution metadata is unavailable; no implementation or cache benefit is inferred."
    native = sum(value.get("implementation") == "native Python test targets" for value in pants)
    workspace = sum(value.get("implementation") == "whole-job workspace execution" for value in pants)
    cache_modes = sorted({str(value.get("pants_result_cache")) for value in pants})
    return (
        f"Recorded Pants execution: {native} native Python test jobs and {workspace} whole-job workspace jobs. "
        f"Result-cache modes: {', '.join(cache_modes)}. "
        "Session-only workspace jobs never replay successes from another run. "
        "Native reuse is counted only from retained Pants result provenance, JUnit and raw coverage; "
        "a cache-policy label alone does not establish a cache hit."
    )


def report(args: argparse.Namespace) -> None:
    route = json.loads(args.route.read_text())
    pages = json.loads(args.jobs.read_text())
    jobs = [job for page in (pages if isinstance(pages, list) else [pages]) for job in page["jobs"]]
    run_path = getattr(args, "run", None)
    run = json.loads(run_path.read_text()) if run_path else {}
    result = {
        "routing": route,
        "run": {key: run.get(key) for key in ("id", "run_attempt", "created_at", "html_url", "head_sha")},
        "timing": {
            engine: timing(jobs, engine, run.get("created_at") if run.get("run_attempt", 1) == 1 else None)
            for engine in ("baseline", "pants")
        },
    }
    errors = []
    if run.get("run_attempt", 1) != 1:
        errors.append(
            "Rerun attempts are not comparable: dispatch a fresh workflow so feedback time has a shared creation timestamp."
        )
    for item in route["matrix"]["include"]:
        directory = args.artifacts / f"full-ci-{item['engine']}-{item['job']}-{item['shard']}"
        metadata = directory / "metadata.json"
        if not metadata.is_file():
            errors.append(f"Missing execution metadata: {directory.name}")
        else:
            value = json.loads(metadata.read_text())
            result.setdefault("execution", {})[directory.name] = {
                key: value.get(key)
                for key in ("implementation", "pants_result_cache", "execution_provenance", "execution_mode")
            }
            if value.get("exit_code") != 0 or value.get("command_exit_code") != 0:
                errors.append(f"Execution failed: {directory.name}")
            if value.get("source_sha") != route["source_sha"]:
                errors.append(f"Wrong sampled source: {directory.name}")
            if value.get("implementation") == "native Python test targets" and route.get("execution_mode"):
                if value.get("execution_mode") != route["execution_mode"]:
                    errors.append(f"Native execution mode differs from requested comparison: {directory.name}")
    comparisons = {}
    for item in route["matrix"]["include"]:
        if item["engine"] != "baseline":
            continue
        label = f"{item['job']}-{item['shard']}"
        pair = {}
        manifests = {}
        for engine in ("baseline", "pants"):
            directory = args.artifacts / f"full-ci-{engine}-{label}"
            manifest = directory / "manifest.json"
            expected_reports = None
            if manifest.is_file():
                manifests[engine] = json.loads(manifest.read_text())
                if item["job"] in ("servers", "changed-servers"):
                    expected_reports = [server.replace("/", "__") + ".xml" for server in manifests[engine]["servers"]]
                elif item["job"] == "lint":
                    expected_reports = []
                elif item["job"] == "legacy-core":
                    expected_reports = ["core.xml", "sandbox.xml"]
                else:
                    expected_reports = [item["job"] + ".xml"]
            else:
                errors.append(f"Missing test scope manifest: {engine}/{label}")
            pair[engine] = summarize_reports(directory / "junit", expected_reports=expected_reports)
        comparisons[label] = compare_reports(pair["baseline"], pair["pants"])
        if not comparisons[label]["equal"]:
            errors.append(f"Test outcome mismatch or incomplete evidence: {label}")
        if len(manifests) == 2 and manifests["baseline"] != manifests["pants"]:
            errors.append(f"Test scope manifest mismatch: {label}")
        for engine in ("baseline", "pants"):
            directory = args.artifacts / f"full-ci-{engine}-{label}"
            execution = result.get("execution", {}).get(directory.name, {})
            if (
                engine == "pants"
                and execution.get("implementation") == "native Python test targets"
                and (
                    execution.get("execution_mode") == "incremental"
                    or execution.get("pants_result_cache") == "incremental-native-tests"
                    or (directory / "native/test-accounting.json").exists()
                    or item["job"] == "legacy-core"
                )
            ):
                if item["job"] == "legacy-core":
                    from collections import Counter

                    groups = {}
                    for group in ("core", "sandbox"):
                        group_root = directory / "native" / group
                        group_report = summarize_reports(group_root / "junit", expected_reports=[group + ".xml"])
                        groups[group] = verify_accounting(
                            group_root, report=group_report, source_sha=route["source_sha"]
                        )
                    # Nested accounting also retains per-batch XML, so select
                    # only the two canonical reports when checking copied cases.
                    group_cases = Counter()
                    for group in ("core", "sandbox"):
                        for case in summarize_reports(directory / "native" / group / "junit")["cases"]:
                            group_cases[(case["file"], case["classname"], case["name"], case["outcome"])] += case[
                                "count"
                            ]
                    copied_cases = Counter(
                        {
                            (case["file"], case["classname"], case["name"], case["outcome"]): case["count"]
                            for case in pair[engine]["cases"]
                        }
                    )
                    accounting = {
                        "valid": all(value["valid"] for value in groups.values()) and group_cases == copied_cases,
                        "errors": [
                            f"{group}: {error}" for group, value in groups.items() for error in value["errors"]
                        ],
                        "groups": groups,
                        "counts": {
                            key: sum(value["counts"][key] for value in groups.values())
                            for key in ("forced", "executed", "cached", "collection_skips", "total")
                        },
                    }
                    if group_cases != copied_cases:
                        accounting["errors"].append("Marker group cases do not partition the complete copied JUnit")
                else:
                    accounting = verify_accounting(
                        directory / "native", report=pair[engine], source_sha=route["source_sha"]
                    )
                if not accounting["valid"]:
                    errors.append(f"Invalid execution/cache accounting for {label}: {accounting['errors']}")
                elif execution.get("execution_mode") == "forced" and (
                    accounting["counts"]["cached"] or accounting["counts"]["executed"]
                ):
                    errors.append(f"Forced comparison contains cacheable test execution: {label}")
            else:
                # Historical metadata establishes forced execution only for the
                # explicit existing modes; never relabel unknown cache policy.
                known_forced = engine == "baseline" or execution.get("pants_result_cache") in {
                    "session-only",
                    "forced-native-tests",
                }
                accounting = {
                    "valid": known_forced,
                    "counts": {
                        "forced": pair[engine]["testcase_count"] if known_forced else 0,
                        "executed": 0,
                        "cached": 0,
                        "collection_skips": 0,
                        "total": pair[engine]["testcase_count"],
                    },
                }
            result.setdefault("test_accounting", {})[f"{engine}/{label}"] = accounting
        if item["job"] in ("core", "sandbox", "legacy-core"):
            coverage_name = (
                f".coverage.core-{item['shard']}"
                if item["job"] == "core"
                else ".coverage.legacy-core"
                if item["job"] == "legacy-core"
                else ".coverage.sandbox"
            )
            measured = {}
            for engine in ("baseline", "pants"):
                coverage = coverage_summary(args.artifacts / f"full-ci-{engine}-{label}" / coverage_name)
                result.setdefault("coverage", {})[f"{engine}/{label}"] = coverage
                measured[engine] = coverage["measured_files"]
                if not coverage["valid"]:
                    errors.append(f"Invalid raw coverage for {engine}/{label}: {coverage['errors']}")
            if measured["baseline"] != measured["pants"]:
                errors.append(f"Measured nemo_gym coverage scope differs: {label}")
            if item["job"] == "legacy-core":
                from full_ci_aggregate import _coverage_lines

                native = args.artifacts / f"full-ci-pants-{label}" / "native"
                try:
                    expected_lines = {}
                    for group in ("core", "sandbox"):
                        proof = json.loads((native / group / "test-accounting.json").read_text())
                        raw_name = f".coverage.core-{proof['shard']}" if group == "core" else ".coverage.sandbox"
                        for name, bitmap in _coverage_lines(native / group / raw_name).items():
                            expected_lines[name] = expected_lines.get(name, 0) | bitmap
                    if _coverage_lines(native.parent / coverage_name) != expected_lines:
                        errors.append(f"Combined legacy coverage omitted or added marker-group lines: {label}")
                except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
                    errors.append(f"Invalid legacy coverage union for {label}: {error}")
    result["outcomes"] = comparisons
    for engine in ("baseline", "pants"):
        expected = len([item for item in route["matrix"]["include"] if item["engine"] == engine])
        expected += int(route.get("separate_coverage", route["run_full"] == "true" or route["run_servers"] == "true"))
        if result["timing"][engine]["job_count"] != expected:
            errors.append(f"Missing or additional measured jobs for {engine}")
    result["errors"] = errors
    result["valid"] = not errors and all(value["successful"] for value in result["timing"].values())
    result["timing_scorecard"] = timing_scorecard(result, jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Full routed CPU CI comparison",
        "",
        f"Source: `{route['source_sha']}`; experiment: `{route['experiment_sha']}`.",
        "",
        "Both arms use the same source diff, runner labels and the sampled main revision's test/job partition.",
        "Feedback time runs from shared workflow creation to each arm's last required job completion, including shared preparation and initial queue delay. Execution span starts at that arm's first job. Both include cache/setup overhead and subsequent queue gaps; runner-minutes sum occupied job time.",
        execution_description(result.get("execution", {})),
        "",
        "| Arm | Feedback seconds | Execution span seconds | Runner-minutes | Jobs |",
        "|---|---:|---:|---:|---:|",
    ]
    for engine, value in result["timing"].items():
        lines.append(
            f"| {engine} | {value['feedback_seconds']} | {value['elapsed_seconds']} | {value['job_seconds'] / 60:.2f} | {value['job_count']} |"
        )
    scorecard = result["timing_scorecard"]
    lines += [
        "",
        "Unit-test timing scorecard (cache warmth: unknown).",
        scorecard["scope_note"],
        "Speed ratios are withheld: the reporter does not assess cache restore/save controls. Failed validation also prevents a speed claim.",
        "",
        "| Scope | Arm | Execution span seconds | Runner-minutes | Forced | Executed cacheable | Reused | Valid |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for scope, card in scorecard["scopes"].items():
        for engine, value in card["arms"].items():
            counts = value["test_counts"]
            lines.append(
                f"| {scope} | {engine} | {value['elapsed_seconds']} | {value['runner_minutes']:.2f} | "
                f"{counts['forced']} | {counts['executed']} | {counts['cached']} | {card['valid']} |"
            )
    lines += ["", "| Arm / required job | Result | Job seconds | Feedback seconds |", "|---|---|---:|---:|"]
    for value in result["timing"].values():
        for job in value["jobs"]:
            lines.append(f"| {job['name']} | {job['conclusion']} | {job['job_seconds']} | {job['feedback_seconds']} |")
    lines += [
        "",
        "| Test scope | Forced | Executed cacheable | Reused | Collection skips | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope, accounting in result.get("test_accounting", {}).items():
        counts = accounting["counts"]
        lines.append(
            f"| {scope} | {counts['forced']} | {counts['executed']} | {counts['cached']} | "
            f"{counts['collection_skips']} | {counts['total']} |"
        )
    lines += [
        "",
        "Baseline/Pants validation: " + ("passed" if result["valid"] else "FAILED; no speedup claim"),
        "",
        *errors,
        "",
        "Outside this paired timing (unchanged workflows):",
        *["- " + value for value in EXCLUSIONS],
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    if not result["valid"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--parent", required=True)
    prep.add_argument("--experiment-sha", required=True)
    prep.add_argument("--force-full", action="store_true")
    prep.add_argument("--execution-mode", choices=("incremental", "forced"), default="incremental")
    prep.add_argument("--cache-namespace", default="live")
    prep.add_argument("--replay-token", default="")
    prep.add_argument("--output", type=Path, required=True)
    summary = sub.add_parser("report")
    for name in ("route", "jobs", "run", "artifacts", "output"):
        summary.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else report)(args)


if __name__ == "__main__":
    main()
