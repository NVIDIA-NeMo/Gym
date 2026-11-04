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
    items = [{"job": "lint", "shard": 0, "shards": 1, "runner": "ubuntu-latest"}]
    if result["run_full"] == "true" or result["run_servers"] == "true":
        items += [{"job": "core", "shard": i, "shards": 4, "runner": CPU} for i in range(4)]
        items += [{"job": "sandbox", "shard": 0, "shards": 1, "runner": CPU}]
    if result["run_full"] == "true":
        items += [{"job": "servers", "shard": i, "shards": 8, "runner": CPU} for i in range(8)]
    elif result["run_servers"] == "true":
        items += [{"job": "changed-servers", "shard": 0, "shards": 1, "runner": CPU}]
    if result["session_routing"]:
        items += [{"job": "session-routing", "shard": 0, "shards": 1, "runner": "ubuntu-latest"}]
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
        exclusions=EXCLUSIONS,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for group, core in (("core_matrix", True), ("other_matrix", False)):
            selected = [
                {key: value for key, value in item.items() if key != "engine"}
                for item in result["matrix"]["include"]
                if item["engine"] == "baseline" and (item["job"] in ("core", "sandbox")) == core
            ]
            print(group + "=" + json.dumps({"include": selected}), file=stream)
        print("coverage=" + str(result["run_full"] == "true" or result["run_servers"] == "true").lower(), file=stream)
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
        "forced-native-tests means --test-force; session-only workspace jobs never replay successes from another run. "
        "Persistent engine/dependency caches may be restored. These parity runs do not establish test-result reuse or selective pruning."
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
                key: value.get(key) for key in ("implementation", "pants_result_cache", "execution_provenance")
            }
            if value.get("exit_code") != 0 or value.get("command_exit_code") != 0:
                errors.append(f"Execution failed: {directory.name}")
            if value.get("source_sha") != route["source_sha"]:
                errors.append(f"Wrong sampled source: {directory.name}")
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
        if item["job"] in ("core", "sandbox"):
            coverage_name = f".coverage.core-{item['shard']}" if item["job"] == "core" else ".coverage.sandbox"
            measured = {}
            for engine in ("baseline", "pants"):
                coverage = coverage_summary(args.artifacts / f"full-ci-{engine}-{label}" / coverage_name)
                result.setdefault("coverage", {})[f"{engine}/{label}"] = coverage
                measured[engine] = coverage["measured_files"]
                if not coverage["valid"]:
                    errors.append(f"Invalid raw coverage for {engine}/{label}: {coverage['errors']}")
            if measured["baseline"] != measured["pants"]:
                errors.append(f"Measured nemo_gym coverage scope differs: {label}")
    result["outcomes"] = comparisons
    for engine in ("baseline", "pants"):
        expected = len([item for item in route["matrix"]["include"] if item["engine"] == engine])
        expected += int(route["run_full"] == "true" or route["run_servers"] == "true")
        if result["timing"][engine]["job_count"] != expected:
            errors.append(f"Missing or additional measured jobs for {engine}")
    result["errors"] = errors
    result["valid"] = not errors and all(value["successful"] for value in result["timing"].values())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Full routed CPU CI comparison",
        "",
        f"Source: `{route['source_sha']}`; experiment: `{route['experiment_sha']}`.",
        "",
        "Both arms use the same source diff and runner labels, four core shards and up to eight server shards.",
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
    lines += ["", "| Arm / required job | Result | Job seconds | Feedback seconds |", "|---|---|---:|---:|"]
    for value in result["timing"].values():
        for job in value["jobs"]:
            lines.append(f"| {job['name']} | {job['conclusion']} | {job['job_seconds']} | {job['feedback_seconds']} |")
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
    prep.add_argument("--output", type=Path, required=True)
    summary = sub.add_parser("report")
    for name in ("route", "jobs", "run", "artifacts", "output"):
        summary.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else report)(args)


if __name__ == "__main__":
    main()
