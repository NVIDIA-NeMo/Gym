#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resume one immutable, sequential historical full-CI replay without redispatching intents."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from full_ci_history import ci_contract
from native_core.incremental import select_policy
from native_core.lock_assets import fingerprint, lock_assets, source_uv_version


REPOSITORY = "NVIDIA-NeMo/Gym"
BRANCH = "glovisotto/pants-pilot"
WORKFLOW = "pants-full-ci.yml"
CUTOFF = "2026-11-09T17:00:00Z"
RECONCILIATION_MINUTES = 15
PREFIX = f"repos/{REPOSITORY}"


def _write(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@contextmanager
def driver_lock(output: Path):
    """The kernel releases this exclusive lock even when the driver is terminated."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "driver.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class GitHub:
    def api(self, path: str, *, paginate: bool = False, payload: dict | None = None) -> dict | list | None:
        command = ["gh", "api", f"{PREFIX}/{path}"]
        if paginate:
            command += ["--paginate", "--slurp"]
        if payload is not None:
            command += ["--method", "POST", "--input", "-"]
        result = subprocess.run(
            command,
            input=json.dumps(payload) if payload is not None else None,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout) if result.stdout.strip() else None

    def download(self, path: str, destination: Path) -> None:
        with destination.open("wb") as stream:
            subprocess.run(["gh", "api", f"{PREFIX}/{path}"], stdout=stream, check=True)


class Driver:
    def __init__(
        self, *, repo: Path, plan: Path, output: Path, controller_sha: str | None = None, github: GitHub | None = None
    ) -> None:
        self.repo, self.output, self.github = repo.resolve(), output.resolve(), github or GitHub()
        if self.output.is_relative_to(self.repo):
            raise ValueError("Replay evidence must live outside the controller checkout")
        self.plan = json.loads(plan.read_text())
        digest = hashlib.sha256(
            json.dumps(
                {k: v for k, v in self.plan.items() if k != "plan_id"}, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if self.plan.get("plan_id") != digest or not self.plan.get("snapshots"):
            raise ValueError("Replay plan is empty or its immutable content digest differs")
        retained_plan = self.output / "plan.json"
        if retained_plan.exists() and json.loads(retained_plan.read_text()) != self.plan:
            raise ValueError("Retained replay plan differs from the requested immutable plan")
        self.path = self.output / "state.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if (
                self.state["plan_id"] != digest
                or self.state["repository"] != REPOSITORY
                or self.state["branch"] != BRANCH
            ):
                raise ValueError("Existing replay belongs to a different plan/repository/branch")
            if controller_sha and controller_sha != self.state["controller_sha"]:
                raise ValueError("Existing replay has a different frozen controller")
        else:
            snapshots = self.plan["snapshots"]
            entries = [{"source_sha": s["source_sha"], "kind": "snapshot", "status": "pending"} for s in snapshots]
            entries.insert(1, {"source_sha": snapshots[0]["source_sha"], "kind": "warm-control", "status": "pending"})
            resolved_controller = controller_sha or _git(self.repo, "rev-parse", "HEAD")
            self.state = {
                "schema_version": 1,
                "plan_id": digest,
                "repository": REPOSITORY,
                "branch": BRANCH,
                "controller_sha": resolved_controller,
                "cache_namespace": f"replay-{digest[:12]}-{resolved_controller[:8]}",
                "cutoff": CUTOFF,
                "entries": entries,
                "current": 0,
                "created_at": _now(),
            }
            self.save()
        if not retained_plan.exists():
            _write(retained_plan, self.plan)

    def save(self) -> None:
        _write(self.path, self.state)

    def guard(self) -> str:
        """New dispatches require the unchanged, published, clean controller."""
        if _git(self.repo, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("Controller checkout is not pristine")
        if (
            _git(self.repo, "branch", "--show-current") != BRANCH
            or _git(self.repo, "rev-parse", "HEAD") != self.state["controller_sha"]
        ):
            raise ValueError("Controller branch/HEAD differs from the frozen replay")
        remote = self.github.api(f"git/ref/heads/{BRANCH}")
        if remote["object"]["sha"] != self.state["controller_sha"]:
            raise ValueError("Published controller HEAD differs; no dispatch attempted")
        paths = (
            subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "-z",
                    "scripts/pants/full_ci*.py",
                    "scripts/pants/full_ci.toml",
                    "scripts/pants/full_ci.BUILD",
                    "scripts/pants/native_core",
                ],
                cwd=self.repo,
            )
            .decode()
            .split("\0")
        )
        inventory = "".join(
            f"{hashlib.sha256((self.repo / name).read_bytes()).hexdigest()}  {name}\n" for name in paths if name
        )
        return hashlib.sha256(inventory.encode()).hexdigest()

    def preflight(self, entry: dict) -> dict:
        snapshot = next(s for s in self.plan["snapshots"] if s["source_sha"] == entry["source_sha"])
        with tempfile.TemporaryDirectory(prefix="source-", dir=self.output) as temporary:
            source = Path(temporary) / "sample"
            subprocess.run(
                ["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout", str(self.repo), str(source)], check=True
            )
            subprocess.run(["git", "checkout", "--quiet", "--detach", entry["source_sha"]], cwd=source, check=True)
            if _git(source, "status", "--porcelain", "--untracked-files=all"):
                raise ValueError("Historical preflight checkout is not clean")
            if (
                fingerprint(source) != snapshot["dependency_inputs"]
                or source_uv_version(source) != snapshot["uv_version"]
            ):
                raise ValueError("Historical dependency inputs differ from the immutable plan")
            contract = ci_contract(source)
            native = self.repo / "scripts/pants/native_core"
            # The existing native helpers are also standalone scripts and use
            # sibling imports. Support their normal import layout in this CLI.
            sys.path.insert(0, str(native))
            try:
                assets = lock_assets(source, native)
            finally:
                sys.path.remove(str(native))
            policy, decision = select_policy(source, native)
            return {
                "ci_contract": contract,
                "lock_assets": str(assets.relative_to(self.repo)),
                "source_sha": entry["source_sha"],
                "incremental_policy": str(policy.relative_to(self.repo)) if policy else None,
                "incremental_eligibility": decision,
            }

    def readiness(self) -> dict:
        """Avoid overlapping pilots and repeat dispatches into a visibly stalled ARC queue."""
        workflows = {
            f".github/workflows/{name}" for name in (WORKFLOW, "pants-core0-diagnostic.yml", "pants-pilot.yml")
        }
        pages = self.github.api(f"actions/runs?branch={BRANCH}&per_page=100", paginate=True)
        active = [
            run["id"]
            for page in pages
            for run in page["workflow_runs"]
            if run.get("path", "").split("@", 1)[0] in workflows and run.get("status") != "completed"
        ]
        if active:
            return {"ready": False, "state": "active-pilot", "run_ids": active}
        jobs, stalled = [], []
        # Unit tests are reusable: their current jobs belong to the main caller,
        # while listing unit-tests.yml itself returns obsolete standalone runs.
        runs = self.github.api("actions/workflows/cicd-main.yml/runs?branch=main&per_page=2")["workflow_runs"]
        now = datetime.fromisoformat(_now().replace("Z", "+00:00"))
        for run in runs:
            pages = self.github.api(f"actions/runs/{run['id']}/jobs?per_page=100", paginate=True)
            current = [
                job for page in pages for job in page["jobs"] if "nemo-ci-aws-cpu-4-16" in job.get("labels", [])
            ]
            jobs.extend(current)
            age = (now - datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))).total_seconds()
            if age > 900:
                stalled.extend(
                    job["id"] for job in current if job.get("status") != "completed" and not job.get("runner_id")
                )
        # A just-finished replay can itself establish fresh ARC capacity when main
        # has no recent run. Keep that observation separate from cache warmth.
        for entry in self.state["entries"]:
            if entry.get("run_id"):
                path = self.output / "hosted" / str(entry["run_id"]) / "jobs.json"
                if path.is_file():
                    jobs.extend(
                        job
                        for page in json.loads(path.read_text())
                        for job in page["jobs"]
                        if "nemo-ci-aws-cpu-4-16" in job.get("labels", [])
                    )
        recent = [
            job["id"]
            for job in jobs
            if job.get("runner_id")
            and job.get("started_at")
            and 0
            <= (
                now - datetime.fromisoformat((job.get("completed_at") or job["started_at"]).replace("Z", "+00:00"))
            ).total_seconds()
            < 7200
        ]
        if recent:
            return {"ready": True, "state": "recent-arc-assignment", "job_ids": recent}
        if stalled:
            return {"ready": False, "state": "stalled-unassigned-arc", "job_ids": stalled}
        return {
            "ready": self.state.get("unknown_readiness_dispatches", 0) == 0,
            "state": "unknown",
            "reason": "No recent ARC assignment; permit at most one bounded attempt",
        }

    def caches(self, directory: Path, contract: str) -> None:
        subprocess.run(
            [
                sys.executable,
                str(self.repo / "scripts/pants/replay_cache.py"),
                "--output",
                str(directory),
                "--contract",
                contract,
                "--namespace",
                self.state["cache_namespace"],
                "--enforce-limit",
            ],
            check=True,
        )

    def verify_run(self, run: dict, entry: dict) -> None:
        expected = f"Pants CPU {entry['replay_token']} source {entry['source_sha']}"
        if (
            run.get("display_title") != expected
            or run.get("event") != "workflow_dispatch"
            or run.get("head_branch") != BRANCH
            or run.get("head_sha") != self.state["controller_sha"]
            or run.get("run_attempt", 1) != 1
        ):
            raise ValueError("Run token/source/event/branch/controller/attempt differs from persisted intent")

    def recover(self, entry: dict) -> dict | None:
        if entry.get("run_id"):
            run = self.github.api(f"actions/runs/{entry['run_id']}")
            self.verify_run(run, entry)
            return run
        pages = self.github.api(
            f"actions/workflows/{WORKFLOW}/runs?event=workflow_dispatch&branch={BRANCH}&per_page=100&created=%3E%3D{entry['intent_at']}",
            paginate=True,
        )
        title = f"Pants CPU {entry['replay_token']} source {entry['source_sha']}"
        candidates = [run for page in pages for run in page["workflow_runs"] if run.get("display_title") == title]
        if len(candidates) > 1:
            raise ValueError("Duplicate runs match the persisted replay token; refusing ambiguous recovery")
        if not candidates:
            return None
        self.verify_run(candidates[0], entry)
        entry["run_id"], entry["status"] = candidates[0]["id"], "running"
        entry.pop("reconciliation_error", None)
        entry["reconciled_at"] = _now()
        self.save()
        return candidates[0]

    def retain(self, run: dict, entry: dict) -> dict:
        directory = self.output / "hosted" / str(run["id"])
        directory.mkdir(parents=True, exist_ok=True)
        _write(directory / "run.json", run)
        pages = self.github.api(f"actions/runs/{run['id']}/attempts/1/jobs?per_page=100", paginate=True)
        _write(directory / "jobs.json", pages)
        jobs = [job for page in pages for job in page["jobs"]]
        execution, cache_lines = [], {}
        for job in jobs:
            assigned = bool(job.get("runner_id"))
            seconds = (
                max(
                    0,
                    (
                        datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00"))
                        - datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
                    ).total_seconds(),
                )
                if assigned and job.get("started_at") and job.get("completed_at")
                else 0
            )
            execution.append(
                {
                    "id": job["id"],
                    "name": job["name"],
                    "assigned": assigned,
                    "execution_seconds": seconds,
                    "conclusion": job.get("conclusion"),
                }
            )
            if assigned:
                log = directory / f"job-{job['id']}.log"
                if not log.exists():
                    temporary = log.with_suffix(".partial")
                    self.github.download(f"actions/jobs/{job['id']}/logs", temporary)
                    temporary.replace(log)
                cache_lines[str(job["id"])] = [
                    line.rstrip()
                    for line in log.open(errors="replace")
                    if re.search(
                        r"cache (hit|miss|restored|saved)|Cache restored|Cache not found|cache-hit", line, re.I
                    )
                ][:200]
        _write(directory / "execution.json", execution)
        _write(directory / "cache-observations.json", cache_lines)
        pages = self.github.api(f"actions/runs/{run['id']}/artifacts?per_page=100", paginate=True)
        _write(directory / "artifact-inventory.json", pages)
        for artifact in [item for page in pages for item in page["artifacts"]]:
            name = artifact["name"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
                raise ValueError("Unsafe artifact directory name")
            target = directory / "artifacts" / name
            if (target / ".download-complete.json").exists():
                continue
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                archive = Path(temporary) / "artifact.zip"
                extracted = Path(temporary) / "files"
                self.github.download(f"actions/artifacts/{artifact['id']}/zip", archive)
                with zipfile.ZipFile(archive) as contents:
                    for member in contents.infolist():
                        if (
                            not (extracted / member.filename).resolve().is_relative_to(extracted.resolve())
                            or (member.external_attr >> 16) & 0o170000 == 0o120000
                        ):
                            raise ValueError("Unsafe artifact member")
                    contents.extractall(extracted)
                _write(extracted / ".download-complete.json", {"artifact_id": artifact["id"]})
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    shutil.rmtree(target)
                extracted.replace(target)
        errors = []
        try:
            route = json.loads((directory / "artifacts/full-ci-plan/routing.json").read_text())
            expected = {
                "source_sha": entry["source_sha"],
                "experiment_sha": self.state["controller_sha"],
                "replay_token": entry["replay_token"],
                "cache_namespace": self.state["cache_namespace"],
                "execution_mode": "incremental",
            }
            if any(route.get(key) != value for key, value in expected.items()) or route.get("force_full") is not True:
                errors.append("Final routing differs from requested immutable full comparison")
            comparison = json.loads((directory / "artifacts/full-ci-comparison/comparison/results.json").read_text())
            if not comparison.get("valid") or comparison.get("routing") != route:
                errors.append("Full comparison report is invalid or belongs to another route")
        except (OSError, ValueError) as error:
            errors.append(f"Missing or unreadable final comparison evidence: {error}")
        if run.get("conclusion") != "success":
            errors.append(f"Workflow concluded {run.get('conclusion')}")
        outcome = {
            "valid": not errors,
            "errors": errors,
            "run_id": run["id"],
            "conclusion": run.get("conclusion"),
            "timing_claim": "No speed ratios computed; inspect per-job cache observations and the full comparison report.",
        }
        _write(directory / "outcome.json", outcome)
        self.caches(directory / "cache-after", entry["cache_contract"])
        return outcome

    def transition(self) -> bool:
        """Perform one durable transition; return false when finished, blocked or past cutoff."""
        if self.state["current"] >= len(self.state["entries"]):
            return False
        entry = self.state["entries"][self.state["current"]]
        if entry["status"] == "failed":
            return False
        if entry["status"] == "pending":
            if _now() >= self.state["cutoff"]:
                return False
            contract = self.guard()
            readiness = self.readiness()
            self.state["readiness"] = dict(readiness, checked_at=_now())
            self.save()
            if not readiness["ready"]:
                return False
            preflight = self.preflight(entry)
            entry_directory = self.output / "entries" / f"{self.state['current']:03d}"
            _write(entry_directory / "preflight.json", preflight)
            self.caches(entry_directory / "cache-before", contract)
            if _now() >= self.state["cutoff"]:
                return False
            if self.guard() != contract:
                raise ValueError("Controller cache contract changed during preflight")
            intent_at = _now()
            entry.update(
                status="dispatch-intent",
                replay_token=f"replay-{self.state['plan_id'][:12]}-{self.state['current']:03d}-{uuid.uuid4().hex[:12]}",
                intent_at=intent_at,
                reconcile_until=(
                    datetime.fromisoformat(intent_at.replace("Z", "+00:00"))
                    + timedelta(minutes=RECONCILIATION_MINUTES)
                )
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                cache_contract=contract,
            )
            if readiness["state"] == "unknown":
                self.state["unknown_readiness_dispatches"] = self.state.get("unknown_readiness_dispatches", 0) + 1
            self.save()  # A crash from here onward must reconcile, never repeat POST.
            try:
                self.github.api(
                    f"actions/workflows/{WORKFLOW}/dispatches",
                    payload={
                        "ref": BRANCH,
                        "inputs": {
                            "source_sha": entry["source_sha"],
                            "force_full": "true",
                            "execution_mode": "incremental",
                            "cache_namespace": self.state["cache_namespace"],
                            "replay_token": entry["replay_token"],
                        },
                    },
                )
            except Exception as error:
                entry["dispatch_error"] = str(error)
                self.save()
                raise
            entry["status"] = "awaiting-run"
            self.save()
            return True
        run = self.recover(entry)
        if run is None:
            # Older persisted intents also acquire a bounded deadline, based on
            # their original intent time rather than each restart's clock.
            deadline = entry.setdefault(
                "reconcile_until",
                (
                    datetime.fromisoformat(entry["intent_at"].replace("Z", "+00:00"))
                    + timedelta(minutes=RECONCILIATION_MINUTES)
                )
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
            )
            if _now() >= deadline:
                entry.update(
                    status="dispatch-unresolved",
                    reconciliation_error=f"No matching run appeared by {deadline} for token {entry['replay_token']}. The dispatch intent is retained and will never be resent automatically. Inspect GitHub and resume this same replay state to recover a late run.",
                )
                self.save()
                return False
            self.save()
            return True
        if run.get("status") != "completed":
            return True
        outcome = self.retain(run, entry)
        entry.update(status="passed" if outcome["valid"] else "failed", outcome=outcome, completed_at=_now())
        if outcome["valid"]:
            self.state["current"] += 1
        self.save()
        return outcome["valid"]

    def accept_failure(self, run_id: int, reason: str) -> None:
        if not reason.strip() or self.state["current"] >= len(self.state["entries"]):
            raise ValueError("Failure acceptance needs a reason and a current failed entry")
        entry = self.state["entries"][self.state["current"]]
        if entry["status"] != "failed" or entry.get("run_id") != run_id:
            raise ValueError("Failure acceptance does not match the blocked run")
        run = self.recover(entry)
        if run is None or run.get("status") != "completed" or run.get("conclusion") != entry["outcome"]["conclusion"]:
            raise ValueError("Run terminal state changed; cannot accept this failure")
        entry.update(status="accepted-failure", accepted_at=_now(), acceptance_reason=reason)
        self.state["current"] += 1
        self.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller-sha")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--accept-failed", type=int)
    parser.add_argument("--reason")
    args = parser.parse_args()
    if bool(args.accept_failed) != bool(args.reason):
        parser.error("--accept-failed and --reason must be supplied together")
    repo = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(repo):
        parser.error("--output must live outside the controller checkout")
    with driver_lock(args.output):
        driver = Driver(
            repo=repo,
            plan=args.plan,
            output=args.output,
            controller_sha=args.controller_sha,
        )
        if args.accept_failed:
            driver.accept_failure(args.accept_failed, args.reason)
            if args.once:
                return
        while True:
            previous = driver.state["current"]
            if not driver.transition() or args.once:
                break
            if driver.state["current"] == previous:
                time.sleep(45)
        if driver.state["current"] < len(driver.state["entries"]) and driver.state["entries"][driver.state["current"]][
            "status"
        ] in {"failed", "dispatch-unresolved"}:
            entry = driver.state["entries"][driver.state["current"]]
            raise SystemExit(
                entry.get("reconciliation_error")
                or f"Replay stopped at failed run {entry.get('run_id')}; retained evidence requires explicit investigation and acceptance."
            )


if __name__ == "__main__":
    main()
