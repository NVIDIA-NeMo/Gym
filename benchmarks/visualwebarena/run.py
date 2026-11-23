# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the maintained VisualWebArena population with verified clean-state epochs."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from nemo_gym.web import WebTask
from resources_servers.webarena_browser.shared_state import STATE_EPOCHS, STATE_INTENTS, validate_state_identity


POPULATION_INTENT_SHA256 = (
    "319a0286faba7bce20a9454f474b3186c597978184f4d10e143d2041f79efd85"  # pragma: allowlist secret
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("site-control requests must not redirect")


class _WorkersNotIdleError(RuntimeError):
    """Resource workers have not yet proved that collection has quiesced."""


class SiteControl:
    """Reset one owned site deployment and verify all declared browser workers are idle."""

    def __init__(
        self,
        *,
        reset_url: str,
        status_url: str,
        health_urls: list[str],
        token_file: Path | None,
        ca_cert: Path | None = None,
        timeout_seconds: float = 1500,
        poll_seconds: float = 2,
    ) -> None:
        reset, status = urlsplit(reset_url), urlsplit(status_url)
        if (reset.scheme, reset.netloc) != (status.scheme, status.netloc):
            raise ValueError("reset and status endpoints must have the same origin")
        if reset.scheme != "https" and not (
            reset.scheme == "http" and reset.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError("site reset requires HTTPS, except for loopback fixtures")
        if not health_urls or not all(math.isfinite(value) and value > 0 for value in (timeout_seconds, poll_seconds)):
            raise ValueError("declare every resource health endpoint and positive reset timeouts")
        self.reset_url, self.status_url, self.health_urls = reset_url, status_url, health_urls
        token = token_file.read_text().strip() if token_file else ""
        if reset.scheme == "https" and not token:
            raise ValueError("a nonempty reset-token-file is required for the site-control service")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=ca_cert)),
        )
        self.timeout_seconds, self.poll_seconds = timeout_seconds, poll_seconds

    def require_idle(self) -> list[dict[str, Any]]:
        observations = []
        for url in self.health_urls:
            with self._opener.open(url, timeout=20) as response:
                health = json.load(response)
            counts = [health.get(key) for key in ("sessions", "creating", "cleanup_pending", "late_release_pending")]
            counts.append((health.get("browser_provider") or {}).get("active_leases"))
            if health.get("status") != "ok" or any(type(count) is not int or count != 0 for count in counts):
                raise _WorkersNotIdleError("site reset requires all declared resource workers to be healthy and idle")
            observations.append(health)
        return observations

    def reset(self) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        observed_reset = False
        try:
            with self._opener.open(
                urllib.request.Request(self.reset_url, headers=self._headers), timeout=min(20, self.timeout_seconds)
            ) as response:
                if response.status not in {200, 202}:
                    raise RuntimeError(f"site reset rejected with HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"site reset rejected with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            # A lost response cannot establish whether reset started. Poll its
            # status; never issue a second potentially destructive trigger.
            pass
        while time.monotonic() < deadline:
            try:
                remaining = deadline - time.monotonic()
                with self._opener.open(
                    urllib.request.Request(self.status_url, headers=self._headers), timeout=min(20, remaining)
                ) as response:
                    status = response.read().decode("utf-8", errors="replace")
                if "Ready for duty!" in status:
                    if observed_reset:
                        return
                    # A 200/202 only acknowledges the trigger. The service
                    # may still be reporting the previous deployment's ready
                    # state before its asynchronous reset starts.
                elif "Reset ongoing" in status:
                    observed_reset = True
                else:
                    raise RuntimeError("site-control service returned an unrecognized reset state")
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"site reset status failed with HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))
        raise TimeoutError("site reset did not prove a reset transition and readiness before its deadline")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def _population(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    by_id = {}
    for row in rows:
        task = WebTask.model_validate(row["web_task"])
        if task.benchmark != "visualwebarena" or task.task_id in by_id:
            raise ValueError("clean-state input must contain unique VisualWebArena task IDs")
        if task.task_kwargs.get("shared_state") is not None:
            raise ValueError("clean-state input must not reuse a previous run's epoch context")
        by_id[task.task_id] = row
    if set(by_id) != {str(index) for index in range(908)}:
        raise ValueError("clean-state policy requires the complete maintained 908-task input")
    for task_id in STATE_INTENTS:
        validate_state_identity(WebTask.model_validate(by_id[task_id]["web_task"]))
    policy = json.loads((Path(__file__).with_name("state_policy.json")).read_text())
    identities = [(str(index), by_id[str(index)]["web_task"]["intent"]) for index in range(908)]
    actual = hashlib.sha256(json.dumps(identities, ensure_ascii=True).encode()).hexdigest()
    if actual != POPULATION_INTENT_SHA256:
        raise ValueError("population intent identity does not match the maintained phase policy")
    return by_id, set(policy["read_only_task_ids"])


def _check_results(
    path: Path,
    expected: set[str],
    *,
    run_id: str,
    epoch: int,
    state_root: Path | None = None,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    seen = set()
    for row in rows:
        task_id = str(row.get("task_id"))
        verifier = row.get("verifier_result") or {}
        reward = row.get("reward")
        if (
            task_id in seen
            or task_id not in expected
            or row.get("benchmark") != "visualwebarena"
            or row.get("mask_sample") is not False
            or row.get("failure_kind")
            or row.get("cleanup_failure_kind")
            or verifier.get("valid_sample") is not True
            or type(reward) not in {int, float}
            or not math.isfinite(reward)
            or not 0 <= reward <= 1
        ):
            raise RuntimeError("phase contains duplicate, invalid, masked or inconclusive rollout results")
        if epoch:
            receipt = (verifier.get("metadata") or {}).get("shared_state") or {}
            if (receipt.get("run_id"), receipt.get("epoch"), receipt.get("task_id")) != (run_id, epoch, task_id):
                raise RuntimeError("protected rollout is missing its verified initial-state receipt")
            site = "classifieds" if task_id in {"678", "751"} else "shopping"
            if state_root is None or receipt.get("site") != site:
                raise RuntimeError("protected rollout receipt has no matching state evidence")
            evidence_dir = state_root / run_id
            recorded = json.loads((evidence_dir / f"epoch-{epoch}-task-{task_id}.json").read_text())
            baseline = json.loads((evidence_dir / f"baseline-{site}.json").read_text())
            digest = hashlib.sha256(json.dumps(baseline, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
            if recorded != receipt or receipt.get("state_sha256") != digest:
                raise RuntimeError("protected rollout receipt disagrees with the retained baseline evidence")
        seen.add(task_id)
    if not allow_missing and seen != expected:
        raise RuntimeError("phase output has missing or extra task IDs")
    return rows


def _collect_phase(
    phase_dir: Path,
    command: list[str],
    expected: set[str],
    *,
    run_id: str,
    epoch: int,
    state_root: Path,
    control: SiteControl,
    max_attempts: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Resume native collector gaps under the same lease, retaining every valid row."""
    deadline = time.monotonic() + timeout_seconds
    retained: dict[str, str] = {}
    for attempt in range(1, max_attempts + 1):
        invocation = list(command)
        if attempt > 1:
            if not (phase_dir / "rollouts_manifest.json").is_file():
                raise RuntimeError("cannot resume phase without its native Gym manifest")
            cleanup_deadline = min(deadline, time.monotonic() + control.timeout_seconds)
            while True:
                try:
                    idle = control.require_idle()
                    break
                except urllib.error.HTTPError:
                    raise
                except (_WorkersNotIdleError, urllib.error.URLError, TimeoutError, ConnectionError):
                    remaining = cleanup_deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    time.sleep(min(control.poll_seconds, remaining))
            _write_json(phase_dir / f"idle-before-retry-{attempt}.json", idle)
            invocation.extend(["--resume", "+resubmit=true"])
        suffix = "" if attempt == 1 else f"-retry-{attempt}"
        _write_json(phase_dir / f"command{suffix}.json", invocation)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("phase collection deadline reached")
        with (phase_dir / f"collector{suffix}.log").open("w") as log:
            subprocess.run(invocation, check=True, timeout=remaining, stdout=log, stderr=subprocess.STDOUT)
        rows = _check_results(
            phase_dir / "rollouts.jsonl",
            expected,
            run_id=run_id,
            epoch=epoch,
            state_root=state_root,
            allow_missing=True,
        )
        observed = {str(row["task_id"]) for row in rows}
        if retained or observed != expected:
            current = {
                str(row["task_id"]): hashlib.sha256(
                    json.dumps(row, sort_keys=True, ensure_ascii=True).encode()
                ).hexdigest()
                for row in rows
            }
            if any(current.get(task_id) != digest for task_id, digest in retained.items()):
                raise RuntimeError("phase resume changed or removed a previously valid rollout")
            retained = current
        if observed == expected:
            if epoch:
                _write_json(phase_dir / "idle-after.json", control.require_idle())
            return rows
        _write_json(
            phase_dir / f"incomplete-attempt-{attempt}.json",
            {
                "missing_task_ids": sorted(expected - observed, key=int),
                "retained_full_row_sha256": retained,
            },
        )
    raise RuntimeError("phase output has missing or extra task IDs after bounded native resumes")


def run(
    *,
    config: Path,
    input_path: Path,
    output_dir: Path,
    state_root: Path,
    control: SiteControl,
    task_ids: set[str] | None = None,
    concurrency: int = 1,
    phase_timeout_seconds: float = 86400,
    phase_max_attempts: int = 3,
    stage: Literal["all", "read-only", "mutating", "protected"] = "all",
) -> Path:
    """Collect normal waves then protected epochs, retaining all native Gym artifacts."""
    by_id, read_only = _population(_read_jsonl(input_path))
    if not config.is_file():
        raise FileNotFoundError(f"Gym configuration does not exist: {config}")
    selected = set(by_id) if task_ids is None else task_ids
    if not selected or not selected <= set(by_id) or concurrency < 1 or phase_max_attempts < 1:
        raise ValueError("select existing task IDs, positive concurrency and positive phase_max_attempts")
    state_root = state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    run_id = uuid4().hex
    run_dir = state_root / run_id
    run_dir.mkdir(mode=0o700)
    normal = selected - STATE_INTENTS.keys()
    phases = [("read-only", 0, normal & read_only), ("mutating", 0, normal - read_only)]
    phases.extend((f"epoch-{epoch}", epoch, selected & set(ids)) for epoch, ids in enumerate(STATE_EPOCHS, 1))
    phases = [(name, epoch, ids) for name, epoch, ids in phases if ids]
    if stage != "all":
        phases = [
            (name, epoch, ids) for name, epoch, ids in phases if (epoch > 0 if stage == "protected" else name == stage)
        ]
        selected = set().union(*(ids for _, _, ids in phases))
        if not selected:
            raise ValueError("selected stage contains no requested tasks")
    manifest = {
        "protocol": "visualwebarena-clean-state-v1",
        "stage": stage,
        "requested_concurrency": concurrency,
        "phase_max_attempts": phase_max_attempts,
        "run_id": run_id,
        "tasks": sorted(selected, key=int),
        "config": str(config.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "phases": [{"name": name, "epoch": epoch, "task_ids": sorted(ids, key=int)} for name, epoch, ids in phases],
    }
    _write_json(output_dir / "protocol.json", manifest)
    poison = state_root / f"requires-recovery-{run_id}.json"
    with (state_root / "site.lease").open("a+", encoding="utf-8") as lease:
        lease_mode = None

        def transition(epoch: int, phase: str, ready: bool = False) -> None:
            lease.seek(0)
            lease.truncate()
            json.dump({"run_id": run_id, "epoch": epoch, "phase": phase, "reset_verified": ready}, lease)
            lease.flush()
            os.fsync(lease.fileno())

        results = []
        collecting = reset_pending = False
        try:
            for name, epoch, ids in phases:
                initial_phase = lease_mode is None
                requested_mode = fcntl.LOCK_EX if epoch or initial_phase else fcntl.LOCK_SH
                if requested_mode != lease_mode:
                    if lease_mode is not None:
                        fcntl.flock(lease, fcntl.LOCK_UN)
                    lease_mode = None
                    fcntl.flock(lease, requested_mode)
                    lease_mode = requested_mode
                if any(state_root.glob("requires-recovery-*.json")):
                    raise RuntimeError(
                        "the site has an unresolved cleanup failure; reconcile its recovery records first"
                    )
                phase_dir = output_dir / name
                phase_dir.mkdir()
                if epoch or initial_phase:
                    _write_json(phase_dir / "idle-before.json", control.require_idle())
                    transition(epoch, "resetting")
                    reset_pending = True
                    control.reset()
                    reset_pending = False
                    transition(epoch, "running", True)
                    if not epoch:
                        # Lock conversion need not be atomic on every shared
                        # filesystem. Validate ownership again under the shared
                        # lease so an intervening reset can never be overlooked.
                        fcntl.flock(lease, fcntl.LOCK_SH)
                        lease_mode = fcntl.LOCK_SH
                        lease.seek(0)
                        if json.load(lease).get("run_id") != run_id:
                            raise RuntimeError("site reset ownership changed before ordinary collection")
                inputs = []
                for task_id in sorted(ids, key=int):
                    row = copy.deepcopy(by_id[task_id])
                    row["web_task"]["mutation_class"] = "read_only" if name == "read-only" else "state_changing"
                    if epoch:
                        row["web_task"].setdefault("task_kwargs", {})["shared_state"] = {
                            "run_id": run_id,
                            "epoch": epoch,
                        }
                    inputs.append(row)
                phase_input, phase_output = phase_dir / "input.jsonl", phase_dir / "rollouts.jsonl"
                phase_input.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in inputs))
                command = [
                    "gym",
                    "eval",
                    "run",
                    "--no-serve",
                    "--config",
                    str(config.resolve()),
                    "--input",
                    str(phase_input.resolve()),
                    "--output",
                    str(phase_output.resolve()),
                    "--num-repeats",
                    "1",
                    "--concurrency",
                    str(min(concurrency, len(ids))),
                ]
                collecting = True
                results.extend(
                    _collect_phase(
                        phase_dir,
                        command,
                        ids,
                        run_id=run_id,
                        epoch=epoch,
                        state_root=state_root,
                        control=control,
                        max_attempts=phase_max_attempts,
                        timeout_seconds=phase_timeout_seconds,
                    )
                )
                collecting = False
            combined = output_dir / "rollouts.jsonl"
            combined.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=True) + "\n"
                    for row in sorted(results, key=lambda r: int(r["task_id"]))
                )
            )
            _check_results(combined, selected, run_id=run_id, epoch=0)
            raw_score = sum(r["reward"] for r in results)
            full_successes = sum(r["reward"] == 1 for r in results)
            _write_json(
                output_dir / "summary.json",
                {
                    "tasks": len(results),
                    "raw_score": raw_score,
                    "raw_rate": raw_score / len(results),
                    "full_successes": full_successes,
                    "full_success_rate": full_successes / len(results),
                    "complete": True,
                    "run_id": run_id,
                },
            )
            if lease_mode == fcntl.LOCK_EX:
                transition(0, "complete")
            return combined
        except BaseException:
            if lease_mode == fcntl.LOCK_EX:
                transition(0, "failed")
            cleanup_unknown = False
            if collecting:
                try:
                    control.require_idle()
                except Exception:
                    cleanup_unknown = True
            if reset_pending or cleanup_unknown:
                _write_json(
                    poison,
                    {
                        "run_id": run_id,
                        "output_dir": str(output_dir.resolve()),
                        "reason": "reset readiness not proven" if reset_pending else "worker cleanup not proven",
                    },
                )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--reset-url", required=True)
    parser.add_argument("--status-url", required=True)
    parser.add_argument("--resource-health-url", action="append", required=True)
    parser.add_argument("--reset-token-file", type=Path)
    parser.add_argument("--ca-cert", type=Path)
    parser.add_argument(
        "--task-ids", help="Comma-separated runtime row indices; input must still contain all 908 rows"
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--phase-max-attempts", type=int, default=3)
    parser.add_argument("--stage", choices=["all", "read-only", "mutating", "protected"], default="all")
    args = parser.parse_args()
    control = SiteControl(
        reset_url=args.reset_url,
        status_url=args.status_url,
        health_urls=args.resource_health_url,
        token_file=args.reset_token_file,
        ca_cert=args.ca_cert,
    )
    run(
        config=args.config,
        input_path=args.input,
        output_dir=args.output_dir,
        state_root=args.state_root,
        control=control,
        task_ids=set(args.task_ids.split(",")) if args.task_ids else None,
        concurrency=args.concurrency,
        phase_max_attempts=args.phase_max_attempts,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
