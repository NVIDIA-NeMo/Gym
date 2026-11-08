# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run prepared gold/empty/timeout controls through SWE HTTP routes and Harbor's oracle CLI.

Each resource control runs the real FastAPI app in-process with its normal cookies and
lifecycle, and uses the real configured sandbox provider. It applies the patch in the
seeded task sandbox, then calls /verify, including normal worktree capture and retries.
This does not exercise an agent /run, model generation, or training. Harbor runs through
gym dataset validate. Every subprocess has its own provider clients and output directory.
"""

import argparse
import concurrent.futures
import functools
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf
from scripts.prepare_swe_verifier_controls import CASES, FORCED_TIMEOUT_MARKER, ROOT, clean_revision, digest
from scripts.swe_rows_to_harbor.convert import task_folder_name


def write_json(path: Path, value: object) -> None:
    """Write evidence with environment-secret values removed if an SDK included one in an error."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def sanitized(node: object) -> object:
        if isinstance(node, str):
            return redact(node)
        if isinstance(node, dict):
            return {key: sanitized(value) for key, value in node.items()}
        if isinstance(node, (list, tuple)):
            return [sanitized(value) for value in node]
        return node

    path.write_text(json.dumps(sanitized(value), indent=2, default=str) + "\n")


def redact(text: str) -> str:
    for name, value in os.environ.items():
        if value and re.search(r"KEY|TOKEN|PASSWORD|SECRET", name, re.IGNORECASE):
            text = text.replace(value, "[REDACTED]")
    return text


def load_provider(path: Path) -> tuple[DictConfig, str]:
    """Keep secrets in environment references; do not persist a resolved provider config."""
    text = path.read_text()
    config = OmegaConf.load(path)
    if set(config) != {"sandbox"}:
        raise ValueError("Provider file must contain only the sandbox block")

    def check(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if re.search(r"api_key|password|token|secret", key, re.IGNORECASE) and value:
                    if not isinstance(value, str) or "${oc.env:" not in value:
                        raise ValueError(f"Provider secret {key} must use an environment reference")
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(OmegaConf.to_container(config, resolve=False))
    defaults = OmegaConf.load(ROOT / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml")
    return OmegaConf.merge(defaults, config), digest(text.encode())


def validate_prepared(prepared: Path) -> dict:
    """Reject changed fixtures, parser sources or code before starting any sandbox."""
    manifest = json.loads((prepared / "controls.json").read_text())
    if (
        clean_revision(ROOT) != manifest["gym_commit"]
        or digest((ROOT / "uv.lock").read_bytes()) != manifest["lock_sha256"]
    ):
        raise ValueError("Prepared controls require the same committed Gym source and lockfile")
    parser_repo = Path(manifest["parser_cache"]) / "SWE-rebench-V2"
    if clean_revision(parser_repo) != manifest["parser_commit"]:
        raise ValueError("Prepared controls require the same frozen parser checkout")
    for name, expected in manifest["files_sha256"].items():
        path = prepared / name
        if not path.resolve().is_relative_to(prepared.resolve()) or digest(path.read_bytes()) != expected:
            raise ValueError(f"Prepared control changed: {name}")
    if [fixture["case"] for fixture in manifest["fixtures"]] != list(CASES):
        raise ValueError("Prepared controls must contain gold, empty and timeout fixtures")
    if (
        len(manifest["tasks"]) != manifest["count"]
        or len({task["instance_id"] for task in manifest["tasks"]}) != manifest["count"]
    ):
        raise ValueError("Prepared controls contain missing or duplicate tasks")
    return manifest


def source_control(folder: Path, provider: Path, route: str, index: int, output: Path) -> None:
    """Exercise one real resource session and leave HTTP, patch and cleanup evidence."""
    from fastapi.testclient import TestClient

    from nemo_gym.client import ServerClient
    from nemo_gym.client.http_client import close_global_aiohttp_client
    from nemo_gym.config import BaseServerConfig
    from nemo_gym.config.global_config import scoped_global_config_dict
    from resources_servers.swe_rebench.app import SWERebenchResourcesServer, SWERebenchResourcesServerConfig
    from resources_servers.swe_rebench.verification import repo_directory

    row = json.loads((folder / "rows.jsonl").read_text().splitlines()[index])
    settings = json.loads((folder / "source-settings.json").read_text())
    settings["verification_log_dir"] = str(output / "verifier")
    provider_config, provider_hash = load_provider(provider)
    session_id, rollout_id = f"control-{uuid4().hex}", f"control-{uuid4().hex}"
    response_stub = {
        "id": "",
        "created_at": 0,
        "model": "",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    evidence = {
        "instance_id": row["instance_id"],
        "route": route,
        "case": folder.name,
        "provider_config_sha256": provider_hash,
        "calls": [],
        "coverage": "real resource API and sandbox; no agent/model",
    }
    with scoped_global_config_dict(provider_config):
        client = ServerClient(
            head_server_config=BaseServerConfig(name="unused-head", host="127.0.0.1", port=1, entrypoint=""),
            global_config_dict=provider_config,
        )
        server = SWERebenchResourcesServer(
            config=SWERebenchResourcesServerConfig(
                **(
                    settings
                    | {
                        "host": "127.0.0.1",
                        "port": 1,
                        "name": "swe_control",
                        "entrypoint": "",
                    }
                )
            ),
            server_client=client,
        )
        with TestClient(server.setup_webserver(), raise_server_exceptions=False) as http:
            sandbox = None

            def post(path, body):
                reply = http.post(path, json=body)
                try:
                    value = reply.json()
                except ValueError:
                    value = {"text": reply.text}
                evidence["calls"].append({"path": path, "status": reply.status_code, "response": value})
                if reply.status_code != 200:
                    raise RuntimeError(f"{path} returned {reply.status_code}; see calls in result.json")
                return value

            try:
                seed = (
                    row
                    if route == "flat"
                    else {
                        "resources_session_id": session_id,
                        "episode_id": {"rollout_id": rollout_id},
                        "task_id": {"taskset": "swe_controls", "task_id": row["instance_id"]},
                        "task_data": row,
                    }
                )
                post("/seed_session", seed)
                if len(server._session_id_to_sandbox) != 1:
                    raise RuntimeError("Expected exactly one owned task sandbox after seed")
                session_id, sandbox = next(iter(server._session_id_to_sandbox.items()))
                if row["patch"]:
                    patch = output / "oracle.patch"
                    patch.write_text(row["patch"])
                    http.portal.call(sandbox.upload, patch, "/tmp/control.patch")
                    result = http.portal.call(
                        functools.partial(
                            sandbox.exec,
                            f"git -C {shlex.quote(repo_directory(row['repo']))} apply --reject --recount "
                            "--ignore-space-change --whitespace=nowarn /tmp/control.patch",
                            timeout_s=120,
                        )
                    )
                    evidence["patch_apply"] = {
                        "return_code": result.return_code,
                        "error_type": result.error_type,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                    if result.error_type is not None or result.return_code != 0:
                        raise RuntimeError("Gold patch could not be applied; see patch_apply")
                evidence["result"] = post("/verify", row | {"response": response_stub})
            except Exception as exc:
                evidence["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    post(
                        "/close_session",
                        {"resources_session_id": session_id, "episode_id": {"rollout_id": rollout_id}},
                    )
                except Exception as exc:
                    evidence["close_error"] = f"{type(exc).__name__}: {exc}"
                http.portal.call(server.shutdown)
                evidence["cleanup"] = {
                    "task_teardown_confirmed": sandbox.teardown_confirmed if sandbox is not None else None,
                    "task_cleanup_errors": list(sandbox.cleanup_errors) if sandbox is not None else [],
                    "owned_task_sandboxes": len(server._session_id_to_sandbox),
                    "owned_verifier_sandboxes": len(server._verifier_sandboxes),
                    "meaning": "Ownership is removed only after provider stop succeeds; not an independent provider inventory.",
                }
                http.portal.call(close_global_aiohttp_client)
        write_json(output / "result.json", evidence)


def run_command(command: list[str], output: Path, *, run_id: str, timeout_s: float) -> int:
    """Run one bounded control, preserving its exact command and sanitized console output."""
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "command.json", {"argv": command, "cwd": str(output), "run_id": run_id})
    env = os.environ | {"NEMO_GYM_RUN_ID": run_id, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    started = time.monotonic()
    with subprocess.Popen(
        command, cwd=output, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
            code = process.returncode
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            # Let normal server shutdown stop sandboxes before forcing process cleanup.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            try:
                stdout, stderr = process.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            code = 124
            if isinstance(exc, KeyboardInterrupt):
                raise
    (output / "stdout.log").write_text(redact(stdout.decode(errors="replace")))
    (output / "stderr.log").write_text(redact(stderr.decode(errors="replace")))
    write_json(output / "process.json", {"return_code": code, "elapsed_seconds": time.monotonic() - started})
    return code


def assess_control(result: dict, case: str, *, source_logs: Path | None = None) -> dict:
    """Keep the injected deadline separate from the cause the runtime can establish."""
    assessment = {"passed": False, "fixture_intent": case, "attempt_evidence": []}
    if case == "timeout":
        masked = (
            result.get("reward") == 0
            and result.get("mask_sample") is True
            and result.get("evaluation_completed") is False
        )
        if not masked:
            return assessment
        attempts = result.get("verification_attempts")
        if isinstance(attempts, int):
            if attempts != 2 or source_logs is None:
                return assessment
            attempts = [{"verifier_logs_dir": str(source_logs / f"attempt-{number}")} for number in (1, 2)]
        if not isinstance(attempts, list) or len(attempts) != 2:
            return assessment
        sandbox_ids = []
        for attempt in attempts:
            logs = Path(attempt["verifier_logs_dir"])
            try:
                grading = json.loads((logs / ("verification.json" if source_logs else "grading.json")).read_text())
                execution = json.loads((logs / "exec_result.json").read_text())
                output = (logs / "test_output.log").read_text()
                sandbox_id = (
                    grading.get("sandbox_id")
                    if source_logs
                    else json.loads((logs / "verifier-cleanup.json").read_text()).get("sandbox_id")
                )
            except (OSError, ValueError):
                return assessment
            typed_timeout = execution.get("error_type") == "timeout" and grading.get("error") == "evaluation timed out"
            interrupted = (
                source_logs is not None
                and execution.get("return_code") == -1
                and execution.get("error_type") is None
                and execution.get("stderr") == "signal: killed"
                and grading.get("error") == "evaluation ended without complete test output (exit -1; cause unknown)"
            )
            assessment["attempt_evidence"].append(
                {
                    "sandbox_id": sandbox_id,
                    "observed_classification": (
                        "typed_timeout"
                        if typed_timeout
                        else "interrupted_test_output"
                        if interrupted
                        else "unclassified"
                    ),
                    "deadline_cause": "confirmed" if typed_timeout else "unavailable",
                    "error": grading.get("error"),
                    "execution": execution,
                }
            )
            if (
                not (typed_timeout or interrupted)
                or not isinstance(sandbox_id, str)
                or not sandbox_id
                or execution.get("timeout_s") != 30
                or execution.get("test_started") is not True
                or execution.get("test_completed") is not False
                or not isinstance(execution.get("exec_elapsed_s"), (int, float))
                or not 0 <= execution["exec_elapsed_s"] < float("inf")
                or FORCED_TIMEOUT_MARKER not in output
            ):
                return assessment
            sandbox_ids.append(sandbox_id)
        assessment["passed"] = len(set(sandbox_ids)) == 2
        return assessment
    assessment["passed"] = (
        result.get("reward") == (1 if case == "gold" else 0)
        and not result.get("mask_sample", False)
        and result.get("evaluation_completed") is True
    )
    return assessment


def harbor_command(
    prepared: Path, provider: Path, target: Path, venv_root: Path, *, case: str, concurrency: int
) -> list[str]:
    """Build the ordinary oracle CLI command with the source control's matched budgets."""
    prefix = "++harbor_environment_server.environment_servers.harbor."
    return [
        str(Path(sys.executable).parent / "gym"),
        "dataset",
        "validate",
        str(prepared / case / "harbor"),
        "--sandbox",
        "opensandbox",
        "--concurrency",
        str(concurrency),
        "--output",
        str(target / "rollouts.jsonl"),
        f"+config_paths=[{json.dumps(str(provider))}]",
        "+resume_from_cache=false",
        "+num_repeats=1",
        "+skip_venv_if_present=true",
        f"+uv_venv_dir={venv_root}",
        f"+results_dir={target / 'results'}",
        prefix + "verifier_completion_key=evaluation_completed",
        prefix + "inconclusive_verification_retries=1",
        prefix + "sandbox_ready_timeout_s=120",
        prefix + "default_episode_timeout_seconds=1200",
        prefix + "cleanup_timeout_seconds=180",
        prefix + "sandbox_ttl_slack_s=420",
        prefix + "enforce_task_timeouts=false",
        prefix + "sandbox_provider_options.platform={os:linux,arch:amd64}",
        prefix + "sandbox_provider_options.resource_requests={cpu:1,memory_mib:4096,disk_gib:30}",
    ]


def run_controls(prepared: Path, provider: Path, output: Path, *, concurrency: int) -> dict:
    """Run the complete matrix without dropping failures or changing its denominator."""
    _, provider_hash = load_provider(provider)
    manifest = validate_prepared(prepared)
    if not 1 <= concurrency <= 8:
        raise ValueError("CPU control concurrency must be between 1 and 8")
    output.mkdir(parents=True, exist_ok=False)
    tasks, cases = manifest["tasks"], manifest["fixtures"]
    summary = {
        "expected_controls": len(tasks) * 9,
        "provider_config_sha256": provider_hash,
        "prepared_manifest_sha256": digest((prepared / "controls.json").read_bytes()),
        "controls": [],
    }
    script = str(Path(__file__).resolve())

    def run_source(job):
        case, route, index = job
        target = output / f"source-{route}" / case / str(index)
        try:
            code = run_command(
                [
                    sys.executable,
                    script,
                    "--source-case",
                    str(prepared / case),
                    "--provider",
                    str(provider),
                    "--route",
                    route,
                    "--index",
                    str(index),
                    "--output",
                    str(target),
                ],
                target,
                run_id=f"swe-control-{uuid4().hex}",
                timeout_s=1800,
            )
            receipt = json.loads((target / "result.json").read_text()) if (target / "result.json").is_file() else {}
        except Exception as exc:
            code, receipt = -1, {"error": f"{type(exc).__name__}: {exc}"}
            write_json(target / "runner-error.json", receipt)
        cleanup = receipt.get("cleanup", {})
        assessment = assess_control(
            receipt.get("result", {}), case, source_logs=target / "verifier" / tasks[index]["instance_id"]
        )
        passed = (
            code == 0
            and assessment["passed"]
            and cleanup.get("task_teardown_confirmed") is True
            and cleanup.get("owned_task_sandboxes") == 0
            and cleanup.get("owned_verifier_sandboxes") == 0
            and "close_error" not in receipt
            and "error" not in receipt
        )
        return {
            **assessment,
            "arm": f"source-{route}",
            "case": case,
            "instance_id": tasks[index]["instance_id"],
            "passed": passed,
            "artifact_dir": str(target),
        }

    jobs = [(case, route, index) for case in CASES for route in ("flat", "typed") for index in range(len(tasks))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        for receipt in executor.map(run_source, jobs):
            summary["controls"].append(receipt)
            write_json(output / "summary.json", summary)
    # Reuse this exact locked interpreter for the ordinary CLI's component processes.
    venv_root = output / "server-envs"
    for component in (
        "environment_servers/harbor",
        "responses_api_agents/harbor_oracle_agent",
        "responses_api_models/dummy_model",
    ):
        link = venv_root / component / ".venv"
        link.parent.mkdir(parents=True)
        link.symlink_to(sys.prefix, target_is_directory=True)
    for fixture in cases:
        case = fixture["case"]
        target = output / "harbor" / case
        command = harbor_command(prepared, provider, target, venv_root, case=case, concurrency=concurrency)
        try:
            code = run_command(command, target, run_id=f"swe-control-{uuid4().hex}", timeout_s=1800 * len(tasks))
        except Exception as exc:
            code = -1
            write_json(target / "runner-error.json", {"error": f"{type(exc).__name__}: {exc}"})
        try:
            rows = [json.loads(line) for line in (target / "rollouts.jsonl").read_text().splitlines()]
        except (OSError, ValueError) as exc:
            rows = []
            write_json(target / "read-error.json", {"error": str(exc)})
        identities = [(row.get("_ng_task_id") or row.get("task_id") or {}).get("task_id") for row in rows]
        expected_ids = {task_folder_name("swe_rebench", task["instance_id"]) for task in tasks}
        valid_population = Counter(identities) == Counter({task_id: 1 for task_id in expected_ids})
        by_id = dict(zip(identities, rows))
        for task in tasks:
            task_id = task_folder_name("swe_rebench", task["instance_id"])
            row = by_id.get(task_id, {})
            # validate may return nonzero for an intentional empty/timeout control; the recorded verdict decides.
            attempts = row.get("verification_attempts", [])
            cleanup_paths = [Path(attempt["verifier_logs_dir"]) / "verifier-cleanup.json" for attempt in attempts]
            if attempts:
                cleanup_paths.append(Path(attempts[0]["verifier_logs_dir"]) / "agent-cleanup.json")
            cleanup = []
            for path in cleanup_paths:
                try:
                    cleanup.append(json.loads(path.read_text()))
                except (OSError, ValueError):
                    cleanup.append({"teardown_confirmed": None, "path": str(path)})
            summary["controls"].append(
                {
                    **(assessment := assess_control(row, case)),
                    "arm": "harbor",
                    "case": case,
                    "instance_id": task["instance_id"],
                    "passed": code in (0, 1)
                    and valid_population
                    and assessment["passed"]
                    and bool(cleanup)
                    and all(item.get("teardown_confirmed") is True for item in cleanup),
                    "cleanup": cleanup,
                    "process_return_code": code,
                    "artifact_dir": str(target),
                    "result_present": bool(row),
                    "population_valid": valid_population,
                }
            )
        write_json(output / "summary.json", summary)
    summary["passed"] = sum(item["passed"] for item in summary["controls"])
    summary["unconfirmed_deadline_causes"] = sum(
        attempt["deadline_cause"] == "unavailable"
        for item in summary["controls"]
        for attempt in item.get("attempt_evidence", [])
    )
    summary["verdict"] = "PASS" if summary["passed"] == summary["expected_controls"] else "FAIL_OR_INCOMPLETE"
    if summary["verdict"] == "PASS" and summary["unconfirmed_deadline_causes"]:
        summary["verdict"] = "PASS_WITH_UNCONFIRMED_DEADLINE_CAUSE"
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--source-case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--route", choices=("flat", "typed"), default="typed", help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.source_case is not None:
        source_control(
            args.source_case.resolve(), args.provider.resolve(), args.route, args.index, args.output.resolve()
        )
    elif args.prepared is not None:
        summary = run_controls(
            args.prepared.resolve(), args.provider.resolve(), args.output.resolve(), concurrency=args.concurrency
        )
        print(
            json.dumps(
                {"verdict": summary["verdict"], "passed": summary["passed"], "expected": summary["expected_controls"]}
            )
        )
        if summary["passed"] != summary["expected_controls"]:
            raise SystemExit(1)
    else:
        parser.error("--prepared is required")


if __name__ == "__main__":
    main()
