# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted lifecycle from https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/browser_worker.py (Apache-2.0).
"""One native browser-use episode, with fresh apps and native final-state grading."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import sys
import traceback
from pathlib import Path

from saas_bench.opencode.core import inventory, measured_score
from saas_bench.opencode.verifier import probe_onlyoffice_api, verifier_diagnostics
from saas_bench.opencode.worker import claim_slot

from responses_api_agents.saas_bench_agent.runtime import configure_cache


def _trace_counts(trajectory: list[dict]) -> tuple[int, int]:
    actions = errors = 0
    for step in trajectory:
        calls = step.get("actions", step.get("action", []))
        if isinstance(calls, list):
            actions += len(calls)
        results = step.get("results", [])
        result_errors = (
            sum(bool(result.get("error")) for result in results if isinstance(result, dict))
            if isinstance(results, list)
            else 0
        )
        errors += result_errors or int(bool(step.get("error")))
    return actions, errors


def execute(config: dict) -> dict:
    configure_cache(config["cache_root"])
    root = Path(config["source_root"]).resolve(strict=True)
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "task_id": config["task_id"],
        "harness_version": "browser-use-0.13.7",
        "harness": "browser_use",
        "status": "error",
        "agent_output": "",
        "trajectory": [],
        "native_trajectory": [],
        "mask_sample": True,
        "failure_kind": "saas_bench:infrastructure",
        "artifacts": str(output),
    }
    slot = None
    sites = []
    try:
        import yaml

        os.chdir(root)
        selected = inventory(root, config.get("tasks_dir", "tasks"), config.get("task_selector", config["task_id"]))
        task = selected[config["task_id"]]
        if task["source_sha256"] != config["source_sha256"]:
            raise ValueError("Prepared task fingerprint differs from bundled source")
        os.environ["SAAS_SLOT_PREFIX"] = "bu_" + output.name[:20]
        os.environ["SAAS_BENCH_TMP"] = str(output / "tmp")
        os.environ["SAAS_ENROOT_STATE_DIR"] = str(output / "enroot-state")
        if os.environ.get("SAAS_NRT_ENROOT", "").lower() in {"1", "true", "yes"}:
            shim_dir = output / "verifier-bin"
            shim_dir.mkdir(exist_ok=True)
            shim = shim_dir / "docker"
            shim.write_text(
                "#!/bin/sh\nexec "
                + shlex.quote(sys.executable)
                + " "
                + shlex.quote(str(root / "scripts/nrt/docker_enroot_shim.py"))
                + ' "$@"\n'
            )
            shim.chmod(0o700)
            os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
        from saas_bench.loader import build_prompt
        from saas_bench.slot import SlotManager
        from saas_bench.verify_runner import run_verify

        apps = yaml.safe_load((root / "saas_bench/apps.yaml").read_text())["apps"]
        sites = task["meta"]["meta_data"]["sites"]
        if set(sites) - set(apps):
            raise ValueError("Unknown application in task")
        hostname = config.get("hostname", "localhost")
        with claim_slot(Path(config["slot_lock_root"]), config["slot_capacity"]) as slot_id:
            slot = SlotManager(apps, slot_id)
            try:
                slot.start_apps(sites, hostname=hostname)
                ports = slot.get_port_map(sites)
                if "onlyoffice" in ports:
                    probe_onlyoffice_api(f"http://{hostname}:{ports['onlyoffice']}")
                prompt, todo, inputs = build_prompt(
                    task, ports, hostname, tasks_root=str(root / config.get("tasks_dir", "tasks"))
                )
                judge_env = {k: os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_JUDGE_MODEL")}
                os.environ["LLM_BASE_URL"] = config["endpoint"]
                os.environ["LLM_API_KEY"] = "EMPTY"
                os.environ["LLM_MODEL"] = config["model_name"]
                os.environ["SAAS_LLM_MAX_TOKENS"] = str(config["output_tokens"])
                os.environ["SAAS_LLM_CONTEXT_WINDOW"] = str(config["context_tokens"])
                from saas_bench.agent import run_task

                actor_error = None
                try:
                    actor = asyncio.run(
                        asyncio.wait_for(
                            run_task(
                                task,
                                config["model_name"],
                                prompt,
                                str(output),
                                max_steps=config["max_steps"],
                                slot_id=slot_id,
                                todo_md=todo,
                                run_idx=0,
                                input_files=inputs,
                                allowed_local_origins=[f"http://{hostname}:{port}" for port in ports.values()],
                            ),
                            timeout=config["agent_timeout"],
                        )
                    )
                except TimeoutError as exc:
                    actor_error = "timeout"
                    actor = {"status": "error", "error": str(exc), "trajectory": []}
                except Exception as exc:
                    actor_error = "agent_error"
                    actor = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "trajectory": []}
                finally:
                    for key, value in judge_env.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                # Grading happens while all app state is still alive, including
                # a legitimate wrong-answer zero or a step-limited rollout.
                verify = run_verify(task, slot_id, ports, hostname, str(output), run_suffix="_r0")
                diagnostics = verifier_diagnostics(verify)
                native = actor.get("trajectory", [])
                actor_failed = actor.get("status") != "completed"
                if actor_failed and actor_error is None:
                    detail = json.dumps({k: actor.get(k) for k in ["error", "error_steps", "stop_reason"]})
                    actor_error = (
                        "context_limit"
                        if "context" in detail.lower()
                        and any(word in detail.lower() for word in ["limit", "overflow", "exceed"])
                        else "agent_error"
                    )
                result.update(actor)
                result.update(
                    native_trajectory=native,
                    verify=verify,
                    verifier_diagnostics=diagnostics,
                    reward=measured_score(verify) if not diagnostics["unavailable"] and not actor_failed else 0.0,
                    mask_sample=diagnostics["unavailable"],
                    failure_kind="saas_bench:verifier"
                    if diagnostics["unavailable"]
                    else "saas_bench:model_ending"
                    if actor_failed
                    else None,
                    status="completed",
                    slot_id=slot_id,
                    port_map=ports,
                )
                tool_calls, tool_errors = _trace_counts(native)
                result["execution_health"] = {
                    "termination": actor_error
                    if actor_failed
                    else "task_incomplete"
                    if verify.get("status") != "PASS"
                    else "final_response",
                    "execution_healthy": not actor_failed,
                    "tool_calls": tool_calls,
                    "tool_errors": tool_errors,
                    "task_success": None if diagnostics["unavailable"] else verify.get("status") == "PASS",
                }
            finally:
                if slot is not None:
                    slot.stop_apps(sites)
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}")
        (output / "worker-error.txt").write_text(traceback.format_exc())
    return result


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())

    def terminate(signum, frame):
        raise KeyboardInterrupt("Episode cancelled")

    signal.signal(signal.SIGTERM, terminate)
    result = execute(config)
    destination = Path(config["output"]) / "result.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
