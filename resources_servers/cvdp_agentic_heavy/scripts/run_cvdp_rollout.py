#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert CVDP and run a NeMo-Gym rollout collection job.

The script is a customer-friendly wrapper around the documented two-terminal
workflow. It checks owner-supplied model environment variables, converts the
official CVDP dataset if needed, starts NeMo-Gym servers, waits for readiness,
collects rollouts, and shuts the servers down.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATHS_ARG = "+config_paths=[resources_servers/cvdp_agentic_heavy/configs/cvdp_agentic_heavy.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
AGENT_NAME = "cvdp_agentic_heavy_agent"
RESOURCE_PREFIX = "cvdp_agentic_heavy.resources_servers.cvdp_agentic_heavy"


def _entrypoint(module: str, function: str) -> list[str]:
    return [sys.executable, "-u", "-c", f"from {module} import {function}; {function}()"]


def _require_model_env() -> None:
    missing = [key for key in ["LLM_BASE_URL", "LLM_API_KEY", "ROLLOUT_MODEL"] if not os.environ.get(key)]
    if missing:
        formatted = ", ".join(missing)
        raise RuntimeError(
            f"Missing required model environment variable(s): {formatted}. "
            "Export them before running a real CVDP rollout."
        )


def _check_backend(backend: str) -> None:
    if backend == "apptainer" and not shutil.which("apptainer"):
        raise RuntimeError("Apptainer was not found on PATH. Install Apptainer or run with --backend docker.")
    if backend == "docker" and not shutil.which("docker"):
        raise RuntimeError("Docker was not found on PATH. Install Docker or run on an Apptainer-capable host.")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _convert_if_needed(args: argparse.Namespace, env: dict[str, str]) -> None:
    output = Path(args.input_jsonl)
    if output.exists() and output.stat().st_size > 0 and not args.force_convert:
        print(f"Using existing converted dataset: {output}")
        return

    cmd = [
        sys.executable,
        "resources_servers/cvdp_agentic_heavy/scripts/convert_to_gym.py",
        "--input",
        args.hf_input,
        "--download-bundles",
        "--repos-cache",
        args.repos_cache,
        "--output",
        args.input_jsonl,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    _run(cmd, cwd=REPO_ROOT, env=env)


def _wait_for_ready(log_path: Path, proc: subprocess.Popen, timeout: int) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            tail = ""
            if log_path.exists():
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            raise RuntimeError(f"ng_run exited before servers were ready. Recent log:\n{tail}")
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            if "All " in text and "servers ready" in text:
                print("NeMo-Gym servers are ready.")
                return
        time.sleep(3)
    raise RuntimeError(f"Timed out waiting {timeout}s for NeMo-Gym servers. See {log_path}")


def _start_servers(args: argparse.Namespace, env: dict[str, str]) -> tuple[subprocess.Popen, Path]:
    log_path = Path(args.server_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _entrypoint("nemo_gym.cli", "run") + [
        CONFIG_PATHS_ARG,
        "+policy_base_url=${oc.env:LLM_BASE_URL}",
        "+policy_api_key=${oc.env:LLM_API_KEY}",
        "+policy_model_name=${oc.env:ROLLOUT_MODEL}",
        f"+{RESOURCE_PREFIX}.execution_backend={args.backend}",
        f"+{RESOURCE_PREFIX}.container_timeout={args.container_timeout}",
    ]
    if args.sif_cache_dir:
        cmd.append(f"+{RESOURCE_PREFIX}.sif_cache_dir={args.sif_cache_dir}")
    if args.harness_workspace_dir:
        cmd.append(f"+{RESOURCE_PREFIX}.harness_workspace_dir={args.harness_workspace_dir}")

    print("+ " + " ".join(shlex.quote(part) for part in cmd) + f" > {log_path} 2>&1")
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    log_file.close()
    return proc, log_path


def _collect(args: argparse.Namespace, env: dict[str, str]) -> None:
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    cmd = _entrypoint("nemo_gym.rollout_collection", "collect_rollouts") + [
        f"+agent_name={AGENT_NAME}",
        CONFIG_PATHS_ARG,
        "+policy_base_url=${oc.env:LLM_BASE_URL}",
        "+policy_api_key=${oc.env:LLM_API_KEY}",
        "+policy_model_name=${oc.env:ROLLOUT_MODEL}",
        f"+input_jsonl_fpath={args.input_jsonl}",
        f"+output_jsonl_fpath={args.output_jsonl}",
        f"+num_repeats={args.num_repeats}",
        f"+num_samples_in_parallel={args.num_samples_in_parallel}",
    ]
    if args.limit:
        cmd.append(f"+limit={args.limit}")
    _run(cmd, cwd=REPO_ROOT, env=env)


def _stop_servers(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["apptainer", "docker"], default=os.environ.get("CVDP_BACKEND", "apptainer"))
    parser.add_argument("--input-jsonl", default="resources_servers/cvdp_agentic_heavy/data/gym_cvdp_v1.1.0_agentic_heavy_code_generation.jsonl")
    parser.add_argument("--output-jsonl", default="results/cvdp_rollouts.jsonl")
    parser.add_argument("--repos-cache", default=os.environ.get("CVDP_REPOS_CACHE", ".cache/cvdp_repos_cache"))
    parser.add_argument("--hf-input", default="hf://cvdp_v1.1.0_agentic_heavy_code_generation.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Optional dataset/rollout limit for smoke runs.")
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--num-samples-in-parallel", type=int, default=1)
    parser.add_argument("--container-timeout", type=int, default=600)
    parser.add_argument("--server-ready-timeout", type=int, default=600)
    parser.add_argument("--server-log", default="results/cvdp_ng_run.log")
    parser.add_argument("--sif-cache-dir", default=os.environ.get("CVDP_SIF_CACHE_DIR", ""))
    parser.add_argument("--harness-workspace-dir", default=os.environ.get("CVDP_HARNESS_WORKSPACE_DIR", ""))
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate prerequisites and conversion, but do not start rollout servers.")
    args = parser.parse_args()

    env = os.environ.copy()
    _require_model_env()
    _check_backend(args.backend)
    _convert_if_needed(args, env)

    if args.check_only:
        print("preflight=passed")
        return

    proc: subprocess.Popen | None = None
    try:
        proc, log_path = _start_servers(args, env)
        _wait_for_ready(log_path, proc, args.server_ready_timeout)
        _collect(args, env)
    finally:
        _stop_servers(proc)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
