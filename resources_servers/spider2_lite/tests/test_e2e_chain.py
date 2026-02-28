# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end chain test for spider2_lite using the canonical NeMo-Gym CLI tools.

Starts a real ng_run process (resource server + agent) via subprocess, then runs
ng_collect_rollouts against example.jsonl, and asserts that all rollouts return
valid rewards with no unknown errors.

This is the canonical NeMo-Gym chain test: it exercises the exact same code path
as a production deployment (ng_run starts isolated server subprocesses; ng_collect_rollouts
queries the HeadServer for addresses and calls the agent /run endpoint over HTTP).

Usage:
    # Run via the server's venv directly (required — avoid `uv run` wrappers which
    # place `uv run` in the ancestor process tree and trigger Ray's uv runtime-env
    # hook, causing ray.init() to fail in the server subprocess).
    resources_servers/spider2_lite/.venv/bin/pytest \
        resources_servers/spider2_lite/tests/test_e2e_chain.py -m e2e_llm -v -s

    # Or set SPIDER2_LLM_URL to skip auto-starting vLLM:
    SPIDER2_LLM_URL=http://localhost:18765/v1 \
        resources_servers/spider2_lite/.venv/bin/pytest \
        resources_servers/spider2_lite/tests/test_e2e_chain.py -m e2e_llm -v -s

Requires: vLLM running (or auto-started by the session-scoped vllm_url fixture).
"""

import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e_llm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # nemo-gym root
_NG_RUN = _REPO_ROOT / ".venv/bin/ng_run"
_NG_COLLECT = _REPO_ROOT / ".venv/bin/ng_collect_rollouts"
_EXAMPLE_JSONL = Path(__file__).parent.parent / "data/example.jsonl"

_CONFIG_PATHS = (
    "resources_servers/spider2_lite/configs/spider2_lite.yaml,"
    "responses_api_models/vllm_model/configs/vllm_model.yaml"
)

_ALL_SERVERS_READY_MARKER = "servers ready!"


def _wait_for_ready(proc: subprocess.Popen, timeout_s: float = 180.0) -> None:
    """
    Read ng_run stdout/stderr until 'servers ready!' appears or the timeout is hit.

    Runs the reader in a daemon thread so readline() never blocks the test indefinitely.
    """
    lines: list[str] = []
    ready_event = threading.Event()

    def _reader():
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode(errors="replace")
            lines.append(line)
            if _ALL_SERVERS_READY_MARKER in line:
                ready_event.set()
                return  # stop consuming; the process keeps running

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ready_event.is_set():
            return
        if proc.poll() is not None:
            raise RuntimeError(f"ng_run exited early (rc={proc.returncode}).\nOutput:\n{''.join(lines[-50:])}")
        time.sleep(0.5)

    raise TimeoutError(f"ng_run did not become ready within {timeout_s}s.\nLast output:\n{''.join(lines[-50:])}")


@pytest.fixture(scope="module")
def ng_run_servers(vllm_url, llm_model):
    """
    Start ng_run with the spider2_lite resource server + vllm_model + simple_agent,
    wait until all servers report ready, then yield.

    ng_run starts a HeadServer (port 11000 by default) plus isolated subprocesses
    for each server — exactly the production topology.
    """
    cmd = [
        str(_NG_RUN),
        f"+config_paths=[{_CONFIG_PATHS}]",
        f"+policy_base_url={vllm_url}",
        "+policy_api_key=token",
        f"+policy_model_name={llm_model}",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # New session so os.killpg tears down ng_run and all its child server processes.
        preexec_fn=os.setsid,
    )

    try:
        _wait_for_ready(proc)
        yield
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_chain_collect_rollouts(ng_run_servers, vllm_url, llm_model, tmp_path):
    """
    Run ng_collect_rollouts end-to-end through the full NeMo-Gym server chain.

    ng_collect_rollouts queries the HeadServer for server addresses, then sends
    each example task to the agent /run endpoint.  Asserts:
    - All rollouts return reward 0.0 or 1.0 (no crashes)
    - No unknown_error (server-side exceptions)
    - No no_sql_extracted (model produced a SQL block)
    """
    output_jsonl = tmp_path / "rollouts.jsonl"
    cmd = [
        str(_NG_COLLECT),
        f"+config_paths=[{_CONFIG_PATHS}]",
        "+agent_name=spider2_lite_simple_agent",
        f"+input_jsonl_fpath={_EXAMPLE_JSONL}",
        f"+output_jsonl_fpath={output_jsonl}",
        f"+policy_base_url={vllm_url}",
        "+policy_api_key=token",
        f"+policy_model_name={llm_model}",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"ng_collect_rollouts failed (rc={result.returncode}).\n"
        f"stdout:\n{result.stdout[-3000:]}\n"
        f"stderr:\n{result.stderr[-1000:]}"
    )

    assert output_jsonl.exists(), "ng_collect_rollouts produced no output file"
    rollouts = [json.loads(line) for line in output_jsonl.read_text().splitlines() if line.strip()]
    assert rollouts, "Output JSONL is empty"

    for r in rollouts:
        instance_id = r.get("instance_id", "?")
        assert r.get("reward") in (0.0, 1.0), f"{instance_id}: invalid reward — {r}"
        assert r.get("failure_reason") != "unknown_error", f"{instance_id}: unknown_error — {r}"
        assert r.get("failure_reason") != "no_sql_extracted", f"{instance_id}: no SQL extracted — {r}"

    n_correct = sum(1 for r in rollouts if r.get("reward") == 1.0)
    print(f"\nChain e2e: {n_correct}/{len(rollouts)} correct")
