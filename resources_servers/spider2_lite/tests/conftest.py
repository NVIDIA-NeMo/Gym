# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

from resources_servers.spider2_lite.setup_spider2 import ensure_spider2_lite

_VLLM_BIN = shutil.which("vllm") or "/home/rlempka/code/.venv/bin/vllm"
_DEFAULT_MODEL = "openai/gpt-oss-20b"
_DEFAULT_PORT = 18765


def pytest_configure(config):
    ensure_spider2_lite()


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


@pytest.fixture(scope="session")
def vllm_url():
    """Start a vLLM server (or use one already running) and yield its base URL."""
    if url := os.environ.get("SPIDER2_LLM_URL"):
        yield url
        return

    if not Path(_VLLM_BIN).is_file():
        pytest.skip(f"vllm binary not found at {_VLLM_BIN}")

    model = os.environ.get("SPIDER2_LLM_MODEL", _DEFAULT_MODEL)
    port = int(os.environ.get("SPIDER2_LLM_PORT", _DEFAULT_PORT))
    gpu = os.environ.get("SPIDER2_LLM_GPU", "0")
    base_url = f"http://localhost:{port}/v1"

    if _is_port_open("localhost", port):
        yield base_url
        return

    cmd = [
        _VLLM_BIN, "serve", model,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--trust-remote-code",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    deadline = time.time() + 240
    while time.time() < deadline:
        if _is_port_open("localhost", port):
            try:
                r = requests.get(f"http://localhost:{port}/health", timeout=2)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            pytest.fail(f"vLLM exited early. Output:\n{out[-3000:]}")
        time.sleep(3)
    else:
        proc.terminate()
        pytest.fail("vLLM did not become ready within 240s")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def llm_model(vllm_url):
    """Return the model name registered with the running vLLM server."""
    r = requests.get(f"{vllm_url}/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]
